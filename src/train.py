import gym
import os
import torch
import numpy as np
from data import Simple1v1Dataset
from torch.utils.data import DataLoader

from Agents.decisition_transformer import DecisionTransformer_Agent
from Agents.ddqn import DDQN_Agent
from logger import Logger
from functions import CreateVideoFromTempImages, SaveTempImage, NotifyDiscord
from data import DTDataset, collate_fn


def get_batches(data: list, batch_size):
    while len(data) % batch_size != 0:
        data.append(data[-1])
    batches = [data[i * batch_size: (i + 1) * batch_size] for i in range(int(len(data) / batch_size))]
    observations = [[element[0].numpy() for element in row] for row in batches]
    actions = [[element[1].numpy() for element in row] for row in batches]
    timesteps = [[element[2].numpy() for element in row] for row in batches]
    rewards = [[element[3].numpy() for element in row] for row in batches]
    return observations, actions, timesteps, rewards


def train_transformer(config: dict, agent: DecisionTransformer_Agent, gym: gym.Env, logger: Logger, data_path):
    save_dir = logger.getSaveFolderPath()
    agent.exploration_rate = 0
    print("Loaded data from", data_path)

    device = config["device"]
    record_epochs = config["recordEvery"]  # record game every x epochs
    epochs = config["epochs"]
    batch_size = config["batchSize"]

    dataset = DTDataset([], agent.action_space_dim, data_path)
    trainingLoader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, num_workers=6)

    for e in range(epochs):
        loss, q = 0.0, 0.0
        agent.net.train()
        for batch in trainingLoader:
            for mini_batch in batch:
                observations = mini_batch[0].to(device)
                actions = mini_batch[1].to(device)
                timesteps = mini_batch[2].to(device)
                rewards = mini_batch[3].to(device)

                loss, q = agent.train(observations, actions, timesteps, rewards)

        agent.net.eval()
        observation, info = gym.reset()
        ticks = 0
        record = (e % record_epochs == 0) or (e == epochs - 1)  # last part to always record last
        if record:
            print("Recording this epoch")
        done = False
        truncated = False

        actionIndex = -1  # first acction is default Do nothing
        reward = config["DTTargetReward"]
        while not done and not truncated:
            ticks += 1

            # AI choose action
            actionIndex, q_values = agent.act(observation, actionIndex, ticks, reward)

            gym.save_player_state()

            # Act
            next_observation, reward, done, truncated, info = gym.step(actionIndex)

            # Record game
            if record:
                SaveTempImage(logger.getSaveFolderPath(), gym.render(q_values, reward), ticks)

            # Logging
            logger.log_step(reward, loss, q)

            # Update state
            observation = next_observation

        logger.log_epoch(e, agent.exploration_rate, agent.optimizer.param_groups[0]["lr"])

        # Record game
        if record:
            CreateVideoFromTempImages(os.path.join(logger.getSaveFolderPath(), "temp"), (e))
            agent.save(save_dir)
    # save model
    agent.save(save_dir)
    NotifyDiscord(f"Training finished. Epochs: {epochs} Name: {save_dir}")
    gym.close()


def train_ddqn(config: dict, agent: DDQN_Agent, gym: gym.Env, logger: Logger):
    agent.net.train()
    save_dir = logger.getSaveFolderPath()

    record_epochs = config["recordEvery"]  # record game every x epochs
    epochs = config["epochs"]
    for e in range(epochs):
        observation, info = gym.reset()
        ticks = 0
        record = (e % record_epochs == 0) or (e == epochs - 1)  # last part to always record last
        if record:
            print("Recording this epoch")
        done = False
        truncated = False
        while not done and not truncated:
            ticks += 1

            # AI choose action
            actionIndex, q_values = agent.act(observation)

            gym.save_player_state()

            # Act
            next_observation, reward, done, truncated, info = gym.step(actionIndex)

            # Record game
            if record:
                SaveTempImage(logger.getSaveFolderPath(), gym.render(q_values, reward), ticks)

            # AI Save memory
            agent.cache(observation, next_observation, actionIndex, reward, (done or truncated))

            # Learn
            q, loss = agent.learn()

            # Logging
            logger.log_step(reward, loss, q)

            # Update state
            observation = next_observation

        logger.log_epoch(e, agent.exploration_rate, agent.optimizer.param_groups[0]["lr"])

        # Record game
        if record:
            CreateVideoFromTempImages(os.path.join(logger.getSaveFolderPath(), "temp"), (e))
            agent.save(logger.getSaveFolderPath())

    # save model
    agent.save(logger.getSaveFolderPath())
    NotifyDiscord(f"Training finished. Epochs: {epochs} Name: {logger.getSaveFolderPath()}")
    gym.close()

