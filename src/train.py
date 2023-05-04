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
    agent.exploration_rate = 0
    save_dir = logger.getSaveFolderPath()

    record_epochs = config["recordEvery"]  # record game every x epochs
    epochs = config["epochs"]
    batch_size = config["batchSize"]

    data = Simple1v1Dataset(data_path)
    gen = torch.Generator().manual_seed(42)
    trainingLoader = DataLoader(data, batch_size=1, shuffle=False)

    for e in range(epochs):
        loss, q = 0.0, 0.0
        agent.net.train()
        for game in trainingLoader:
            batches = get_batches(game, batch_size=batch_size)
            for i in range(len(batches[0])):
                observations = np.array(batches[0][i])
                actions = np.array(batches[1][i])
                timesteps = np.array(batches[2][i])
                rewards = np.array(batches[3][i])

                loss, q = agent.train(observations, actions, timesteps, rewards)

        agent.net.eval()
        observation, info = gym.reset()
        ticks = 1
        record = (e % record_epochs == 0) or (e == epochs - 1)  # last part to always record last
        if record:
            print("Recording this epoch")
        done = False
        truncated = False

        actionIndex = -1  # first acction is default Do nothing
        reward = 10
        agent.states_sequence = [observation]
        agent.rewards_sequence = [reward]
        agent.timesteps_sequence = [ticks]
        actionArr = np.zeros(agent.action_space_dim)
        actionArr[actionIndex] = 1
        agent.actions_sequence = [actionArr]
        while not done and not truncated:
            ticks += 1

            # AI choose action
            actionIndex, q_values = agent.act()

            gym.save_player_state()

            # Act
            next_observation, reward, done, truncated, info = gym.step(actionIndex)

            agent.append(next_observation, actionIndex, ticks, reward)

            # Record game
            if record:
                SaveTempImage(logger.getSaveFolderPath(), gym.render(q_values), ticks)

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

def train_transformer_test(config: dict, agent: DecisionTransformer_Agent, gym: gym.Env, logger: Logger):
    agent.net.train()
    save_dir = logger.getSaveFolderPath()
    # agent.saveHyperParameters(save_dir)

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

        actionIndex = -1  # first acction is default Do nothing
        reward = 0

        while not done and not truncated:
            ticks += 1

            # AI choose action
            actionIndex, q_values = agent.act(observation, actionIndex, ticks, reward)

            gym.save_player_state()

            # Act
            next_observation, reward, done, truncated, info = gym.step(actionIndex)

            # Record game
            if record:
                SaveTempImage(save_dir, gym.render(q_values), ticks)

            # AI Save memory
            # agent.cache(observation, next_observation, actionIndex, reward, (done or truncated))

            # Learn
            loss, q = agent.learn()

            # Logging
            logger.log_step(reward, loss, q)

            # Update state
            observation = next_observation

        logger.log_epoch(e, agent.exploration_rate, agent.optimizer.param_groups[0]["lr"])

        # Record game
        if record:
            CreateVideoFromTempImages(os.path.join(save_dir, "temp"), (e))
            agent.save(save_dir)

    # save model
    agent.save(save_dir)
    NotifyDiscord(f"Training finished. Epochs: {epochs} Name: {save_dir}")
    gym.close()


def train_ddqn(config: dict, agent: DDQN_Agent, gym: gym.Env, logger: Logger):
    agent.net.train()
    save_dir = logger.getSaveFolderPath()
    agent.saveHyperParameters(save_dir)

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
                SaveTempImage(logger.getSaveFolderPath(), gym.render(q_values), ticks)

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

