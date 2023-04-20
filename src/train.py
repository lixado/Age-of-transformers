from collections import deque
import gym
import os
import torch
import numpy as np
from data import Simple1v1Dataset
from torch.utils.data import DataLoader

from Agents.decisition_transformer import DecisionTransformer_Agent
from logger import Logger
from functions import CreateVideoFromTempImages, SaveTempImage, NotifyDiscord

def get_batches(data: list, batch_size):
    while len(data) % batch_size != 0:
        data.append(data[-1])
    batches = [data[i*batch_size : (i+1)*batch_size] for i in range(int(len(data)/batch_size))]
    observations = [[element[0] for element in row] for row in batches]
    actions = [[element[1] for element in row] for row in batches]
    timesteps = [[element[2] for element in row] for row in batches]
    rewards = [[element[3] for element in row] for row in batches]
    return observations, actions, timesteps, rewards


def train_transformer(config: dict, agent: DecisionTransformer_Agent, gym: gym.Env, logger: Logger, data_path):
    agent.net.train()
    save_dir = logger.getSaveFolderPath()

    epochs = config["epochs"]
    batch_size = config["batchSize"]

    data = Simple1v1Dataset(data_path)
    gen = torch.Generator().manual_seed(42)
    trainingData, validationData = torch.utils.data.random_split(data, [0.8, 0.2], generator=gen)
    trainingLoader = DataLoader(trainingData, batch_size=1, shuffle=False)

    for e in range(epochs):
        for game in trainingLoader:
            batches = get_batches(game, batch_size=batch_size)
            for i in range(len(batches)):
                observations = batches[0][i]
                actions = batches[1][i]
                timesteps = batches[2][i]
                rewards = batches[3][i]

                agent.act(observations, actions, timesteps, rewards)

                loss, q = agent.learn()

                logger.log_step(0.0, loss, q)

        logger.log_epoch(e, agent.exploration_rate, agent.optimizer.param_groups[0]["lr"])


def train(config: dict, agent: DecisionTransformer_Agent, gym: gym.Env, logger: Logger):
    agent.net.train()
    save_dir = logger.getSaveFolderPath()
    #agent.saveHyperParameters(save_dir)

    record_epochs = config["recordEvery"] # record game every x epochs
    epochs = config["epochs"]
    for e in range(epochs):
        observation, info = gym.reset()
        ticks = 0
        record = (e % record_epochs == 0) or (e == epochs-1) # last part to always record last
        if record:
            print("Recording this epoch")
        done = False
        truncated = False
        
        actionIndex = -1 # first acction is default Do nothing
        reward = 0

        while not done and not truncated:
            ticks += 1


            # AI choose action
            actionIndex, q_values = agent.act(observation, actionIndex, ticks, reward)
            
            # Act
            next_observation, reward, done, truncated, info = gym.step(actionIndex)

            # Record game
            if record:
                SaveTempImage(logger.getSaveFolderPath(), gym.render(q_values), ticks)


            # AI Save memory
            #agent.cache(observation, next_observation, actionIndex, reward, (done or truncated))

            # Learn
            loss, q = agent.learn()

            
            
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
    #agent.save(logger.getSaveFolderPath())
    NotifyDiscord(f"Training finished. Epochs: {epochs} Name: {logger.getSaveFolderPath()}")
    gym.close()
