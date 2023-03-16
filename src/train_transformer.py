import gym
import os
import torch.utils.data
from Agents.ddqn import DDQN_Agent
from logger import Logger
from functions import CreateVideoFromTempImages, SaveTempImage, NotifyDiscord
from src.data import AllActions1v1Dataset
from torch.utils.data import DataLoader
import numpy as np

def get_batches(data: list, batch_size):

    while len(data) % batch_size != 0:
        data.append(data[-1])
    batches = np.array([data[i*batch_size : (i+1)*batch_size] for i in range(len(data)/batch_size)])
    return batches


def train(config: dict, agent: DDQN_Agent, gym: gym.Env, logger: Logger, workingDir):
    agent.net.train()
    save_dir = logger.getSaveFolderPath()
    data_dir = os.path.join(workingDir, "data")
    agent.saveHyperParameters(save_dir)

    epochs = config["epochs"]
    batch_size = config["batchSize"]

    data = AllActions1v1Dataset(data_dir)
    trainingData = data[0:int(len(data)*0.8)]
    trainingLoader = DataLoader(trainingData, batch_size=1, shuffle=False)

    validationData = data[int(len(data)*0.8):int(len(data)-len(data)*0.2)]
    validationLoader = DataLoader(validationData, batch_size=1, shuffle=False)
    games=[]
    for e in range(epochs):
        for game in trainingLoader:
            batches = DataLoader(game, batch_size=batch_size)
            for batch in batches:
                print(batch)


    # save model
    agent.save(logger.getSaveFolderPath())
    NotifyDiscord(f"Training finished. Epochs: {epochs} Name: {logger.getSaveFolderPath()}")
    gym.close()
