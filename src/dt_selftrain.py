import gym
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import random

from tqdm import tqdm

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

class DTDataset(Dataset):
    def __init__(self, data):
        self.games = data

    def __len__(self):
        return len(self.games)

    def __getitem__(self, idx):
        return self.games[idx]


def train_dt_self(config: dict, agent: DecisionTransformer_Agent, gym: gym.Env, logger: Logger):
    save_dir = logger.getSaveFolderPath()

    record_epochs = config["recordEvery"]  # record game every x epochs
    epochs = config["epochs"]
    batch_size = config["batchSize"]
    episodes = 100

    dt_epochs = 50


    for e in range(epochs):
        generated_data = []

        # generate data
        agent.net.eval()
        agent.exploration_rate = 1

        record = (e % record_epochs == 0) or (e == epochs - 1)  # last part to always record last
        if record:
            print("Recording this epoch")

        for epi in range(episodes):
            observation, info = gym.reset()
            ticks = 0

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

            memory = []
            while not done and not truncated:
                ticks += 1

                if e == 0: # full random on first epoch
                    agent.exploration_rate = 1
                
                actionIndex, q_values = agent.act()

                gym.save_player_state()

                # Act
                next_observation, reward, done, truncated, info = gym.step(actionIndex)

                agent.append(next_observation, actionIndex, ticks, reward)

                logger.log_step(reward, 0, 0)

                # save data
                memory.append([observation, actionIndex, ticks, reward])

                # Record game
                if record:
                    SaveTempImage(logger.getSaveFolderPath(), gym.render(q_values, reward), ticks)

                # Update state
                observation = next_observation


            logger.log_epoch((e+1)*(epi+1), agent.exploration_rate, agent.optimizer.param_groups[0]["lr"])
            # Record game
            if record:
                CreateVideoFromTempImages(os.path.join(logger.getSaveFolderPath(), "temp"), (e))
                agent.save(save_dir)
                record = False
            generated_data.append(memory)



        # train agent
        agent.net.train()
        agent.exploration_rate = 0
        dataset = DTDataset(generated_data)
        trainingLoader = DataLoader(dataset, batch_size=1, shuffle=False) # need to make it so can take multiple batches and sequence lengths should vary

        losses = []
        for _ in tqdm(range(dt_epochs)):
            loss, q = 0.0, 0.0
            for game in trainingLoader:
                batches = get_batches(game, batch_size=batch_size)
                for i in range(len(batches[0])):
                    observations = np.array(batches[0][i])
                    actions = np.array(batches[1][i])
                    timesteps = np.array(batches[2][i])
                    rewards = np.array(batches[3][i])

                    loss, q = agent.train(observations, actions, timesteps, rewards)
                    losses.append(loss)

        print(f"Loss avg: {np.mean(losses)} Loss min:  {np.min(losses)} Loss max: {np.max(losses)}")


    # save model
    agent.save(save_dir)
    NotifyDiscord(f"Training finished. Epochs: {epochs} Name: {save_dir}")
    gym.close()