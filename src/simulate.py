import random
import pickle

import gym
import os

from logger import Logger
from functions import CreateVideoFromTempImages, SaveTempImage


def simulate(config: dict, gym: gym.Env, save_path: str):
    data_path = os.path.join(save_path, 'data')
    os.makedirs(data_path, exist_ok=True) # create temp folder if not exist

    record_epochs = config["recordEvery"] # record game every x epochs
    epochs = config["epochs"]+1528
    for e in range(1528, epochs):
        observation, info = gym.reset()
        ticks = 0
        record = (e % record_epochs == 0) or (e == epochs-1) # last part to always record last
        if record:
            print("Recording this epoch")
        done = False
        truncated = False
        memory = []
        while not done and not truncated:
            ticks += 1

            actionIndex = random.randint(0, len(gym.action_space)-1)

            # Act
            next_observation, reward, done, truncated, info = gym.step(actionIndex)

            # Record game
            if record:
                SaveTempImage(save_path, gym.render(), ticks)

            # save data
            memory.append([observation, next_observation, actionIndex, reward, done])

            # Update state
            observation = next_observation

        # Record game
        if record:
            CreateVideoFromTempImages(os.path.join(save_path, "temp"), (e))
        print(f"Epoch {e} done")

        # save game data
        with open(f"{os.path.join(data_path, f'game_{e}.pickle')}", "wb") as f:
            pickle.dump(memory, f, protocol=pickle.HIGHEST_PROTOCOL)
    gym.close()
