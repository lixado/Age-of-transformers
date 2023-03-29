from collections import deque
import gym
import os
import torch
import numpy as np

from Agents.decisition_transformer import DecisionTransformer_Agent
from logger import Logger
from functions import CreateVideoFromTempImages, SaveTempImage, NotifyDiscord


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
        
        # save sequences
        length = int(agent.n_positions/3)
        states_sequence = deque(maxlen=length)
        actions_sequence = deque(maxlen=length)
        timesteps_sequence = deque(maxlen=length)
        rewards_sequence = deque(maxlen=length)
        actionIndex = -1 # first acction is default Do nothing
        reward = 0


        while not done and not truncated:
            ticks += 1

            # Save sequence
            states_sequence.append(observation)
            # save action tensor as [0,0,1,0,0,0] depending on which action
            actionTensor = np.zeros(agent.action_space_dim)
            actionTensor[actionIndex] = 1
            actions_sequence.append(actionTensor)
            timesteps_sequence.append(ticks)
            rewards_sequence.append(reward)

            # AI choose action
            actionIndex, q_values = agent.act(states_sequence, actions_sequence, timesteps_sequence, rewards_sequence)
            
            # Act
            next_observation, reward, done, truncated, info = gym.step(actionIndex)

            # Record game
            if record:
                SaveTempImage(logger.getSaveFolderPath(), gym.render(q_values), ticks)


            # AI Save memory
            #agent.cache(observation, next_observation, actionIndex, reward, (done or truncated))

            # Learn
            #q, loss = agent.learn()
            
            # Logging
            #logger.log_step(reward, loss, q)

            # Update state
            observation = next_observation

        #logger.log_epoch(e, agent.exploration_rate, agent.optimizer.param_groups[0]["lr"])

        # Record game
        if record:
            CreateVideoFromTempImages(os.path.join(logger.getSaveFolderPath(), "temp"), (e))

    # save model
    #agent.save(logger.getSaveFolderPath())
    NotifyDiscord(f"Training finished. Epochs: {epochs} Name: {logger.getSaveFolderPath()}")
    gym.close()
