import gym
import os

from Agents.ddqn import DDQN_Agent
from logger import Logger
from functions import CreateVideoFromTempImages, SaveTempImage, NotifyDiscord


def train(config: dict, agents, gym: gym.Env, logger: Logger):
    agents[0].net.train()
    agents[1].net.train()
    save_dir = logger.getSaveFolderPath()
    agents[0].saveHyperParameters(save_dir)

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
        while not done and not truncated:
            ticks += 1

            # AI choose action
            actionIndex0, q_values0 = agents[0].act(observation)
            actionIndex1, q_values1 = agents[1].act(observation)
            
            # Act
            next_observation, rewards, done, truncated, info = gym.step([actionIndex0, actionIndex1])

            # Record game
            if record:
                SaveTempImage(logger.getSaveFolderPath(), gym.render(q_values0), ticks)

            # AI Save memory
            agents[0].cache(observation, next_observation, actionIndex0, rewards[0], (done or truncated))
            agents[1].cache(observation, next_observation, actionIndex1, rewards[1], (done or truncated))

            # Learn
            q, loss = agents[0].learn()
            _, _ = agents[1].learn()
            
            # Logging
            logger.log_step(rewards[0], loss, q)

            # Update state
            observation = next_observation

        logger.log_epoch(e, agents[0].exploration_rate, agents[0].optimizer.param_groups[0]["lr"])

        # Record game
        if record:
            CreateVideoFromTempImages(os.path.join(logger.getSaveFolderPath(), "temp"), (e))
            agents[0].save(logger.getSaveFolderPath())

    # save model
    agents[0].save(logger.getSaveFolderPath())
    NotifyDiscord(f"Training finished. Epochs: {epochs} Name: {logger.getSaveFolderPath()}")
    gym.close()
