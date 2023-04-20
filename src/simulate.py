import random
import pickle
import gym
import os
from functions import CreateVideoFromTempImages, SaveTempImage


def simulate(config: dict, agent, gym: gym.Env, logger, model_path=None):
    record_epochs = config["recordEvery"] # record game every x epochs
    epochs = config["epochs"]
    save_path = logger.getSaveFolderPath()
    data_path = os.path.join(save_path, "data")
    os.makedirs(data_path, exist_ok=True) # create temp folder if not exist

    if model_path != None:
        agent.loadModel(model_path)
        agent.net.eval()
    for e in range(epochs):
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

            if model_path != None:
                actionIndex, q_values = agent.act(observation)
            else:
                actionIndex, q_values = random.randint(0, len(gym.action_space)-1), []

            # Act
            next_observation, reward, done, truncated, info = gym.step(actionIndex)

            # Record game
            if record:
                SaveTempImage(save_path, gym.render(q_values), ticks)

            logger.log_step(reward, 0, 0)

            # save data
            memory.append([observation[-1], actionIndex, ticks, reward])
            # Update state
            observation = next_observation

        logger.log_epoch(e, agent.exploration_rate, 0)

        # Record game
        if record:
            CreateVideoFromTempImages(os.path.join(save_path, "temp"), (e))
        print(f"Epoch {e} done")

        # save game data
        with open(f"{os.path.join(data_path, f'game_{e:04}.pickle')}", "wb") as f:
            pickle.dump(memory, f, protocol=pickle.HIGHEST_PROTOCOL)
    gym.close()
