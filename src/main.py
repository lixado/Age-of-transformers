import os
import sys
from DeepRTS import Engine, Constants
import torch
from Gyms.Simple1v1 import Simple1v1Gym
from logger import Logger
from functions import GetConfigDict, TransformImage, CreateVideoFromTempImages, SaveTempImage
from constants import inv_action_space
from Agents.ddqn import DDQN_Agent
from gym.wrappers import FrameStack, NormalizeObservation, ResizeObservation, GrayScaleObservation

STATE_SHAPE = (84, 84) # model input shapes
FRAME_STACK = 4 # how many frames to stack
MAP = "10x10-2p-ffa-Eblil.json"

if __name__ == "__main__":
    workingDir = os.getcwd()
    if not os.path.exists(os.path.join(workingDir, "src")):
        sys.exit(f'Working directory: {workingDir} not correct, should be "Age-of-transformers/" not "{os.path.basename(os.path.normpath(workingDir))}"')

    config = GetConfigDict(workingDir)
    print("Config: ", config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: ", device)

    logger = Logger(workingDir)

    
    """
        Start gym
    """
    gym = Simple1v1Gym(MAP)
    print("Action space: ", [inv_action_space[i] for i in gym.action_space])

    # gym wrappers
    gym = ResizeObservation(gym, STATE_SHAPE)  # reshape
    gym = GrayScaleObservation(gym)
    gym = NormalizeObservation(gym)  # normalize the values
    gym = FrameStack(gym, num_stack=FRAME_STACK)

    """
        Start agent
    """
    state_sizes = (FRAME_STACK, ) + STATE_SHAPE # number of image stacked
    agent = DDQN_Agent(state_dim=state_sizes, action_space_dim=len(gym.action_space), save_dir=logger.getSaveFolderPath())
    agent.device = device

    """
        Training loop
    """
    record_epochs = int(config["epochs"] / 2) # record game every (total_epochs/x) epochs
    for e in range(config["epochs"]):
        observation, info = gym.reset()
        ticks = 0
        recording_images = []
        record = (e % record_epochs == 0) or (e == config["epochs"]-1) # last part to always record last
        if record:
            print("Recording this epoch")
        done = False
        while not done and ticks < 8000:
            ticks += 1

            # Record game
            if record:
                SaveTempImage(logger.getSaveFolderPath(), gym.render(), ticks)

            # AI choose action
            actionIndex = agent.act(observation)
            
            # Act
            next_observation, reward, done, _, info = gym.step(actionIndex)

            # AI Save memory
            agent.cache(observation, next_observation, actionIndex, reward, done)

            # Learn
            q, loss = agent.learn()
            
            # Logging
            logger.log_step(reward, loss, q)

            # Update state
            observation = next_observation

        logger.log_epoch(e, agent.exploration_rate)

        # Record game
        if record:
            CreateVideoFromTempImages(os.path.join(logger.getSaveFolderPath(), "temp"), (e+1))

    # save model
    agent.save()
    gym.close()