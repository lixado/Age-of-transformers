import os
import sys
from DeepRTS import Engine, Constants
import cv2
import torch
from Gyms.Simple1v1 import Simple1v1Gym
from logger import Logger
from functions import GetConfigDict, CreateVideoFromTempImages, SaveTempImage, NotifyDiscord
from constants import inv_action_space
from Agents.ddqn import DDQN_Agent
from gym.wrappers import FrameStack, TransformObservation, ResizeObservation, GrayScaleObservation

from wrappers import SkipFrame, RepeatFrame

STATE_SHAPE = (84, 84) # model input shapes
FRAME_STACK = 1 #4 # how many frames to stack gets last x frames
SKIP_FRAME = 0#20 # do action and then do nothing for x frames
REPEAT_FRAME = 0 # same action for x frames 
MAP = "10x10-2p-ffa-Eblil.json"


def train(config: dict, agent: DDQN_Agent, gym: Simple1v1Gym):
    agent.net.train()

    record_epochs = 5 # record game every x epochs
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
            actionIndex, q_values = agent.act(observation)
            
            # Act
            next_observation, reward, done, truncated, info = gym.step(actionIndex)

            # Record game
            if record:
                SaveTempImage(logger.getSaveFolderPath(), gym.render(q_values), ticks)

            # use this to see image example
            #cv2.imshow('image', next_observation[0])
            #cv2.waitKey(3)

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
            CreateVideoFromTempImages(os.path.join(logger.getSaveFolderPath(), "temp"), (e))

    # save model
    agent.save()
    NotifyDiscord(f"Training finished. Epochs: {epochs} Name: {logger.getSaveFolderPath()}")
    gym.close()


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
        Handle input default train
        0 = Train
        1 = Eval
        2 = Playground
    """
    mode = 0

    
    """
        Start gym
    """
    gym = Simple1v1Gym(MAP, 0, config["stepsMax"])
    print("Action space: ", [inv_action_space[i] for i in gym.action_space])

    # gym wrappers
    if SKIP_FRAME != 0:
        gym = SkipFrame(gym, SKIP_FRAME)
    if REPEAT_FRAME != 0:
        gym = RepeatFrame(gym, REPEAT_FRAME)
    gym = ResizeObservation(gym, STATE_SHAPE)  # reshape
    gym = GrayScaleObservation(gym)
    gym = TransformObservation(gym, f=lambda x: x / 255.)  # normalize the values [0, 1]
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
    if mode == 0:
        train(config, agent, gym)
