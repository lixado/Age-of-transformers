import os
import sys
import torch
from Gyms.Simple1v1 import Simple1v1Gym
from logger import Logger
from functions import GetConfigDict
from constants import inv_action_space
from Agents.ddqn import DDQN_Agent
from gym.wrappers import FrameStack, NormalizeObservation, ResizeObservation, GrayScaleObservation
from eval import evaluate
from train import train

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
    env = Simple1v1Gym(MAP, 0)
    print("Action space: ", [inv_action_space[i] for i in env.action_space])

    # gym wrappers
    env = ResizeObservation(env, STATE_SHAPE)  # reshape
    env = GrayScaleObservation(env)
    #env = NormalizeObservation(env)  # normalize the values this makes the image impossible to read for humans
    env = FrameStack(env, num_stack=FRAME_STACK)

    """
        Start agent
    """
    state_sizes = (FRAME_STACK, ) + STATE_SHAPE # number of image stacked
    agent = DDQN_Agent(state_dim=state_sizes, action_space_dim=len(env.action_space), save_dir=logger.getSaveFolderPath())
    agent.device = device

    option = input(f"(0) Train\n(1) Evaluate")

    """
        Training loop
    """
    if option == 0:
        train(env, agent, logger, config)
    """
        Evaluation
    """
    if option==1:
        modelPath = "/age-of-transformers/results/20230222-14-14-04/model.chkpt"
        evaluate(env, agent, logger, modelPath)

