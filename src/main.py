import os
import sys
import torch
from Gyms.Random1v1 import Random1v1Gym
from Gyms.Simple1v1 import Simple1v1Gym
from logger import Logger
from functions import GetConfigDict
from constants import inv_action_space
from Agents.ddqn import DDQN_Agent
from gym.wrappers import FrameStack, TransformObservation, TimeLimit
from train import train
from eval import evaluate
from playground import playground
from simulate import simulate

from wrappers import SkipFrame, RepeatFrame

STATE_SHAPE = (32, 32) # model input shapes
FRAME_STACK = 3 # get latest x frames into model
SKIP_FRAME = 10 # do no action for x frames then do action
REPEAT_FRAME = 0 # same action for x frames 

torch.set_float32_matmul_precision('high')

if __name__ == "__main__":
    workingDir = os.getcwd()
    if not os.path.exists(os.path.join(workingDir, "src")):
        sys.exit(f'Working directory: {workingDir} not correct, should be "Age-of-transformers/" not "{os.path.basename(os.path.normpath(workingDir))}"')

    config = GetConfigDict(workingDir)
    print("Config: ", config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: ", device)

    """
        Handle input default train
        0 = Train
        1 = Eval
        2 = Playground
        3 = Simulate games
    """
    modes = ["Train", "Eval", "Playground", "Simulate"]
    for cnt, modeName in enumerate(modes, 1):
        sys.stdout.write("[%d] %s\n\r" % (cnt, modeName))

    mode = (int(input("Select mode[1-%s]: " % cnt)) - 1) if "mode" not in config else config["mode"] # get from config file if exists

    print(f"{modes[mode]} mode.") if "mode" not in config else print(f"{modes[mode]} mode. Auto from config.json file.")

    
    """
        Start gym
    """
    gym = Random1v1Gym(0, config["stepsMax"])
    print("Action space: ", [inv_action_space[i] for i in gym.action_space])

    # gym wrappers
    if SKIP_FRAME != 0:
        gym = SkipFrame(gym, SKIP_FRAME)
    if REPEAT_FRAME != 0:
        gym = RepeatFrame(gym, REPEAT_FRAME)
    gym = TransformObservation(gym, f=lambda x: x / 13.)  # normalize the values [0, 1] #MAX VALUE=20
    gym = FrameStack(gym, num_stack=FRAME_STACK, lz4_compress=False)
    gym = TimeLimit(gym, max_episode_steps=config["stepsMax"])

    """
        Start agent
    """
    state_sizes = (FRAME_STACK, ) + STATE_SHAPE # number of image stacked
    agent = DDQN_Agent(state_dim=state_sizes, action_space_dim=len(gym.action_space))
    agent.device = device

    """
        Training loop
    """
    if mode == 0:
        logger = Logger(workingDir)
        train(config, agent, gym, logger)
    elif mode == 1:
        # get latest model path
        results = os.path.join(workingDir, "results")
        folders = os.listdir(results)
        paths = [os.path.join(results, basename) for basename in folders]
        latestFolder = paths[-1]
        modelPath = os.path.join(latestFolder, "model.chkpt")
        evaluate(agent, gym, modelPath)
    elif mode == 2:
        playground(gym)
    elif mode == 3:
        logger = Logger(workingDir)
        simulate(config, gym, logger.getSaveFolderPath())
    else:
        print("Mode not avaliable")
