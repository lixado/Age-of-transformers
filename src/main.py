import os
import sys
import torch
from logger import Logger
from functions import GetConfigDict, alphanum_key
from constants import inv_action_space
from Agents.decisition_transformer import DecisionTransformer_Agent
from Agents.ddqn import DDQN_Agent
from gym.wrappers import TransformObservation, FrameStack, TimeLimit
from Gyms.Simple1v1 import Simple1v1Gym
from Gyms.Random1v1 import Random1v1Gym
from Gyms.Harvest import HarvestGym
from train import train_transformer, train_ddqn
from eval import evaluate
from playground import playground
from simulate import simulate

from wrappers import SkipFrame, RepeatFrame

STATE_SHAPE = (32, 32) # model input shapes
FRAME_STACK = 3

def chooseModel(folderPath):
    folderList = [name for name in os.listdir(folderPath) if os.path.isdir(os.path.join(folderPath, name)) and len(os.listdir(os.path.join(folderPath, name))) != 0]

    if len(folderList) == 0:
        print("No models to load in path: ", folderPath)
        quit()

    for cnt, fileName in enumerate(folderList, 1):
        sys.stdout.write("[%d] %s\n\r" % (cnt, fileName))

    choice = int(input("Select folder with platformer model[1-%s]: " % cnt)) - 1
    folder = folderList[choice]
    print(folder)

    fileList = [f for f in os.listdir(os.path.join(folderPath, folder)) if f.endswith(".chkpt")]

    if len(fileList) == 0:
        print("No models to load in path: ", folder)
        quit()

    modelPath = os.path.join(folderPath, folder, fileList[0])
    return modelPath


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

    gymModes = ["Simple1v1", "Random1v1", "Harvest"]
    for cnt, modeName in enumerate(gymModes, 1):
        sys.stdout.write("[%d] %s\n\r" % (cnt, modeName))

    gymMode = (int(input("Select gym[1-%s]: " % cnt)) - 1) if "gym" not in config else config[
        "gym"]  # get from config file if exists

    print(f"{gymModes[gymMode]} gym.") if "gym" not in config else print(f"{modes[mode]} gym. Auto from config.json file.")
    
    """
        Start gym
    """
    if gymMode == 0:
        gym = Simple1v1Gym(config["stepsMax"], STATE_SHAPE)
    elif gymMode == 1:
        gym = Random1v1Gym(config["stepsMax"], STATE_SHAPE)
    elif gymMode == 2:
        gym = HarvestGym(config["stepsMax"], STATE_SHAPE)
    else:
        print("Invalid gym")
        quit(0)
    print("Action space: ", [inv_action_space[i] for i in gym.action_space])

    # gym wrappers
    if config["skipFrame"] != 0:
        gym = SkipFrame(gym, config["skipFrame"])
    if config["repeatFrame"] != 0:
        gym = RepeatFrame(gym, config["repeatFrame"])
    gym = TransformObservation(gym, f=lambda x: x / 20.)  # normalize the values [0, 1] #MAX VALUE=20
    gym = TimeLimit(gym, max_episode_steps=config["stepsMax"])

    """
        Start agent
    """
    state_sizes = STATE_SHAPE # number of image stacked
    agent = None
    #agent = DecisionTransformer_Agent(state_dim=state_sizes, action_space_dim=len(gym.action_space), device=device, max_steps=config["stepsMax"]+1, batch_size=config["batchSize"])
    ddqn_agent = DDQN_Agent(state_dim=(FRAME_STACK,) + STATE_SHAPE, action_space_dim=len(gym.action_space))

    """
        Training loop
    """
    if mode == 0:
        logger = Logger(workingDir)
        if agent != None:
            data_path = os.path.join(workingDir, "simple1v1_data")
            train_transformer(config, agent, gym, logger, data_path)
        else:
            gym = FrameStack(gym, num_stack=FRAME_STACK, lz4_compress=False)
            train_ddqn(config, ddqn_agent, gym, logger)
    elif mode == 1:
        if agent == None:
            gym = FrameStack(gym, num_stack=FRAME_STACK, lz4_compress=False)
            
        modelPath = chooseModel(os.path.join(workingDir, "results"))
        evaluate(ddqn_agent, gym, modelPath)
    elif mode == 2:
        playground(gym)
    elif mode == 3:
        gym = FrameStack(gym, num_stack=FRAME_STACK, lz4_compress=False)

        modelPath = chooseModel(os.path.join(workingDir, "results"))
        logger = Logger(workingDir)
        simulate(config, ddqn_agent, gym, logger, modelPath)
    else:
        print("Mode not avaliable")

