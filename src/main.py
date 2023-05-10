import os
import sys
import torch
from dt_selftrain import train_dt_self
from logger import Logger
from functions import GetConfigDict, alphanum_key, chooseModel
from constants import inv_action_space
from Agents.decisition_transformer import DecisionTransformer_Agent
from Agents.ddqn import DDQN_Agent
from gym.wrappers import TransformObservation, FrameStack, TimeLimit, NormalizeReward, NormalizeObservation
from Gyms.Simple1v1 import Simple1v1Gym
from Gyms.Random1v1 import Random1v1Gym
from Gyms.Harvest import HarvestGym
from Gyms.Full1v1 import Full1v1Gym
from train import train_transformer, train_ddqn
from eval import evaluate
from playground import playground
from simulate import simulate

from wrappers import SkipFrame, RepeatFrame

STATE_SHAPE = (32, 32) # model input shapes
FRAME_STACK = 3


if __name__ == "__main__":
    workingDir = os.getcwd()
    if not os.path.exists(os.path.join(workingDir, "src")):
        sys.exit(f'Working directory: {workingDir} not correct, should be "Age-of-transformers/" not "{os.path.basename(os.path.normpath(workingDir))}"')

    config = GetConfigDict(workingDir)
    config["device"] = "cuda" if torch.cuda.is_available() and config["device"] == "cuda" else "cpu"
    print("Config: ", config)


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
        Handle env input
        0 = Simple1v1
        1 = Random1v1
        2 = Full1v1
        3 = Harvest
    """
    gymModes = ["Simple1v1", "Random1v1", "Full1v1", "Harvest"]
    for cnt, modeName in enumerate(gymModes, 1):
        sys.stdout.write("[%d] %s\n\r" % (cnt, modeName))
    gymMode = (int(input("Select gym[1-%s]: " % cnt)) - 1) if "gym" not in config else config["gym"]  # get from config file if exists
    print(f"{gymModes[gymMode]} gym.") if "gym" not in config else print(f"{modes[mode]} gym. Auto from config.json file.")

    
    
    """
        Start gym
    """
    if gymMode == 0:
        gym = Simple1v1Gym(STATE_SHAPE)
    elif gymMode == 1:
        gym = Random1v1Gym(STATE_SHAPE)
    elif gymMode == 2:
        gym = Full1v1Gym(STATE_SHAPE)
    elif gymMode == 3:
        gym = HarvestGym(STATE_SHAPE)
    else:
        print("Invalid gym")
        quit(0)
    print("Action space: ", [inv_action_space[i] for i in gym.action_space])

    # gym wrappers
    if config["skipFrame"] != 0:
        gym = SkipFrame(gym, config["skipFrame"])
    if config["repeatFrame"] != 0:
        gym = RepeatFrame(gym, config["repeatFrame"])
    if config["normalize"]:
        gym = NormalizeObservation(gym)
        gym = NormalizeReward(gym)
    gym = TimeLimit(gym, max_episode_steps=config["stepsMax"])


    """
        Start agent
    """
    agent = None
    if config["agent"] == 0:
        agent = DDQN_Agent(state_dim=(FRAME_STACK,) + STATE_SHAPE, action_space_dim=len(gym.action_space), config=config)
    elif config["agent"] > 0:
        agent = DecisionTransformer_Agent(state_dim=STATE_SHAPE, action_space_dim=len(gym.action_space), config=config)
    

    """
        Training loop
    """
    if mode == 0:
        logger = Logger(workingDir)

        if config["agent"] == 0:
            gym = FrameStack(gym, num_stack=FRAME_STACK, lz4_compress=False)
            train_ddqn(config, agent, gym, logger)
        elif config["agent"] == 1:
            #data_path = os.path.join(workingDir, "ddqn_harvest_data_3")
            data_path = os.path.join(workingDir, "simple1v1_data")
            train_transformer(config, agent, gym, logger, data_path)
        elif config["agent"] == 2:
            train_dt_self(config, agent, gym, logger)
    elif mode == 1:
        if config["agent"] == 0:
            gym = FrameStack(gym, num_stack=FRAME_STACK, lz4_compress=False)

        modelPath = chooseModel(os.path.join(workingDir, "results"))
        evaluate(agent, gym, modelPath)
    elif mode == 2:
        playground(gym)
    elif mode == 3:
        gym = FrameStack(gym, num_stack=FRAME_STACK, lz4_compress=False)

        modelPath = chooseModel(os.path.join(workingDir, "results"))
        logger = Logger(workingDir)
        simulate(config, agent, gym, logger, modelPath)
    else:
        print("Mode not avaliable")

