import random
import os
import sys
import cv2
import numpy as np
from DeepRTS import Engine, Constants
from logger import Logger
from functions import GetConfigDict
from constants import inv_action_space
from Agents.ddqn import DDQN_Agent

STATE_SHAPE = (20, 16)

def TransformImage(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    resize = cv2.resize(gray, STATE_SHAPE[::-1])
    framestack = np.expand_dims(resize, 0)
    
    return framestack


if __name__ == "__main__":
    workingDir = os.getcwd()
    if not os.path.exists(os.path.join(workingDir, "src")):
        sys.exit(f'Working directory: {workingDir} not correct, should be "Age-of-transformers/" not "{os.path.basename(os.path.normpath(workingDir))}"')

    config = GetConfigDict(workingDir)
    print("Config: ", config)

    logger = Logger(workingDir)

    
    """
        Start engine
    """
    engineConfig: Engine.Config = Engine.Config().defaults()
    engineConfig.set_console_caption_enabled(True)
    engineConfig.set_gui("Blend2DGui")
    game: Engine.Game = Engine.Game("10x10-2p-ffa-Eblil.json", engineConfig)
    game.set_max_fps(0)  # 0 = unlimited
    player0: Engine.Player = game.add_player()
    player1: Engine.Player = game.add_player()

    action_space_ids = [3, 4, 5, 6, 11] # move and attack simple
    #action_space_ids = list(range(Engine.Constants.ACTION_MIN, Engine.Constants.ACTION_MAX + 1))
    print("Action space: ", [inv_action_space[i] for i in action_space_ids])


    """
        Start agent
    """
    state_sizes = (1, ) + STATE_SHAPE # number of image stacked
    agent = DDQN_Agent(state_dim=state_sizes, action_space_dim=len(action_space_ids), save_dir=logger.getSaveSolver())


    """
        Training loop
    """
    for e in range(config["epochs"]):

        game.start()
        observation = TransformImage(game.render())
        i = 0
        done = False
        while done:
            i += 1

            # AI make action
            actionId = agent.act(observation) # choose what to do
            player0.do_action(action_space_ids[actionId]) # do it

            # Next frame
            game.update()
            next_observation = TransformImage(game.render())

            # reward function
            reward = player0.statistic_damage_done - 1 # the more dmg done the more reward but each tick is -1 to kill faster

            done = not game.is_terminal() and i < 20000 # limit time playable

            # AI Save memory
            agent.cache(observation, next_observation, actionId, reward, done)

            # Learn
            q, loss = agent.learn()
            
            # Logging
            logger.log_step(reward, loss, q)

            # Update state
            observation = next_observation

            if i % 600 == 2:
                print(f'Working on it. {reward}, action: {inv_action_space[action_space_ids[actionId]]}')

        game.reset()
        logger.log_epoch(e, agent.exploration_rate)

    # save model
    agent.save()