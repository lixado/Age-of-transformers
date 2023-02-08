import random
import os
import sys
from DeepRTS import Engine, Constants
from logger import Logger
from functions import GetConfigDict
from constants import inv_action_space
from Agents.ddqn import DDQN_Agent


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
    state_sizes = (320, 320, 4)
    agent = DDQN_Agent(state_dim=state_sizes, action_space_dim=len(action_space_ids), save_dir=logger.getSaveSolver())


    """
        Training loop
    """
    for e in range(config["epochs"]):
        if e % 10 == 0:
            print(e)

        game.start()
        observation = game.render()
        i = 0
        while not game.is_terminal():
            i += 1

            # AI make action
            actionId = agent.act(observation) # choose what to do
            player0.do_action(action_space_ids[actionId]) # do it

            # Next frame
            game.update()
            next_observation = game.render()

            # reward function
            reward = player0.statistic_damage_done # the more dmg done the more reward

            # AI Save memory
            agent.cache(observation, next_observation, actionId, reward, game.is_terminal())

            # Learn
            q, loss = agent.learn()
            
            # Logging
			#logger.log_step(reward, loss, q, player.scheduler.get_last_lr())

            # Update state
            observation = next_observation

            if i % 100000000:
                print(f'Working on it. {reward}, action: {inv_action_space[action_space_ids[actionId]]}')

        game.reset()

    # save model
    agent.save()