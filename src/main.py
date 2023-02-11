import os
import sys
from DeepRTS import Engine, Constants
from logger import Logger
from functions import GetConfigDict, TransformImage, CreateVideoFromTempImages, SaveTempImage
from constants import inv_action_space
from Agents.ddqn import DDQN_Agent

STATE_SHAPE = (20, 16)

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
    agent = DDQN_Agent(state_dim=state_sizes, action_space_dim=len(action_space_ids), save_dir=logger.getSaveFolderPath())


    """
        Training loop
    """
    record_steps = int(config["epochs"] / 2) # record game every (total_epochs/x) epochs
    for e in range(config["epochs"]):

        game.start()
        observation = TransformImage(game.render(), STATE_SHAPE)
        i = 0
        recording_images = []
        record = (e % record_steps == 0) or (e == config["epochs"]-1) # last part to always record last
        if record:
            print("Recording this epoch")
        done = True
        prevDmg = 0
        while done:
            i += 1

            # Record game
            if record:
                image = game.render()
                SaveTempImage(logger.getSaveFolderPath(), image, i)


            # AI make action
            actionId = agent.act(observation) # choose what to do
            player0.do_action(action_space_ids[actionId]) # do it

            # Next frame
            game.update()
            next_observation = TransformImage(game.render(), STATE_SHAPE)

            # reward 
            reward = (player0.statistic_damage_done - prevDmg)/(i) + int(player0.evaluate_player_state == Constants.PlayerState.Victory)*100 # if win +1 otherwise 0
            prevDmg = player0.statistic_damage_done


            done = not game.is_terminal() #not game.is_terminal() and i < 1000 # limit time playable

            # AI Save memory
            agent.cache(observation, next_observation, actionId, reward, done)

            # Learn
            q, loss = agent.learn()
            
            # Logging
            logger.log_step(reward, loss, q)

            # Update state
            observation = next_observation


        game.reset()
        logger.log_epoch(e, agent.exploration_rate)

        # Record game
        if record:
            CreateVideoFromTempImages(os.path.join(logger.getSaveFolderPath(), "temp"), (e+1))

    # save model
    agent.save()