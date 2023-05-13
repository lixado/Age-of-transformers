import os
import sys
import numpy as np
import pygame
from logger import Logger
from tqdm import tqdm
from Gyms.CustomGym import CustomGym
from functions import SaveTempImage, CreateVideoFromTempImages


def evaluate(agent, gym: CustomGym, modelPath, logger: Logger):
    workingDir = os.getcwd()
    if not os.path.exists(os.path.join(workingDir, "src")):
        sys.exit(
            f'Working directory: {workingDir} not correct, should be "Age-of-transformers/" not "{os.path.basename(os.path.normpath(workingDir))}"')

    save_path = logger.getSaveFolderPath()

    display = False
    record = False

    if display:
        pygame.init()
        pygame.display.set_caption('DeepRTS v3.0')  # set the pygame window name

        WIDTH = gym.render_shape[0]*2 # because of dashboard
        HEIGHT = gym.render_shape[1]
        canvas = pygame.display.set_mode((WIDTH, HEIGHT))

    #Set agent model and evaluation mode
    agent.loadModel(modelPath)
    agent.net.eval()

    evals = []

    tests = 1000
    for e in tqdm(range(tests)):
        state, info = gym.reset()
        action0 = -1
        reward = 10
        tick = 0

        agent.states_sequence = [state]
        agent.rewards_sequence = [reward]
        agent.timesteps_sequence = [tick]
        actionArr = np.zeros(agent.action_space_dim)
        actionArr[action0] = 1
        agent.actions_sequence = [actionArr]
        while True:#not done:
            tick+=1
            #Actions
            action0, q_values = agent.act()

            gym.save_player_state()

            state, reward, done, truncated, info = gym.step(action0)

            agent.append(state, action0, tick, reward)


            image = gym.render(q_values, reward)

            if display:
                img = pygame.transform.rotate(pygame.surfarray.make_surface(image), -90)
                img = pygame.transform.flip(img, True, False)

                canvas.blit(img, (0, 0))
                pygame.display.update()

            # Record game
            if record:
                SaveTempImage(save_path, gym.render(q_values, reward), tick)

            tests -= 1

            logger.log_step(reward, 0, 0)

            if done or truncated:
                evals.append(info["eval"])
                break
        logger.log_epoch(e, agent.exploration_rate, 0, False)

        # Record game
        if record:
            CreateVideoFromTempImages(os.path.join(save_path, "temp"), (e))

    gym.evalPrint(evals)
    

    gym.close()
    if display:
        pygame.quit()

