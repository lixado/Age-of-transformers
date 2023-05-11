import os
import sys

import gym
from Agents.ddqn import DDQN_Agent
import pygame
import time
from tqdm import tqdm

from Gyms.CustomGym import CustomGym

def action(ev):
    action = 16
    for event in ev:
        if event.type == pygame.KEYDOWN: # if some key was pressed
            match event.key:
                case pygame.K_1:
                    action = 1
                case pygame.K_2:
                    action = 2

                case pygame.K_UP:
                    action = 3
                case pygame.K_DOWN:
                    action = 4
                case pygame.K_LEFT:
                    action = 5
                case pygame.K_RIGHT:
                    action = 6

                case pygame.K_SPACE:
                    action = 11
                case pygame.K_x:
                    action = 12

                case pygame.K_3:
                    action = 13
                case pygame.K_4:
                    action = 14
                case pygame.K_5:
                    action = 15

        if event.type == pygame.KEYUP:
            match event.key:
                case pygame.K_ESCAPE:
                    return -1
    return action


def evaluate(agent: DDQN_Agent, gym: CustomGym, modelPath):
    workingDir = os.getcwd()
    if not os.path.exists(os.path.join(workingDir, "src")):
        sys.exit(
            f'Working directory: {workingDir} not correct, should be "Age-of-transformers/" not "{os.path.basename(os.path.normpath(workingDir))}"')

    #logger = Logger(workingDir) might be needed

    display = False

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
    for _ in tqdm(range(tests)):
        state, info = gym.reset()
        action0 = -1  # first acction is default Do nothing
        reward = 1200
        tick = 0
        
        start = time.time()
        while True:#not done:
            tick += 1
            #Actions
            action0, q_values = agent.act(state, action0, tick, reward)

            gym.save_player_state()

            state, reward, done, truncated, info = gym.step(action0)

            image = gym.render(q_values, reward)

            if display:
                img = pygame.transform.rotate(pygame.surfarray.make_surface(image), -90)
                img = pygame.transform.flip(img, True, False)

                canvas.blit(img, (0, 0))
                pygame.display.update()


                ev = pygame.event.get()
                action1 = action(ev)

                if action1 == -1: # pressed escape leave
                    play = False
                    break

                gym.player1.do_action(action1)

            tests -= 1

            if done or truncated:
                evals.append(info["eval"])
                break


    gym.evalPrint(evals)
    

    gym.close()
    if display:
        pygame.quit()

