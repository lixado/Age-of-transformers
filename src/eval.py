import os
import sys

import gym
from Agents.ddqn import DDQN_Agent
from DeepRTS import Engine, Constants
import pygame
import time
from constants import inv_action_space



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

def evaluate(agent: DDQN_Agent, gym: gym.Env, modelPath):
    workingDir = os.getcwd()
    if not os.path.exists(os.path.join(workingDir, "src")):
        sys.exit(
            f'Working directory: {workingDir} not correct, should be "Age-of-transformers/" not "{os.path.basename(os.path.normpath(workingDir))}"')

    #logger = Logger(workingDir) might be needed

    pygame.init()
    pygame.display.set_caption('DeepRTS v3.0')  # set the pygame window name

    WIDTH = gym.shape[0]*2 # because of dashboard
    HEIGHT = gym.shape[1]
    canvas = pygame.display.set_mode((WIDTH, HEIGHT))

    #Set agent model and evaluation mode
    agent.loadModel(modelPath)
    agent.net.eval()

    play = True
    while play:
        state, info = gym.reset()
        done = False
        j = 0
        start = time.time()
        action1 = 16
        while True:#not done:
            j += 1
            #Actions
            action0, q_values = agent.act(state)
            state, reward, done, _, _ = gym.step(action0)


            image = gym.render(q_values)

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

            if done:
                break
            
        print("Done")
    gym.close()
    pygame.quit()

