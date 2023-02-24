import os
import random
from Agents.ddqn import DDQN_Agent
from DeepRTS import Engine, Constants
import pygame
import cv2
import math
import time
import copy
from logger import Logger
from constants import inv_action_space
from Gyms.Simple1v1 import Simple1v1Gym
from gym.wrappers import FrameStack, TransformObservation, ResizeObservation, GrayScaleObservation
from functions import SaveTempImage


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
    return action

def evaluate(env, agent, modelPath):
    workingDir = os.getcwd()
    if not os.path.exists(os.path.join(workingDir, "src")):
        sys.exit(
            f'Working directory: {workingDir} not correct, should be "Age-of-transformers/" not "{os.path.basename(os.path.normpath(workingDir))}"')

    episodes = 100

    logger = Logger(workingDir)

    pygame.init()

    pygame.display.set_caption('DeepRTS v3.0')  # set the pygame window name
    mapSize = [10, 10]
    tileSize = 32
    WIDTH = mapSize[0] * tileSize
    HEIGHT = mapSize[1] * tileSize
    canvas = pygame.display.set_mode((WIDTH, HEIGHT))

    print("Action space: ", [inv_action_space[i] for i in env.action_space])

    #Set agent model and evaluation mode
    agent.loadModel(modelPath)
    agent.net.eval()

    for i in range(episodes):
        state, info = env.reset()
        done = False
        j = 0
        start = time.time()
        while not done:
            j += 1
            image = env.render()

            canvas.blit(pygame.surfarray.make_surface(image), (0, 0))
            pygame.display.update()

            #Actions
            action0 = agent.act(state)
            state, reward, done, _, _ = env.step(action0)

            ev = pygame.event.get()
            action1 = action(ev)
            env.player1.do_action(action1)

            if done:
                break
        print("Done")
        end = time.time() - start

