from asyncio import sleep
from time import time
from typing import List
from DeepRTS import Engine, Constants
import gym
import pygame
import cv2
import numpy as np
from PIL import Image
from constants import action_space, inv_action_space

"""
    Constants
"""
def playground(gym: gym.Env):
    """
        Start pygame GUI
    """
    pygame.init()
    pygame.display.set_caption('DeepRTS v3.0') # set the pygame window name
    size = (gym.initial_shape[0], gym.initial_shape[1])
    canvas = pygame.display.set_mode(size) # pygame.FULLSCREEN

    """
        Start engine
    """
    gym.game.set_max_fps(60)  # 0 = unlimited


    """
        Game loop
    """
    gym.game.start()
    old = time()
    done = False
    while not done:
        gym.game.reset()
        while not gym.game.is_terminal():

            # event loop
            ev = pygame.event.get()
            action = Controls(ev) # default do nothing
            if action == -1:
                done = True
                break

            if action != 16:
                print(inv_action_space[action])


            gym.player0.do_action(action)
            gym.game.update()

            state = gym.game.state # 10x10x10 not sure
            image = gym.game.render() # 4x320x320 image
            rgb_image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

            canvas.blit(pygame.surfarray.make_surface(rgb_image), (0,0))
            pygame.display.update()

            # print every second
            temp = time()
            if (temp - old) >= 1:
                print(gym.player0.statistic_damage_done)
                gym.game.caption()
                old = temp

        result = "LOST" if (gym.player0.evaluate_player_state() == Constants.PlayerState.Defeat) else "WON"
        print(f"Done you {result}")


    gym.close()
    pygame.quit()

def Controls(ev: List[pygame.event.Event]):
    for event in ev:
        if event.type == pygame.KEYDOWN: # if some key was pressed
            match event.key:
                case pygame.K_DOWN:
                    return 4
                case pygame.K_UP:
                    return 3
                case pygame.K_LEFT:
                    return 5      
                case pygame.K_RIGHT:
                    return 6
                case pygame.K_SPACE:
                    return 11
                case pygame.K_x:
                    return 12
                case pygame.K_c:
                    return 13
                case pygame.K_v:
                    return 14
                case pygame.K_b:
                    return 15
        if event.type == pygame.KEYUP:
            match event.key:
                case pygame.K_ESCAPE:
                    return -1
    return 16 # default
