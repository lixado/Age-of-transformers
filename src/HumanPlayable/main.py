from asyncio import sleep
from time import time
from DeepRTS import Engine
import pygame
import cv2
import numpy as np
from PIL import Image
from constants import action_space, inv_action_space

"""
    Constants
"""
tileSize = 32

map = "10x10-2p-ffa-Eblil.json"
mapsize = (int(map.split("x")[0]), int(map.split("x")[1].split("-")[0]))

"""
    Start pygame GUI
"""
pygame.init()

pygame.display.set_caption('DeepRTS v3.0') # set the pygame window name

WIDTH = mapsize[0]*tileSize
HEIGHT = mapsize[1]*tileSize
canvas = pygame.display.set_mode((WIDTH, HEIGHT))
 

"""
    Start engine
"""
config: Engine.Config = Engine.Config().defaults()
config.set_console_caption_enabled(True)
config.set_gui("Blend2DGui")

game: Engine.Game = Engine.Game("10x10-2p-ffa-Eblil.json", config)
game.set_max_fps(240)  # 0 = unlimited
player0: Engine.Player = game.add_player()
player1: Engine.Player = game.add_player()


"""
    Game loop
"""
game.start()
old = time()
while not game.is_terminal():

    # event loop
    ev = pygame.event.get()
    action = 16 # default do nothing
    for event in ev:
        if event.type == pygame.KEYDOWN: # if some key was pressed
            match event.key:
                case pygame.K_DOWN:
                    action = 6
                case pygame.K_UP:
                    action = 5
                case pygame.K_LEFT:
                    action = 3         
                case pygame.K_RIGHT:
                    action = 4
                case pygame.K_SPACE:
                    action = 11
                case pygame.K_x:
                    action = 12
                case pygame.K_c:
                    action = 13
                case pygame.K_v:
                    action = 14
                case pygame.K_b:
                    action = 15

        if action != 16:
            print(inv_action_space[action])
            break # only 1 action per loop

    
    player0.do_action(action)
    player1.do_action(16) # 16 = do nothing

    game.update()
    print(player0.statistic_damage_done)
    print(player0.statistic_damage_done)
    state = game.state # 10x10x10 not sure
    image = game.render() # 4x320x320 image
    #im = cv2.imread(image,mode='RGB')
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    canvas.blit(pygame.surfarray.make_surface(rgb_image), (0,0))
    pygame.display.update()

    temp = time()
    if (temp - old) >= 1:
        game.caption()
        old = temp

pygame.quit()