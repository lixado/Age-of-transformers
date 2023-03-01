import random
from DeepRTS import Engine
import pygame
import cv2
from functions import CreateVideoFromTempImages, SaveTempImage


"""
    Constants
"""
tileSize = 32

map = "15x15-2p-ffa-Cresal.json"
mapsize = (int(map.split("x")[0]), int(map.split("x")[1].split("-")[0]))


def test(logger):
    """
        Start pygame GUI
    """
    #pygame.init()

    #pygame.display.set_caption('DeepRTS v3.0')  # set the pygame window name

    #WIDTH = mapsize[0] * tileSize
    #HEIGHT = mapsize[1] * tileSize
    #canvas = pygame.display.set_mode((WIDTH, HEIGHT))

    """
        Start engine
    """
    config: Engine.Config = Engine.Config().defaults()
    config.set_gui("Blend2DGui")

    game: Engine.Game = Engine.Game(map, config)
    game.set_max_fps(0)  # 0 = unlimited
    player0: Engine.Player = game.add_player()
    player1: Engine.Player = game.add_player()

    """
        Game loop
    """
    game.start()
    for i in range(10):
        game.reset()
        i=0
        while not game.is_terminal():
            player0.do_action(random.randint(1, 16))
            player1.do_action(random.randint(1, 16))

            game.update()

            image = game.render()  # 4x320x320 image

            #rgb_image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

            #canvas.blit(pygame.surfarray.make_surface(rgb_image), (0, 0))
            #pygame.display.update()
            SaveTempImage(logger.getSaveFolderPath(), image, i)

            i+=1
    pygame.quit()


if __name__ == "__main__":
    test()