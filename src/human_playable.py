from DeepRTS import Engine
import cv2
from constants import inv_action_space
import sys




if __name__ == "__main__":

    num_episodes = 100000

    config: Engine.Config = Engine.Config().defaults()
    config.set_console_caption_enabled(True)
    config.set_gui("Blend2DGui")

    game: Engine.Game = Engine.Game("10x10-2p-ffa-Eblil.json", config)
    player0: Engine.Player = game.add_player()
    player1: Engine.Player = game.add_player()

    game.set_max_fps(0)  # 0 = unlimited
    game.start()

    for i in range(Engine.Constants.ACTION_MIN, Engine.Constants.ACTION_MAX+1):
        print(f"{i}. ", inv_action_space[i])

    game.reset()
    while not game.is_terminal():
        

        player0.do_action(16)

        player1.do_action(16) # 16 = do nothing

        game.update()
        state = game.state
        image = game.render()
        

        cv2.imshow("DeepRTS", image)
        cv2.waitKey(1)
        game.caption() #print fps