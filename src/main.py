from DeepRTS import Engine

if __name__ == "__main__":
    # Minimal example of DeepRTS in python.
    # on my machine, the recorded FPS is: 60772
    # if you run purely in C++, the number is 4 001 004

    num_episodes = 100000

    config: Engine.Config = Engine.Config().defaults()
    config.set_console_caption_enabled(True)
    config.set_gui("Blend2DGui")

    game: Engine.Game = Engine.Game("10x10-2v2.json", config)
    player0: Engine.Player = game.add_player()
    player1: Engine.Player = game.add_player()

    game.set_max_fps(0)  # 0 = unlimited
    game.start()

    for i in range(num_episodes):

        game.reset()
        while not game.is_terminal():
            player0.do_action(Engine.Random.get(Engine.Constants.ACTION_MIN, Engine.Constants.ACTION_MAX))
            player1.do_action(Engine.Random.get(Engine.Constants.ACTION_MIN, Engine.Constants.ACTION_MAX))

            game.update()
            state = game.state
            image = game.render()

            # cv2.imshow("DeepRTS", image)
            # cv2.waitKey(1)
            game.caption()