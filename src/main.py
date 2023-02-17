import random
from ddqn import DDQN_Agent
from DeepRTS import Engine
import pygame
import cv2

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

if __name__ == "__main__":
    random_play = True
    episodes = 100

    pygame.init()

    pygame.display.set_caption('DeepRTS v3.0')  # set the pygame window name
    mapSize=[15,15]
    tileSize=32
    WIDTH = mapSize[0] * tileSize
    HEIGHT = mapSize[1] * tileSize
    canvas = pygame.display.set_mode((WIDTH, HEIGHT))

    engine_config: Engine.Config = Engine.Config.defaults()
    engine_config.set_instant_building(True)
    engine_config.set_barracks(True)
    engine_config.set_farm(True)
    engine_config.set_footman(True)
    engine_config.set_archer(False)
    engine_config.set_start_lumber(1000)
    engine_config.set_start_gold(10000)
    engine_config.set_start_stone(1000)
    engine_config.set_console_caption_enabled(False)

    env = Engine.Game(
        "15x15-2p-ffa-Cresal.json", engine_config
    )
    env.set_max_fps(60)
    player1 = env.add_player()
    player2 = env.add_player()

    state_dimensions=(env.get_width(), env.get_height(), 4)
    agent = DDQN_Agent(state_dimensions, 5, "../episodes", "hei")
    print(dir(env.tilemap.get_tile(1, 1)))

    for i in range(episodes):
        env.start()

        state = env.state
        done = False

        while not env.is_terminal():
            # event loop
            ev = pygame.event.get()
            image = env.render()

            rgb_image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            canvas.blit(pygame.surfarray.make_surface(rgb_image), (0, 0))
            pygame.display.update()

            #action1 = random.randrange(1,16)
            action2 = random.randrange(1, 16)

            action1 = action(ev)
            player1.do_action(action1)
            player2.do_action(action2)
            print(player1.gold, player1.stone, player1.lumber)
            env.update()
            next_state = env.state
            #next_state, reward, done, _ = env.step(action)

            agent.cache(state, next_state, action1, 0, done)
            #q, loss = agent.learn()
            state = next_state

            if (done):
                break
        env.reset()
        env.stop()
