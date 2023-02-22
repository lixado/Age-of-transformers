import random
from ddqn import DDQN_Agent
from DeepRTS import Engine, Constants
import pygame
import cv2
import math
import time
import copy

class PlayerState():
    def __init__(self, player):
        #Current states
        self.player_state = player.evaluate_player_state() # Defeat, victory or playing
        self.gold = player.gold
        self.lumber = player.lumber
        self.stone = player.stone
        self.food = player.food

        # Statistics
        self.statistic_damage_done = player.statistic_damage_done
        self.statistic_damage_taken = player.statistic_damage_taken
        self.statistic_gathered_gold = player.statistic_gathered_gold
        self.statistic_gathered_lumber = player.statistic_gathered_lumber
        self.statistic_gathered_stone = player.statistic_gathered_stone
        self.statistic_units_created = player.statistic_units_created

    def evaluate_player_state(self):
        return self.player_state

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

def get_reward(player0, previousPlayer0: PlayerState, player1):
    if player0.evaluate_player_state() != Constants.PlayerState.Defeat and player1.evaluate_player_state() == Constants.PlayerState.Defeat:
        return 1000
    if player0.evaluate_player_state() == Constants.PlayerState.Defeat and player1.evaluate_player_state() != Constants.PlayerState.Defeat:
        return -100
    if player0.statistic_damage_done > previousPlayer0.statistic_damage_done and player1.statistic_damage_taken > 0 :
        return 100
    return 0


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
    #engine_config.set_instant_building(True)
    #engine_config.set_barracks(True)
    #engine_config.set_farm(True)
    #engine_config.set_footman(True)
    #engine_config.set_archer(False)
    #engine_config.set_start_lumber(1000)
    #engine_config.set_start_gold(0)
    #engine_config.set_start_stone(1000)
    engine_config.set_console_caption_enabled(False)

    env = Engine.Game(
        "15x15-2p-ffa-Cresal.json", engine_config
    )
    env.set_max_fps(60)
    player0 = env.add_player()
    player1 = env.add_player()

    state_dimensions=(env.get_width(), env.get_height(), 4)
    agent = DDQN_Agent(state_dimensions, 5, "../episodes", "hei")
    print(dir(env))
    prevDmg = 0

    for i in range(episodes):
        env.start()

        state = env.state
        done = False
        j = 0
        start = time.time()

        while not env.is_terminal():
            j += 1
            # event loop
            previousPlayer0 = PlayerState(player0)
            ev = pygame.event.get()
            image = env.render()

            rgb_image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            canvas.blit(pygame.surfarray.make_surface(rgb_image), (0, 0))
            pygame.display.update()
            #action1 = random.randrange(1,16)
            #action2 = random.randrange(1, 16)

            action1 = action(ev)
            player0.do_action(action1)

            env.update()
            next_state = env.state

            #player1.do_action(action2)
            # reward


            reward = get_reward(player0, previousPlayer0, player1)
            #print(player0.evaluate_player_state(), ",", player1.evaluate_player_state())
            #print(player0.statistic_damage_taken,",", player1.statistic_damage_taken)
            print(reward)
            #next_state, reward, done, _ = env.step(action)

            agent.cache(state, next_state, action1, 0, done)
            #q, loss = agent.learn()
            state = next_state

            if (done):
                break
        end = time.time()-start
        env.reset()
