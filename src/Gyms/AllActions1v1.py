import random

import cv2
import gym
import numpy as np
from constants import inv_action_space
from DeepRTS import Engine, Constants
from gym.spaces import Box
from src.functions import PlayerState

MAP = "15x15-2p-ffa-Cresal.json"


def conditional_reward(player0, previousPlayer0: PlayerState, player1, ticks):
    if player0.evaluate_player_state() != Constants.PlayerState.Defeat and player1.evaluate_player_state() == Constants.PlayerState.Defeat:
        return 100 / ticks
    if player0.evaluate_player_state() == Constants.PlayerState.Defeat and player1.evaluate_player_state() != Constants.PlayerState.Defeat:
        return -10
    if player0.statistic_damage_done > previousPlayer0.statistic_damage_done and player1.statistic_damage_taken > 0:
        return 10 / ticks
    return 0


class AllActions1v1(gym.Env):
    def __init__(self, mode, max_episode_steps):
        self.max_episode_steps = max_episode_steps
        self.elapsed_steps = None

        tilesize = 32
        mapSize = MAP.split("-")[0].split("x")
        self.initial_shape = (int(mapSize[0])*tilesize, int(mapSize[1])*tilesize, 3) # W, H, C
        print("Initial_shape: ", self.initial_shape)
        
        """
            Start engine
        """
        engineConfig: Engine.Config = Engine.Config().defaults()
        engineConfig.set_gui("Blend2DGui")
        engineConfig.set_instant_building(True)
        engineConfig.set_barracks(True)
        engineConfig.set_farm(True)
        engineConfig.set_footman(True)
        engineConfig.set_start_lumber(1000)
        engineConfig.set_start_gold(1000)
        engineConfig.set_start_stone(1000)
        self.game: Engine.Game = Engine.Game(MAP, engineConfig)
        self.game.set_max_fps(0)  # 0 = unlimited
        # add 2 players
        self.player0: Engine.Player = self.game.add_player()
        self.player1: Engine.Player = self.game.add_player()
        self.action_space = [i for i in range(1, 17)]  # 1-16, all actions, (see deep-rts/bindings/Constants.cpp)
        self.mode = mode
        self.game.start()
        self.observation_space = Box(low=0, high=255, shape=self.initial_shape, dtype=np.uint8)

    def step(self, actionIndex):
        self.elapsed_steps += 1
        self.action = actionIndex

        previousPlayer0 = PlayerState(self.player0)

        self.player0.do_action(self.action_space[actionIndex])

        actionIndex2 = random.randint(0, len(self.action_space) - 1)
        self.player1.do_action(self.action_space[actionIndex2])  # do nothing

        self.game.update()

        self.reward = conditional_reward(self.player0, previousPlayer0, self.player1, self.elapsed_steps)

        truncated = self.elapsed_steps > self.max_episode_steps # useless value needs to be here for frame stack wrapper
        return self._get_obs(), self.reward, self.game.is_terminal(), truncated, self._get_info()
    
    def render(self):
        """
            Return RGB image but this one will not be changed by wrappers
        """
        image = cv2.cvtColor(self.game.render(), cv2.COLOR_RGBA2RGB)
        dashboard = np.zeros(self.initial_shape,dtype=np.uint8)
        dashboard.fill(255)
        

        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (10, self.initial_shape[1]-10)
        fontScale = 0.5
        spacing = int(40 * fontScale)
        color = (0, 0, 0)
        thickness = 1

        texts = [f"Update Nr.: {self.elapsed_steps}",
                 f"player0.damage_done: {self.player0.statistic_damage_done}",
                 f"Reward: {self.reward}",
                 f"Action: {inv_action_space[self.action_space[self.action]]}"]

        for text in texts[::-1]:        
            dashboard = cv2.putText(dashboard, text, org, font, fontScale, color, thickness, cv2.LINE_AA, False)
            org = (org[0], org[1] - spacing)

        image = cv2.hconcat([dashboard, image])
        return image


    def _get_obs(self):
        return cv2.cvtColor(self.game.render(), cv2.COLOR_RGBA2RGB)


    def _get_info(self):
        return {}


    def reset(self):
        self.elapsed_steps = 0
        self.game.reset()

        return self._get_obs(), self._get_info()
    

    def close(self):
        self.game.stop()
