import cv2
import gym
import numpy as np
from constants import inv_action_space
from DeepRTS import Engine, Constants
from gym.spaces import Box 

class Simple1v1Gym(gym.Env):
    def __init__(self, map):
        """
            Start engine
        """
        engineConfig: Engine.Config = Engine.Config().defaults()
        engineConfig.set_gui("Blend2DGui")
        self.game: Engine.Game = Engine.Game(map, engineConfig)
        self.game.set_max_fps(60)  # 0 = unlimited
        # add 2 players
        self.player0: Engine.Player = self.game.add_player()
        self.player1: Engine.Player = self.game.add_player()
        
        self.action_space = [3, 4, 5, 6, 11, 16] # move and attack simple
        # max actions = list(range(Engine.Constants.ACTION_MIN, Engine.Constants.ACTION_MAX + 1))

        self.game.start()
        self.observation_space = Box(low=0, high=255, shape=(320, 320, 3), dtype=np.uint8)


    def step(self, actionIndex):
        self.steps += 1

        self.player0.do_action(self.action_space[actionIndex])
        self.game.update()

        # reward 
        winnerReward = int(self.player1.evaluate_player_state() == Constants.PlayerState.Defeat)*10 - int(self.player0.evaluate_player_state() == Constants.PlayerState.Defeat)*10 # if win then +10 if loss -10
        dmgReward = 1 - ((100 - self.player0.statistic_damage_done) / 100)**0.5 # rewards exponentioally based on dmg done ehre 100 = max dmg
        timeConservation = (8000 - self.steps) / 8000 # * the dmg reward, higher the lesser time has passed
        reward = dmgReward * timeConservation + winnerReward

        truncated = False # useless value needs to be here for frame stack wrapper

        return self._get_obs(), reward, self.game.is_terminal(), truncated, self._get_info()

    def render(self):
        """
            Return RGB image but this one will not be changed by wrappers
        """
        return cv2.cvtColor(self.game.render(), cv2.COLOR_RGBA2RGB)


    def _get_obs(self):
        return  cv2.cvtColor(self.game.render(), cv2.COLOR_RGBA2RGB)


    def _get_info(self):
        return {"steps": self.steps}


    def reset(self):
        self.steps = 0
        self.game.reset()

        return self._get_obs(), self._get_info()
    

    def close(self):
        self.game.stop()
