import cv2
import gym
import numpy as np
from constants import inv_action_space
from DeepRTS import Engine, Constants
from gym.spaces import Box 

class Simple1v1Gym(gym.Env):
    def __init__(self, map, max_episode_steps):
        self.max_episode_steps = max_episode_steps
        self.elapsed_steps = None

        """
            Start engine
        """
        engineConfig: Engine.Config = Engine.Config().defaults()
        engineConfig.set_gui("Blend2DGui")
        self.game: Engine.Game = Engine.Game(map, engineConfig)
        self.game.set_max_fps(0)  # 0 = unlimited
        # add 2 players
        self.player0: Engine.Player = self.game.add_player()
        self.player1: Engine.Player = self.game.add_player()
        
        self.action_space = [3, 4, 5, 6, 11, 16] # move and attack simple
        # max actions = list(range(Engine.Constants.ACTION_MIN, Engine.Constants.ACTION_MAX + 1))

        self.game.start()
        self.observation_space = Box(low=0, high=255, shape=(320, 320, 3), dtype=np.uint8)

        self.prevDmg = 0


    def step(self, actionIndex):
        self.elapsed_steps += 1

        self.player0.do_action(self.action_space[actionIndex])
        self.player1.do_action(16) # do nothing
        self.game.update()

        # reward 
        winnerReward = int(self.player1.evaluate_player_state() == Constants.PlayerState.Defeat)*1 # if win +1
        dmgReward = 1 - ((100 - (self.player0.statistic_damage_done - self.prevDmg)) / 100)**0.5 # rewards exponentioally based on dmg done ehre 100 = max dmg
        timeConservation = (self.max_episode_steps - self.elapsed_steps) / self.max_episode_steps # * the dmg reward, higher the lesser time has passed
        reward = dmgReward * timeConservation + winnerReward

        self.prevDmg = int(self.player0.statistic_damage_done)

        truncated = self.elapsed_steps > self.max_episode_steps # useless value needs to be here for frame stack wrapper

        return self._get_obs(), reward, self.game.is_terminal(), truncated, self._get_info()

    def render(self):
        """
            Return RGB image but this one will not be changed by wrappers
        """
        return cv2.cvtColor(self.game.render(), cv2.COLOR_RGBA2RGB)


    def _get_obs(self):
        return  cv2.cvtColor(self.game.render(), cv2.COLOR_RGBA2RGB)


    def _get_info(self):
        return {}


    def reset(self):
        self.elapsed_steps = 0
        self.game.reset()

        return self._get_obs(), self._get_info()
    

    def close(self):
        self.game.stop()
