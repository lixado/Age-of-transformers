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

        self.shape = (320, 320, 3) # W, H, C
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
        self.observation_space = Box(low=0, high=255, shape=self.shape, dtype=np.uint8)


    def step(self, actionIndex):
        self.elapsed_steps += 1
        self.action = actionIndex

        self.player0.do_action(self.action_space[actionIndex])
        self.player1.do_action(16) # do nothing
        self.game.update()

        # reward 
        winnerReward = int(self.player1.evaluate_player_state() == Constants.PlayerState.Defeat) - int(self.player0.evaluate_player_state() == Constants.PlayerState.Defeat)*1 # if win +1
        dmgReward = 1 - ((100 - self.player0.statistic_damage_done) / 100)**0.5 # rewards exponentioally based on dmg done ehre 100 = max dmg
        timeConservation = (self.max_episode_steps - self.elapsed_steps) / self.max_episode_steps # * the dmg reward, higher the lesser time has passed
        self.reward = dmgReward * timeConservation + winnerReward

        truncated = self.elapsed_steps > self.max_episode_steps # useless value needs to be here for frame stack wrapper

        return self._get_obs(), self.reward, self.game.is_terminal(), truncated, self._get_info()

    def render(self):
        """
            Return RGB image but this one will not be changed by wrappers
        """
        image = cv2.cvtColor(self.game.render(), cv2.COLOR_RGBA2RGB)
        dashboard = np.zeros(self.shape,dtype=np.uint8)
        dashboard.fill(255)
        

        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (10, self.shape[1]-10)
        fontScale = 0.5
        spacing = int(40 * fontScale)
        color = (0, 0, 0)
        thickness = 1

        texts = [f"player0.statistic_damage_done: {self.player0.statistic_damage_done}",
                f"Reward: {self.reward}",
                f"Action: {inv_action_space[self.action_space[self.action]]}"]

        for text in texts[::-1]:        
            dashboard = cv2.putText(dashboard, text, org, font, fontScale, color, thickness, cv2.LINE_AA, False)
            org = (org[0], org[1] - spacing)

        image = cv2.hconcat([dashboard, image])
        return image


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
