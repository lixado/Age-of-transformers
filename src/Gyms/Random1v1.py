import cv2
import gym
import numpy as np
from constants import inv_action_space
from DeepRTS import Engine, Constants
from gym.spaces import Box
from functions import PlayerState
import random
from Gyms.CustomGym import CustomGym


MAP = "10x10-2p-ffa-Eblil.json"


class Random1v1Gym(CustomGym):
    def __init__(self, max_episode_steps, shape):
        engineConfig: Engine.Config = Engine.Config().defaults()
        engineConfig.set_gui("Blend2DGui")
        engineConfig.set_auto_attack(True)

        self.action_space = [3, 4, 5, 6, 11, 16]  # move and attack simple
        self.reward = 0

        super().__init__(max_episode_steps, shape, MAP, engineConfig)

        self.player1: Engine.Player = self.game.add_player()

        self.previousPlayer0 = PlayerState(self.player0)

    def step(self, actionIndex):
        self.elapsed_steps += 1
        self.action = actionIndex

        # randomize action order to euqalize
        if random.random() < 0.5:
            self.player0.do_action(self.action_space[actionIndex])
            self.player1.do_action(random.choice(self.action_space))  # do nothing
        else:
            self.player1.do_action(random.choice(self.action_space))  # do nothing
            self.player0.do_action(self.action_space[actionIndex])

        self.game.update()

        # reward
        doneDmg = int(self.player0.statistic_damage_done > self.previousPlayer0.statistic_damage_done)
        reward = doneDmg*2 - 1

        return self._get_obs(), reward, self.game.is_terminal(), False, self._get_info()
    

    def _get_info(self):
        if self.player1.evaluate_player_state() == Constants.PlayerState.Defeat:
            return {"eval": self.elapsed_steps}
        
        return {"eval": None}
    
    def evalPrint(self, evals):
        evalsNotNone = [x for x in evals if x is not None]
        print(f"Total kills: {len(evalsNotNone)} out of {len(evals)}, Avg (ticks): {sum(evalsNotNone)/len(evalsNotNone) if len(evalsNotNone) > 0 else 0}, Min (ticks): {min(evalsNotNone) if len(evalsNotNone) > 0 else 0}, Max (ticks): {max(evalsNotNone) if len(evalsNotNone) > 0 else 0}")

    def render(self, q_values, reward):
        """
            Return RGB image but this one will not be changed by wrappers
        """
        image = cv2.cvtColor(self.game.render(), cv2.COLOR_RGBA2RGB)
        dashboard = np.zeros(image.shape, dtype=np.uint8)
        dashboard.fill(255)

        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (10, image.shape[1] - 10)
        fontScale = 0.5
        spacing = int(40 * fontScale)
        color = (0, 0, 0)
        thickness = 1

        texts = [f"Update Nr.: {self.elapsed_steps}",
                 f"Q_values:",
                 f"Q_Left: {q_values[0]}",
                 f"Q_Right: {q_values[1]}",
                 f"Q_Up: {q_values[2]}",
                 f"Q_Down: {q_values[3]}",
                 f"Q_Attack: {q_values[4]}",
                 f"Q_None: {q_values[5]}",
                 f"player0.statistic_damage_done: {self.player0.statistic_damage_done}",
                 f"Reward: {reward}",
                 f"Action: {inv_action_space[self.action_space[self.action]]}"]

        for text in texts[::-1]:
            dashboard = cv2.putText(dashboard, text, org, font, fontScale, color, thickness, cv2.LINE_AA, False)
            org = (org[0], org[1] - spacing)

        image = cv2.hconcat([dashboard, image])
        return image