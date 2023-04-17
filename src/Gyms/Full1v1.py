import cv2
import numpy as np
from constants import inv_action_space
from DeepRTS import Engine, Constants
from functions import PlayerState
from random import randint

from Gyms.CustomGym import CustomGym

MAP = "10x10-2p-ffa-Eblil.json"

def conditional_reward(player0, previousPlayer0: PlayerState, player1, ticks):
    if player0.evaluate_player_state() != Constants.PlayerState.Defeat and player1.evaluate_player_state() == Constants.PlayerState.Defeat:
        return 10000/ticks
    if player0.evaluate_player_state() == Constants.PlayerState.Defeat and player1.evaluate_player_state() != Constants.PlayerState.Defeat:
        return -0.001*ticks
    if player0.statistic_damage_done > previousPlayer0.statistic_damage_done and player1.statistic_damage_taken > 0:
        return 1000/ticks
    return -1

class Full1v1Gym(CustomGym):
    def __init__(self, max_episode_steps, shape):
        engineConfig: Engine.Config = Engine.Config().defaults()
        engineConfig.set_gui("Blend2DGui")
        engineConfig.set_auto_attack(True)

        self.action_space = [3, 4, 5, 6, 11, 16] # move and attack simple
        self.reward = 0

        super().__init__(max_episode_steps, shape, MAP, engineConfig)


    def step(self, actionIndices):
        self.elapsed_steps += 1
        self.action1 = actionIndices[0]
        self.action2 = actionIndices[1]

        previousPlayer0 = PlayerState(self.player0)
        previousPlayer1 = PlayerState(self.player1)

        if randint(0,1) == 0:
            self.player1.do_action(self.action_space[actionIndices[1]])
            self.player0.do_action(self.action_space[actionIndices[0]])
        else:
            self.player0.do_action(self.action_space[actionIndices[0]])
            self.player1.do_action(self.action_space[actionIndices[1]])

        self.game.update()

        # reward
        self.rewards = np.array([
            conditional_reward(self.player0, previousPlayer0, self.player1, self.elapsed_steps),
            conditional_reward(self.player1, previousPlayer1, self.player0, self.elapsed_steps),
        ])

        truncated = self.elapsed_steps > self.max_episode_steps # useless value needs to be here for frame stack wrapper
        return self._get_obs(), self.rewards, self.game.is_terminal(), truncated, self._get_info()


    def render(self, q_values):
        """
            Return RGB image but this one will not be changed by wrappers
        """
        image = cv2.cvtColor(self.game.render(), cv2.COLOR_RGBA2RGB)
        dashboard = np.zeros(image.shape, dtype=np.uint8)
        dashboard.fill(255)


        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (10, image.shape[1]-10)
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
                 f"Reward: {self.reward}",
                 f"Action: {inv_action_space[self.action_space[self.action1]]}"]

        for text in texts[::-1]:
            dashboard = cv2.putText(dashboard, text, org, font, fontScale, color, thickness, cv2.LINE_AA, False)
            org = (org[0], org[1] - spacing)

        image = cv2.hconcat([dashboard, image])
        return image
