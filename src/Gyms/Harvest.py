import cv2
import numpy as np
from constants import inv_action_space
from DeepRTS import Engine, Constants

from Gyms.CustomGym import CustomGym
from functions import PlayerState

MAP = "10x10-2p-ffa-Eblil.json"


def harvest_reward(player0, previousPlayer0: PlayerState, ticks):
    reward = 0
    target = player0.get_targeted_unit()
    if target is None or (target is not None and target.can_move is False):
        reward -= 0.1

    # Rewards
    if player0.statistic_gathered_stone > previousPlayer0.statistic_gathered_stone:
        reward += 1
    if player0.statistic_gathered_gold > previousPlayer0.statistic_gathered_gold:
        reward += 1
    if player0.statistic_gathered_lumber > previousPlayer0.statistic_gathered_lumber:
        reward += 1
    if player0.num_town_hall > previousPlayer0.num_town_hall:
        reward += 1
    if player0.num_peasant > previousPlayer0.num_peasant:
        reward += 1
    return reward


class HarvestGym(CustomGym):
    def __init__(self, max_episode_steps, shape):
        engineConfig: Engine.Config = Engine.Config().defaults()
        engineConfig.set_gui("Blend2DGui")
        engineConfig.set_instant_building(True)
        engineConfig.set_barracks(True)
        engineConfig.set_farm(True)
        engineConfig.set_footman(True)
        engineConfig.set_start_lumber(1000)
        engineConfig.set_start_gold(1000)
        engineConfig.set_start_stone(1000)

        self.action_space = [i for i in range(1, 17)]  # 1-16, all actions, (see deep-rts/bindings/Constants.cpp)

        super().__init__(max_episode_steps, shape, MAP, engineConfig)

        self.previousPlayer0 = PlayerState(self.player0)

    def step(self, actionIndex):
        self.elapsed_steps += 1
        self.action = actionIndex

        self.player0.do_action(self.action_space[actionIndex])

        self.game.update()

        reward = harvest_reward(self.player0, self.previousPlayer0, self.elapsed_steps)

        return self._get_obs(), reward, self.game.is_terminal(), False, self._get_info()

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
                 f"player0.gathered_stone: {self.player0.statistic_gathered_stone}",
                 f"player0.gathered_gold: {self.player0.statistic_gathered_gold}",
                 f"player0.gathered_lumber: {self.player0.statistic_gathered_lumber}",
                 f"Q_values:",
                 f"Q_Prev_Unit: {q_values[0]}",
                 f"Q_Next_Unit: {q_values[1]}",
                 f"Q_Left: {q_values[2]}",
                 f"Q_Right: {q_values[3]}",
                 f"Q_Up: {q_values[4]}",
                 f"Q_Down: {q_values[5]}",
                 f"Q_Attack: {q_values[10]}",
                 f"Q_Harvest: {q_values[11]}",
                 f"Q_Build0: {q_values[12]}",
                 f"Reward: {reward}",
                 f"Action: {inv_action_space[self.action_space[self.action]]}"]

        for text in texts[::-1]:
            dashboard = cv2.putText(dashboard, text, org, font, fontScale, color, thickness, cv2.LINE_AA, False)
            org = (org[0], org[1] - spacing)
        image = cv2.hconcat([dashboard, image])
        return image