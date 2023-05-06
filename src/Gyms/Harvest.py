import cv2
import numpy as np
from constants import inv_action_space
from DeepRTS import Engine, Constants
import random
from Gyms.CustomGym import CustomGym
from functions import PlayerState

MAP = "10x10-2p-ffa-Eblil.json"


#['add_unit', 'do_action', 'do_manual_action', 'evaluate_player_state', 'food', 'food_consumption', 'get_id',
# 'get_name', 'get_score', 'get_targeted_unit', 'gold', 'left_click', 'lumber', 'num_archer', 'num_barrack',
# 'num_farm', 'num_footman', 'num_peasant', 'num_town_hall', 'right_click', 'set_name', 'set_state',
# 'set_targeted_unit_id', 'spawn_unit', 'statistic_damage_done', 'statistic_damage_taken', 'statistic_gathered_gold',
# 'statistic_gathered_lumber', 'statistic_gathered_stone', 'statistic_units_created', 'stone']
def harvest_reward(player0, previousPlayer0: PlayerState, player1, ticks):
    reward = -1

    #Building rewards
    if player0.num_town_hall > previousPlayer0.num_town_hall:
        reward += 100
    if player0.num_barrack > previousPlayer0.num_barrack:
        reward += 100
    if player0.num_peasant > previousPlayer0.num_peasant:
        reward += 100
    if player0.num_footman > previousPlayer0.num_footman:
        reward += 100

    #Attack reward
    if player0.evaluate_player_state() != Constants.PlayerState.Defeat and player1.evaluate_player_state() == Constants.PlayerState.Defeat:
        reward += 10000 / ticks
    if player0.evaluate_player_state() == Constants.PlayerState.Defeat and player1.evaluate_player_state() != Constants.PlayerState.Defeat:
        reward += -0.001 * ticks
    if player0.statistic_damage_done > previousPlayer0.statistic_damage_done and player1.statistic_damage_taken > 0:
        reward += 1000 / ticks
    return reward


class HarvestGym(CustomGym):
    def __init__(self, max_episode_steps, shape):
        engineConfig: Engine.Config = Engine.Config().defaults()
        engineConfig.set_gui("Blend2DGui")
        engineConfig.set_instant_building(True)
        engineConfig.set_harvest_forever(False)
        engineConfig.set_barracks(True)
        engineConfig.set_farm(True)
        engineConfig.set_footman(True)
        engineConfig.set_start_lumber(5000)
        engineConfig.set_start_gold(5000)
        engineConfig.set_start_stone(5000)

        self.action_space = [i for i in range(1, 17)]  # 1-16, all actions, (see deep-rts/bindings/Constants.cpp)

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

        reward = harvest_reward(self.player0, self.previousPlayer0, self.player1, self.elapsed_steps)

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
                 f"Q_Build1: {q_values[13]}",
                 f"Q_Build2: {q_values[14]}",
                 f"Reward: {reward}",
                 f"Action: {inv_action_space[self.action_space[self.action]]}"]

        for text in texts[::-1]:
            dashboard = cv2.putText(dashboard, text, org, font, fontScale, color, thickness, cv2.LINE_AA, False)
            org = (org[0], org[1] - spacing)
        image = cv2.hconcat([dashboard, image])
        return image