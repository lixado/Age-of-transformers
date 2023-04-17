import gym
import numpy as np
from DeepRTS import Engine, Constants
from gym.spaces import Box

class CustomGym(gym.Env):
    def __init__(self, max_episode_steps, shape, game_map, config):
        self.max_episode_steps = max_episode_steps
        self.elapsed_steps = None

        self.initial_shape = shape
        print("Initial_shape: ", self.initial_shape)
        """
            Start engine
        """
        self.game: Engine.Game = Engine.Game(game_map, config)
        self.game.set_max_fps(0)  # 0 = unlimited
        # add 2 players
        self.player0: Engine.Player = self.game.add_player()
        self.player1: Engine.Player = self.game.add_player()

        self.game.start()
        self.observation_space = Box(low=-1., high=20., shape=self.initial_shape, dtype=np.uint8)
        self.render_shape = self.game.render().shape

    def _get_obs(self):
        stateResized = np.resize(np.ndarray.flatten(self.game.state), self.initial_shape)
        return stateResized


    def _get_info(self):
        return {}


    def reset(self):
        self.elapsed_steps = 0
        self.game.reset()

        return self._get_obs(), self._get_info()
    

    def close(self):
        self.game.stop()
