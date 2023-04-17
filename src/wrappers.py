import gym
import numpy as np
class RepeatFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """
        This wrapper will do the same action for x number of frames
        Return only every `skip`-th frame
        """
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = np.array([0.0, 0.0])
        done = False
        for _ in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            if done or truncated:
                break
        return obs, total_reward, done, truncated, info
    
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """
        This wrapper will do no action for x amount of times and then do the asked action
        Return only every `skip`-th frame
        """
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = np.array([0.0, 0.0])
        done = False
        for _ in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, truncated, info = self.env.step([-1 for _ in range(len(action))]) # do nothing for skip frames
            total_reward += reward
            if done or truncated:
                break

        obs, reward, done, truncated, info = self.env.step(action) # do action
        return obs, total_reward+reward, done, truncated, info