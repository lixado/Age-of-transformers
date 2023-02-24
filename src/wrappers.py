import gym

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
        total_reward = 0.0
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
        This wrapper will do the same action for x number of frames
        Return only every `skip`-th frame
        """
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        done = False

        obs, reward, done, truncated, info = self.env.step(action) # do action
        total_reward += reward
        for _ in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, truncated, info = self.env.step(-1) # do nothing for skip frames
            total_reward += reward
            if done or truncated:
                break
        return obs, total_reward, done, truncated, info