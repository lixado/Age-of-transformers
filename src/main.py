import gymnasium as gym
from gymnasium.utils.save_video import save_video
from logger import Logger
from agents.ddqn import DDQN_Agent
import copy

logger = Logger()

env = gym.make("BipedalWalker-v3", hardcore=True, render_mode="rgb_array_list")

agent = DDQN_Agent(env.observation_space.shape, len(env.action_space), logger.getSaveSolver())


step_starting_index = 0
episode_index = 0
observation, info = env.reset()
step_index = 0
while episode_index < 10:
    prev_observation = copy.copy(observation)
    step_index += 1
    action = env.action_space.sample()
    print(env.action_space)


    observation, reward, terminated, truncated, info = env.step(action)

    # Remember
    #agent.cache(prev_observation, observation, actionId, reward, (terminated or truncated))
    # Learn
    #q, loss = agent.learn()

    if terminated or truncated:
        save_video(
         env.render(),
         "videos",
         fps=env.metadata["render_fps"],
         step_starting_index=step_starting_index,
         episode_index=episode_index
        )
        r = []
        step_starting_index = step_index + 1
        episode_index += 1
        observation, info = env.reset()



env.close()