import gymnasium as gym

import QLearning


# env = gym.make("FrozenLake-v1", is_slippery=False)
# QLearning.train(env, 1000)

env2 = gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array")
env2 = gym.wrappers.RecordVideo(env2, 'video')
QLearning.view(env2, 0)
