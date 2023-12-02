import gymnasium as gym

env = gym.make("Blackjack-v1", sab=True, render_mode="rgb_array")
# print(gym.envs.registry.keys())
done = False
observation, info = env.reset()

for _ in range(100):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
