import random

import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
total_score = 0

episodes = 20
for episode in range(episodes):
    observation, info = env.reset()
    done = False
    score = 0

    while not done:
        if observation[2] > 0:
            if observation[2] > .12 or observation[3] > -0.3:
                action = 1
            else:
                action = 0
        else:
            if observation[2] < -0.12 or observation[3] < 0.3:
                action = 0
            else:
                action = 1
        # action = random.randint(0, 1)

        observation, reward, terminated, truncated, info = env.step(action)
        score += reward
        env.render()

        if terminated or truncated:
            done = True

    print(f"Episode: {episode}, Score: {score}")
    total_score += score

env.close()

print(f"Average Score: {total_score/episodes}")