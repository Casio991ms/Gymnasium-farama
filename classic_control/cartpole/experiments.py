import gymnasium as gym
import numpy as np

env = gym.make("CartPole-v1")

best_average_score = 0
best_average_distance = 5
best_angle = 0
best_angle_velocity = 0

for angle in np.arange(0, 0.2, 0.01):
    for angle_velocity in np.arange(0, 2, 0.1):
        total_score = 0
        total_distance = 0
        episodes = 200
        for episode in range(episodes):
            observation, info = env.reset()
            done = False
            score = 0
            distance = 0

            while not done:
                if observation[2] > 0:
                    if observation[2] > angle or observation[3] > -angle_velocity:
                        action = 1
                    else:
                        action = 0
                else:
                    if observation[2] < -angle or observation[3] < angle_velocity:
                        action = 0
                    else:
                        action = 1

                observation, reward, terminated, truncated, info = env.step(action)
                score += reward
                distance += abs(observation[0])
                # env.render()

                if terminated or truncated:
                    done = True

            total_score += score
            total_distance += distance

        average_score = total_score/episodes
        average_distance = total_distance/episodes
        print(f"Angle: {angle}, Angular velocity: {angle_velocity}")
        print(f"Average Score: {average_score}")
        print(f"Average Distance: {average_distance}")
        print("----------------------------------------------------")

        if average_score > best_average_score or (average_score == best_average_score and average_distance < best_average_distance):
            best_average_score = average_score
            best_average_distance = average_distance
            best_angle = angle
            best_angle_velocity = angle_velocity

env.close()

print(f"Best Average Score: {best_average_score}")
print(f"Best Average Distance: {best_average_distance}")
print(f"Best Angle: {best_angle}")
print(f"Best Angular Velocity: {best_angle_velocity}")