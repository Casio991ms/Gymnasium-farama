import pickle

import numpy as np
from gymnasium import Env


def train(env: Env, episode_count: int,
          alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 1, epsilon_decay: float = 0.001):
    rng = np.random.default_rng()
    rewards = np.zeros(episode_count)

    q = np.zeros((env.observation_space.n, env.action_space.n))

    for episode in range(episode_count):
        state, info = env.reset()

        terminated = False
        truncated = False

        while not terminated and not truncated:
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state, :])

            new_state, reward, terminated, truncated, info = env.step(action)

            q[state, action] += alpha * (
                    reward + gamma * (np.max(q[new_state, :]) - q[state, action])
            )

            state = new_state

        epsilon *= (1 - epsilon_decay)
        rewards[episode] = reward

    env.close()

    file = open("frozen_lake.pkl", "wb")
    pickle.dump(q, file)
    file.close()


def view(env: Env, epsilon: float = 0.1):
    rng = np.random.default_rng()

    file = open("frozen_lake.pkl", "rb")
    q = pickle.load(file)
    file.close()

    print(env.desc)

    print_v_from_q(q, 4)

    state, info = env.reset()

    terminated = False
    truncated = False

    while not terminated and not truncated:
        if rng.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q[state, :])

        print(action)

        new_state, reward, terminated, truncated, info = env.step(action)
        state = new_state

    env.close()


def print_v_from_q(q, row_count):
    for state in range(len(q)):
        v = np.max(q[state, :])
        print(v, end=' ')
        if (state + 1) % row_count == 0:
            print()
