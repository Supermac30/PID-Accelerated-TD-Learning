"""
Code that takes the (state, action) pairs in the replay buffer in models/$env/buffer.npy
and runs the trained model in models/$env/$env.zip on it,
getting a Q-value for each (state, action) pair by finding the discounted
reward of a monte carlo simulation.

The ((state, action), Q value) pairs are then saved in models/$env/bufferQValues.npy
"""

import gymnasium as gym
import numpy as np
import torch as th


def run_simulation(model, env, state, action, gamma, seed):
    """
    Runs a monte carlo simulation starting from the given state and action,
    returning the discounted reward.
    """
    env.reset()
    env.seed(seed)
    env.env.state = state
    env.step(action)
    total_reward = 0
    discount = 1
    while True:
        action = model.predict(env.env.state.reshape(1, -1))[0]
        state, reward, done, _ = env.step(action)
        total_reward += discount * reward
        discount *= gamma
        if done:
            break
    return total_reward


@hydra.main(config_path='../../config/DQNExperiments', config_name='DQNExperiment')
def main(cfg):
    env_name = cfg['env']
    env = gym.make(env_name)
    model = th.load(f'models/{env_name}/{env_name}.zip')
    buffer = np.load(f'models/{env_name}/buffer.npy')
    bufferQValues = np.zeros(len(buffer))
    for i in range(len(buffer)):
        bufferQValues[i] = run_simulation(model, env, buffer[i][0], buffer[i][1], cfg['gamma'], cfg['seed'])
    np.save(f'models/{env_name}/bufferQValues.npy', bufferQValues)


if __name__ == '__main__':
    main()