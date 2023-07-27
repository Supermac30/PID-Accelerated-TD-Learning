"""
Code that takes the (state, action) pairs in the replay buffer in models/$env/buffer.npy
and runs the trained model in models/$env/$env.zip on it,
getting a Q-value for each (state, action) pair by finding the discounted
reward of a monte carlo simulation.

The ((state, action), Q value) pairs are then saved in models/$env/bufferQValues.npy
"""

import gymnasium as gym
import numpy as np
import hydra
import logging

from Experiments.DQNExperiments.DQNExperiment import get_model


base_directory = "/h/bedaywim/PID-Accelerated-TD-Learning"

def run_simulation(model, env, state, action, gamma, seed):
    """
    Runs a monte carlo simulation starting from the given state and action,
    returning the discounted reward.
    """
    env.reset(seed=int(seed))
    env.state = state
    env.step(int(action))
    total_reward = 0
    discount = 1

    done = False
    truncated = False
    while not (done or truncated):
        action = model.predict(env.state.reshape(1, -1))[0].item()
        state, reward, done, truncated, _ = env.step(action)
        total_reward += discount * reward
        discount *= gamma
    return total_reward


@hydra.main(version_base=None, config_path='../../config/DQNExperiments', config_name='DQNExperiment')
def main(cfg):
    env_cfg = next(iter(cfg['env'].values()))
    env_name = env_cfg['env']
    env = gym.make(env_name)
    model = get_model(env_name)
    buffer = np.load(f'{base_directory}/models/{env_name}/buffer.npy')
    bufferQValues = np.zeros(len(buffer))

    seed_generator = np.random.default_rng(cfg['seed'])

    for i in range(len(buffer)):
        state, action = buffer[i][:-1], buffer[i][-1]
        bufferQValues[i] = run_simulation(model, env, state, action, env_cfg['gamma'], seed_generator.integers(0, 2**32 - 1))
    np.save(f'{base_directory}/models/{env_name}/bufferQValues.npy', bufferQValues)
    logging.info(f'Saved buffer Q values to {base_directory}/models/{env_name}/bufferQValues.npy')


if __name__ == '__main__':
    main()