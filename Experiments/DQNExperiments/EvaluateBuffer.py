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
import pickle
import logging

import globals
from Experiments.ExperimentHelpers import get_model

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


def set_random_seed(env, seed):
    """
    A hacky way to set the random seed of an environment without resetting the state.
    Gymnasium really doesn't make it easy to do this.
    """
    unwrapped_env = env.unwrapped
    super(type(unwrapped_env), unwrapped_env).reset(seed=seed)


@hydra.main(version_base=None, config_path='../../config/DQNExperiments', config_name='DQNExperiment')
def main(cfg):
    env_cfg = next(iter(cfg['env'].values()))
    env_name = env_cfg['env']
    env = gym.make(env_name)
    model = get_model(env_name)
    buffer = pickle.load(open(f'{globals.base_directory}/models/{env_name}/buffer.pkl', 'rb'))
    # Create a numpy array that stores (state, action, q_value) tuples, where state is of shape (env.observation_space.shape[0],) and action is a scalar
    bufferQValues = np.zeros((len(buffer), env.observation_space.shape[0] + 2))

    seed_generator = np.random.default_rng(cfg['seed'])

    for i, state, action in enumerate(buffer):
        q_value = np.mean([
            run_simulation(model, env, state, action, env_cfg['gamma'], seed_generator.integers(0, 2**32 - 1))
            for _ in range(cfg['num_simulations'])
        ])
        bufferQValues[i] = np.array([*state, action, q_value])
    np.save(f'{globals.base_directory}/models/{env_name}/bufferQValues.npy', bufferQValues)
    logging.info(f'Saved buffer Q values to {globals.base_directory}/models/{env_name}/bufferQValues.npy')


if __name__ == '__main__':
    main()