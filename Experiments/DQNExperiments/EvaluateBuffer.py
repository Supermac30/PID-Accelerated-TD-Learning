import gymnasium as gym
import hydra
import pickle
import logging

import globals
from TabularPID.OptimalRates.EvaluateBuffer import *
from TabularPID.AgentBuilders.DQNBuilder import get_model

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

    for i, (wrapped_env, action) in enumerate(buffer):
        env = wrapped_env.envs[0]
        q_value = np.mean([
            run_simulation(model, env, action, env_cfg['gamma'], seed_generator.integers(0, 2**32 - 1))
            for _ in range(cfg['num_simulations'])
        ])
        bufferQValues[i] = np.array([*env.unwrapped.state, action, q_value])
    np.save(f'{globals.base_directory}/models/{env_name}/bufferQValues.npy', bufferQValues)
    logging.info(f'Saved buffer Q values to {globals.base_directory}/models/{env_name}/bufferQValues.npy')


if __name__ == '__main__':
    main()