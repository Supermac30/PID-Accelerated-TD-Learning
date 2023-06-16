import gymnasium as gym
import torch as th
from TabularPID.MDPs.GymWrapper import create_gym_wrapper
from TabularPID.Agents.DQN.DQN import PID_DQN
import logging

def build_PID_DQN(kp, ki, kd, alpha, beta, env_name, gamma, optimizer, replay_memory_size, batch_size,
                  learning_rate, tau, initial_eps, exploration_fraction, minimum_eps, epsilon_decay_step,
                  train_step, target_update_interval, d_tau, inner_size, slow_motion, tensorboard_log=None, seed=42,
                  adapt_gains=False, meta_lr=0.1, epsilon=0.1):
    """Build the PID DQN agent
    """
    env = create_gym_wrapper(env_name, slow_motion=slow_motion)
    optimizer_class = create_optimizer(optimizer)

    return PID_DQN(
        kp, ki, kd, alpha, beta, d_tau, adapt_gains, meta_lr, epsilon,
        policy='MlpPolicy',
        env=env,
        learning_rate=learning_rate,
        buffer_size=replay_memory_size,
        learning_starts=batch_size, # Buffer size starts at batch size
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        gradient_steps=train_step,
        target_update_interval=target_update_interval,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=initial_eps,
        exploration_final_eps=minimum_eps,
        tensorboard_log=tensorboard_log,
        policy_kwargs=dict(net_arch=[inner_size, inner_size],
                           optimizer_class=optimizer_class),
        seed=seed
    )

def create_optimizer(optimizer):
    if optimizer == 'Adam':
        return th.optim.Adam
    elif optimizer == 'RMSprop':
        return th.optim.RMSprop
    elif optimizer == 'SGD':
        return th.optim.SGD
    else:
        raise NotImplementedError