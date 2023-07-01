import torch as th
import os

from TabularPID.MDPs.GymWrapper import create_environment
from TabularPID.Agents.DQN.DQN import PID_DQN
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import get_latest_run_id

def build_PID_DQN(kp, ki, kd, alpha, beta, env_name, gamma, optimizer, replay_memory_size, batch_size,
                  learning_rate, tau, initial_eps, exploration_fraction, minimum_eps,
                  gradient_steps, train_freq, target_update_interval, d_tau, inner_size,
                  slow_motion, learning_starts, tabular_d=False, tensorboard_log=None, seed=42,
                  adapt_gains=False, meta_lr=0.1, epsilon=0.1, log_name="",
                  name_append="", should_stop=False, diagonal_adaptation=False):
    """Build the PID DQN agent
    """
    env, is_atari, stopping_criterion = create_environment(env_name, slow_motion=slow_motion)
    optimizer_class = create_optimizer(optimizer)

    if is_atari:
        policy_type = "CnnPolicy"
        optimize_memory_usage = True
    else:
        policy_type = "MlpPolicy"
        optimize_memory_usage = False

    dqn = PID_DQN(
        kp, ki, kd, alpha, beta, d_tau, adapt_gains, meta_lr, epsilon, stopping_criterion, tabular_d,
        policy=policy_type,
        env=env,
        learning_rate=learning_rate,
        buffer_size=replay_memory_size,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        gradient_steps=gradient_steps,
        train_freq=train_freq,
        target_update_interval=target_update_interval,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=initial_eps,
        exploration_final_eps=minimum_eps,
        optimize_memory_usage=optimize_memory_usage,
        learning_starts=learning_starts,
        tensorboard_log=tensorboard_log,
        policy_kwargs=dict(net_arch=[inner_size, inner_size],
                           optimizer_class=optimizer_class),
        seed=seed,
        should_stop=should_stop,
        diagonal_adaptation=diagonal_adaptation
    )

    if log_name == "":
        latest_run_id = get_latest_run_id("tensorboard", "run")
        log_name = f"tensorboard_{latest_run_id + 1}"
    log_name += f"/{name_append}"

    dqn.set_logger(
        configure(
            folder=os.path.join(tensorboard_log, log_name),
            format_strings=["log", "tensorboard", "csv"],
        )
    )

    return dqn

def create_optimizer(optimizer):
    if optimizer == 'Adam':
        return th.optim.Adam
    elif optimizer == 'RMSprop':
        return th.optim.RMSprop
    elif optimizer == 'SGD':
        return th.optim.SGD
    else:
        raise NotImplementedError