import torch as th
import os

import globals
import gymnasium as gym
from TabularPID.MDPs.GymWrapper import create_environment
from stable_baselines3.dqn import DQN as unmodified_DQN
from TabularPID.Agents.DQN.DQN import PID_DQN
from TabularPID.Agents.DQN.FQI_DQN import PID_FQI
from TabularPID.Agents.DQN.DQN_gain_adapter import NoGainAdapter, SingleGainAdapter, DiagonalGainAdapter, NetworkGainAdapter
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import get_latest_run_id


def build_PID_DQN(gain_adapter, env_name, gamma, optimizer, replay_memory_size, batch_size,
                  learning_rate, tau, initial_eps, exploration_fraction, minimum_eps,
                  gradient_steps, train_freq, target_update_interval, d_tau, inner_size,
                  slow_motion, learning_starts, tabular_d=False, tensorboard_log=None, seed=42,
                  log_name="", name_append="", should_stop=False, device="cuda", dump_buffer=False,
                  is_double=False, visualize=False, policy_evaluation=False):
    """Build the PID DQN agent
    """
    env, is_atari, stopping_criterion = create_environment(env_name, slow_motion=slow_motion)
    if visualize:
        env = gym.wrappers.RecordVideo(env, 'video', episode_trigger = lambda x: x % 25 == 0)
    optimizer_class = create_optimizer(optimizer)

    if is_atari:
        policy_type = "CnnPolicy"
    else:
        policy_type = "MlpPolicy"

    dqn = PID_DQN(
        d_tau, stopping_criterion, tabular_d, gain_adapter,
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
        optimize_memory_usage=False,
        learning_starts=learning_starts,
        tensorboard_log=tensorboard_log,
        policy_kwargs=dict(net_arch=[inner_size, inner_size],
                           optimizer_class=optimizer_class),
        seed=seed,
        should_stop=should_stop,
        device=device,
        dump_buffer=dump_buffer,
        is_double=is_double,
        optimal_model=get_model(env_name),
        policy_evaluation=policy_evaluation
    )

    gain_adapter.set_model(dqn)

    if log_name == "":
        latest_run_id = get_latest_run_id("tensorboard", "run")
        log_name = f"tensorboard_{latest_run_id + 1}"
    log_name += f"/{name_append}"

    dqn.set_logger(
        configure(
            folder=os.path.join(tensorboard_log, log_name),
            format_strings=["tensorboard"],
        )
    )

    return dqn

def build_PID_FQI(gain_adapter, env_name, gamma, optimizer, replay_memory_size, batch_size,
                  learning_rate, initial_eps, exploration_fraction, minimum_eps,
                  gradient_steps, train_freq, target_update_interval, inner_size,
                  slow_motion, learning_starts, tabular_d=False, tensorboard_log=None, seed=42,
                  log_name="", name_append="", should_stop=False, device="cuda", visualize=False, policy_evaluation=False):
    """Build the PID DQN agent
    """
    env, is_atari, stopping_criterion = create_environment(env_name, slow_motion=slow_motion)
    if visualize:
        env = gym.wrappers.RecordVideo(env, 'video', episode_trigger = lambda x: x % 25 == 0)
    optimizer_class = create_optimizer(optimizer)

    if is_atari:
        policy_type = "CnnPolicy"
        optimize_memory_usage = True
    else:
        policy_type = "MlpPolicy"
        optimize_memory_usage = False

    dqn = PID_FQI(
        stopping_criterion, gain_adapter,
        policy=policy_type,
        env=env,
        learning_rate=learning_rate,
        buffer_size=replay_memory_size,
        batch_size=batch_size,
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
        device=device,
        optimal_model=get_model(env_name),
        policy_evaluation=policy_evaluation
    )
    
    gain_adapter.set_model(dqn)

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

def get_model(env_name):
    """Return the model with the same env_name from the models directory"""
    model_dir = list(filter(lambda x: x.startswith(env_name), os.listdir(f"{globals.base_directory}/models")))
    if model_dir == []:
        return None
    model_dir = model_dir[0]
    return unmodified_DQN.load(f"{globals.base_directory}/models/{model_dir}/{model_dir}.zip")

def build_gain_adapter(adapter_type, kp, ki, kd, alpha, beta, meta_lr, epsilon, use_previous_BRs, meta_lr_p=-1, meta_lr_I=-1, meta_lr_d=-1):
    if adapter_type == "NoGainAdapter":
        gain_adapter = NoGainAdapter
    elif adapter_type == "SingleGainAdapter":
        gain_adapter = SingleGainAdapter
    elif adapter_type == "DiagonalGainAdapter":
        gain_adapter = DiagonalGainAdapter
    elif adapter_type == "NetworkGainAdapter":
        gain_adapter = NetworkGainAdapter
    else:
        raise NotImplementedError

    return gain_adapter(kp, ki, kd, alpha, beta, meta_lr, epsilon, use_previous_BRs, meta_lr_p=meta_lr_p, meta_lr_d=meta_lr_d, meta_lr_i=meta_lr_I)

def create_optimizer(optimizer):
    if optimizer == 'Adam':
        return th.optim.Adam
    elif optimizer == 'RMSprop':
        return th.optim.RMSprop
    elif optimizer == 'SGD':
        return th.optim.SGD
    else:
        raise NotImplementedError
