import torch as th
import os

from TabularPID.MDPs.GymWrapper import create_environment
from TabularPID.Agents.DQN.DQN import PID_DQN
from TabularPID.Agents.DQN.FQI_DQN import PID_FQI
from TabularPID.Agents.DQN.DQN_gain_adapter import NoGainAdapter, SingleGainAdapter, DiagonalGainAdapter, NetworkGainAdapter
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import get_latest_run_id


def build_PID_DQN(gain_adapter, env_name, gamma, optimizer, replay_memory_size, batch_size,
                  learning_rate, tau, initial_eps, exploration_fraction, minimum_eps,
                  gradient_steps, train_freq, target_update_interval, d_tau, inner_size,
                  slow_motion, learning_starts, tabular_d=False, tensorboard_log=None, seed=42,
                  log_name="", name_append="", should_stop=False, device="cuda", dump_buffer=False):
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
        optimize_memory_usage=optimize_memory_usage,
        learning_starts=learning_starts,
        tensorboard_log=tensorboard_log,
        policy_kwargs=dict(net_arch=[inner_size, inner_size],
                           optimizer_class=optimizer_class),
        seed=seed,
        should_stop=should_stop,
        device=device,
        dump_buffer=dump_buffer
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


def build_PID_FQI(gain_adapter, env_name, gamma, optimizer, replay_memory_size, batch_size,
                  learning_rate, initial_eps, exploration_fraction, minimum_eps,
                  gradient_steps, train_freq, target_update_interval, inner_size,
                  slow_motion, learning_starts, tabular_d=False, tensorboard_log=None, seed=42,
                  log_name="", name_append="", should_stop=False, device="cuda"):
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
        device=device
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


def build_gain_adapter(adapter_type, kp, ki, kd, alpha, beta, meta_lr, epsilon, use_previous_BRs):
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

    return gain_adapter(kp, ki, kd, alpha, beta, meta_lr, epsilon, use_previous_BRs)

def create_optimizer(optimizer):
    if optimizer == 'Adam':
        return th.optim.Adam
    elif optimizer == 'RMSprop':
        return th.optim.RMSprop
    elif optimizer == 'SGD':
        return th.optim.SGD
    else:
        raise NotImplementedError