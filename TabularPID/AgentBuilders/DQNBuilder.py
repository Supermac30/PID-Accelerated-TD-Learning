import gymnasium as gym
from TabularPID.Agents.DQN import PID_DQN

def build_PID_DQN(kp, ki, kd, alpha, beta, env_name, gamma, optimizer, replay_memory_size, batch_size,
                  learning_rate, tau, epsilon, epsilon_decay, epsilon_min, epsilon_decay_step, train_step, D_tau, inner_size):
    """Build the PID DQN agent
    """
    env = gym.make(env_name, render_mode="rgb_array")
    return PID_DQN(
        kp, ki, kd, alpha, beta,
        env,
        gamma,
        optimizer,
        replay_memory_size,
        batch_size,
        learning_rate,
        tau,
        epsilon,
        epsilon_decay,
        epsilon_min,
        epsilon_decay_step,
        train_step,
        D_tau=D_tau,
        inner_size=inner_size
    )


def build_adaptive_DQN(kp, ki, kd, alpha, beta, env_name, gamma, optimizer, replay_memory_size, batch_size,
                       learning_rate, tau, epsilon, epsilon_decay, epsilon_min, epsilon_decay_step, train_step, meta_lr, D_tau, inner_size):
    """Build the PID DQN agent
    """
    env = gym.make(env_name, render_mode="rgb_array")
    return PID_DQN(
        kp, ki, kd, alpha, beta,
        env,
        gamma,
        optimizer,
        replay_memory_size,
        batch_size,
        learning_rate,
        tau,
        epsilon,
        epsilon_decay,
        epsilon_min,
        epsilon_decay_step,
        train_step,
        adapt_gains=True,
        meta_lr=meta_lr,
        D_tau=D_tau,
        inner_size=inner_size
    )