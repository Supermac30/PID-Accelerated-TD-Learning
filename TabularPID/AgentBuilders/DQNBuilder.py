import gym
from TabularPID.Agents.DQN import PID_DQN

def build_PID_DQN(kp, ki, kd, alpha, beta, env_name, gamma, optimizer, replay_memory_size, batch_size,
                  learning_rate, target_net_update_steps, epsilon,
                  epsilon_decay, epsilon_min, epsilon_decay_step, train_step):
    """Build the PID DQN agent
    """
    env = gym.make(env_name)
    return PID_DQN(
        kp, ki, kd, alpha, beta,
        env,
        gamma,
        optimizer,
        replay_memory_size,
        batch_size,
        learning_rate,
        target_net_update_steps,
        epsilon,
        epsilon_decay,
        epsilon_min,
        epsilon_decay_step,
        train_step
    )
