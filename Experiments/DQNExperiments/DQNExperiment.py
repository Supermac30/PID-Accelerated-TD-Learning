import matplotlib.pyplot as plt
import hydra

import os

from Experiments.ExperimentHelpers import *
from TabularPID.AgentBuilders.DQNBuilder import build_PID_DQN
from TabularPID.Agents.DQN.DQN import PID_DQN
from TabularPID.MDPs.GymWrapper import GymWrapper

@hydra.main(version_base=None, config_path="../../config/DQNExperiments", config_name="DQNExperiment")
def control_experiment(cfg):
    """Attempt to replicate results in figure 2 of PID Accelerated VI"""
    fig, ax = plt.subplots(1, 1)
    fig_clean, ax_clean = plt.subplots(1, 1)
    seed = pick_seed(cfg['seed'])

    for kp, kd, ki, alpha, beta, optimizer, \
        replay_memory_size, batch_size, learning_rate, \
        tau, initial_eps, exploration_fraction, \
        minimum_eps, epsilon_decay_step, train_step, d_tau, inner_size in \
        zip(cfg['kp'], cfg['kd'], cfg['ki'], cfg['alpha'], cfg['beta'], cfg['optimizer'], \
            cfg['replay_memory_size'], cfg['batch_size'], cfg['learning_rate'], \
            cfg['tau'], cfg['initial_eps'], cfg['exploration_fraction'], \
            cfg['minimum_eps'], cfg['epsilon_decay_step'], cfg['train_step'], cfg['d_tau'], cfg['inner_size']):
        agent = build_PID_DQN(
            kp, ki, kd, alpha, beta, 
            cfg['env'], cfg['gamma'], optimizer,
            replay_memory_size, batch_size, learning_rate,
            tau, initial_eps, exploration_fraction,
            minimum_eps, epsilon_decay_step, train_step, d_tau, inner_size, cfg['slow_motion'],
            tensorboard_log=cfg['tensorboard_log'], seed=seed,
            adapt_gains=cfg['adapt_gains'], meta_lr=cfg['meta_lr'], epsilon=cfg['epsilon']
        )

        total_history = np.zeros((cfg['num_iterations'],))
        for _ in range(cfg['num_runs']):
            agent = agent.learn(
                total_timesteps=cfg['num_iterations'],
                log_interval=cfg['log_interval'],
                progress_bar=cfg['progress_bar'],
                tb_log_name=f"kp={kp} kd={kd} ki={ki} alpha={alpha} beta={beta}"
            )
            agent.logger.dump()
        """
        save_array(history, f"kp={kp} kd={kd} ki={ki} alpha={alpha} beta={beta}", ax, normalize=cfg['normalize'])
        clean_history = np.convolve(history, np.ones((cfg['num_average'],))/cfg['num_average'], mode='valid')
        save_array(clean_history, f"kp={kp} kd={kd} ki={ki} alpha={alpha} beta={beta}", ax_clean, normalize=cfg['normalize'])
        """
        agent.visualize_episode()
        

    """Plotting
    ax.title.set_text(f"PID-DQN: {cfg['env']}")
    ax.legend()
    if cfg['use_episodes']:
        ax.set_xlabel('Episode')
    else:
        ax.set_xlabel('Iteration')
    ax.set_ylabel('Reward')
    fig.savefig("plot")
    fig.show()

    ax_clean.title.set_text(f"PID-DQN Averaged over {cfg['num_average']}: {cfg['env']}")
    ax_clean.legend()
    if cfg['use_episodes']:
        ax.set_xlabel('Episode')
    else:
        ax.set_xlabel('Iteration')
    ax_clean.set_ylabel('Reward')
    fig_clean.savefig("CleanPlot")
    fig_clean.show()
    """


if __name__ == "__main__":
    control_experiment()