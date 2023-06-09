import matplotlib.pyplot as plt
import hydra

from Experiments.ExperimentHelpers import *
from TabularPID.AgentBuilders.DQNBuilder import build_PID_DQN
from TabularPID.Agents.DQN import PID_DQN

@hydra.main(version_base=None, config_path="../../config/DQNExperiments", config_name="DQNExperiment")
def control_experiment(cfg):
    """Attempt to replicate results in figure 2 of PID Accelerated VI"""
    fig, ax = plt.subplots(1, 1)
    fig_clean, ax_clean = plt.subplots(1, 1)
    for kp, kd, ki, alpha, beta, optimizer, \
        replay_memory_size, batch_size, learning_rate, \
        tau, epsilon, epsilon_decay, \
        epsilon_min, epsilon_decay_step, train_step, D_tau, inner_size in \
        zip(cfg['kp'], cfg['kd'], cfg['ki'], cfg['alpha'], cfg['beta'], cfg['optimizer'], \
            cfg['replay_memory_size'], cfg['batch_size'], cfg['learning_rate'], \
            cfg['tau'], cfg['epsilon'], cfg['epsilon_decay'], \
            cfg['epsilon_min'], cfg['epsilon_decay_step'], cfg['train_step'], cfg['D_tau'], cfg['inner_size']):
        agent = build_PID_DQN(
            kp, ki, kd, alpha, beta,
            cfg['env'], cfg['gamma'], optimizer,
            replay_memory_size, batch_size, learning_rate,
            tau, epsilon, epsilon_decay,
            epsilon_min, epsilon_decay_step, train_step, D_tau, inner_size
        )

        if cfg['use_episodes']:
            total_history = np.zeros((cfg['num_episodes'],))
        else:
            total_history = np.zeros((cfg['num_iterations'],))
        for _ in range(cfg['num_runs']):
            history, _ = agent.rollout(max_num_iterations=cfg['num_iterations'], max_num_episodes=cfg['num_episodes'], use_episodes=cfg['use_episodes'], debug_num_steps=cfg['debug_num_steps'])
            total_history += history
        save_array(history, f"kp={kp} kd={kd} ki={ki} alpha={alpha} beta={beta}", ax, normalize=cfg['normalize'])
        clean_history = np.convolve(history, np.ones((cfg['num_average'],))/cfg['num_average'], mode='valid')
        save_array(clean_history, f"kp={kp} kd={kd} ki={ki} alpha={alpha} beta={beta}", ax_clean, normalize=cfg['normalize'])
        agent.visualize_episode()


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


if __name__ == "__main__":
    control_experiment()