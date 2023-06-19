import matplotlib.pyplot as plt
import hydra

from Experiments.ExperimentHelpers import *
from TabularPID.AgentBuilders.DQNBuilder import build_PID_DQN

@hydra.main(version_base=None, config_path="../../config/DQNExperiments", config_name="DQNExperiment")
def control_experiment(cfg):
    """Attempt to replicate results in figure 2 of PID Accelerated VI"""
    fig, ax = plt.subplots(1, 1)
    fig_clean, ax_clean = plt.subplots(1, 1)
    seed = pick_seed(cfg['seed'])

    for kp, kd, ki, alpha, beta, optimizer, \
        replay_memory_size, batch_size, learning_rate, \
        tau, initial_eps, exploration_fraction, \
        minimum_eps, gradient_steps, train_freq, target_update_interval, d_tau, \
        inner_size, adapt_gains, epsilon, meta_lr, learning_starts in \
        zip(cfg['kp'], cfg['kd'], cfg['ki'], cfg['alpha'], cfg['beta'], cfg['optimizer'], \
            cfg['replay_memory_size'], cfg['batch_size'], cfg['learning_rate'], \
            cfg['tau'], cfg['initial_eps'], cfg['exploration_fraction'], \
            cfg['minimum_eps'], cfg['gradient_steps'], cfg['train_freq'], cfg['target_update_interval'], \
            cfg['d_tau'], cfg['inner_size'], cfg['adapt_gains'], cfg['epsilon'], cfg['meta_lr'], cfg['learning_starts']):
        agent = build_PID_DQN(
            kp, ki, kd, alpha, beta, 
            cfg['env'], cfg['gamma'], optimizer,
            replay_memory_size, batch_size, learning_rate,
            tau, initial_eps, exploration_fraction,
            minimum_eps, gradient_steps, train_freq, target_update_interval,
            d_tau, inner_size, cfg['slow_motion'], learning_starts,
            tensorboard_log=cfg['tensorboard_log'], seed=seed,
            adapt_gains=adapt_gains, meta_lr=meta_lr, epsilon=epsilon
        )

        total_history = np.zeros((cfg['num_iterations'],))
        for _ in range(cfg['num_runs']):
            agent = agent.learn(
                total_timesteps=cfg['num_iterations'],
                log_interval=cfg['log_interval'],
                progress_bar=cfg['progress_bar'],
                tb_log_name=f"kp={kp}kd={kd}ki={ki}alpha={alpha}beta={beta}" + (f"epsilon={epsilon}meta_lr={meta_lr}" if adapt_gains else "")
            )
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