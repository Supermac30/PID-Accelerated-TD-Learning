import matplotlib.pyplot as plt
import hydra
import pandas as pd

from Experiments.ExperimentHelpers import *
from TabularPID.AgentBuilders.DQNBuilder import build_PID_DQN

@hydra.main(version_base=None, config_path="../../config/DQNExperiments")
def control_experiment(cfg):
    """Attempt to replicate results in figure 2 of PID Accelerated VI"""
    global seed, environment_name, directory

    directory = os.getcwd()  # A small hack to get the current directory outside of hydra

    environment_name = cfg['env']
    if cfg['seed'] != -1:
        seed = cfg['seed']
    logging.info(f"The chosen seed is: {seed}")
    log_name = f"kp={cfg['kp']} kd={cfg['kd']} ki={cfg['ki']} alpha={cfg['alpha']} beta={cfg['beta']} d_tau={cfg['d_tau']} " \
          + (f"epsilon={cfg['epsilon']} meta_lr={cfg['meta_lr']}" if cfg['adapt_gains'] else "")
    agent = build_PID_DQN(
        cfg['kp'], cfg['ki'], cfg['kd'], cfg['alpha'], cfg['beta'], 
        cfg['env'], cfg['gamma'], cfg['optimizer'],
        cfg['replay_memory_size'], cfg['batch_size'], cfg['learning_rate'],
        cfg['tau'], cfg['initial_eps'], cfg['exploration_fraction'],
        cfg['minimum_eps'], cfg['gradient_steps'], cfg['train_freq'], cfg['target_update_interval'],
        cfg['d_tau'], cfg['inner_size'], cfg['slow_motion'], cfg['learning_starts'],
        tensorboard_log=cfg['tensorboard_log'], seed=seed,
        adapt_gains=cfg['adapt_gains'], meta_lr=cfg['meta_lr'], epsilon=cfg['epsilon'], log_name=log_name
    )

    for _ in range(cfg['num_runs']):
        agent = agent.learn(
            total_timesteps=cfg['num_iterations'],
            log_interval=cfg['log_interval'],
            progress_bar=cfg['progress_bar'],
            tb_log_name=log_name
        )
    agent.visualize_episode()

def graph_experiment():
    """Plots all the data in the directory tensorboard."""
    # Create a graph for plotting the rewards called fig and ax
    total_fig, total_ax = plt.subplots()

    # Loop through all directories in f"{directory}/tensorboard"
    for subdir in os.listdir(f"{directory}/tensorboard"):
        for file in os.listdir(f"{directory}/tensorboard/{subdir}"):
            # If the file is a csv file
            if file.endswith(".csv"):
                # Read the csv file
                df = pd.read_csv(f"{directory}/tensorboard/{subdir}/{file}")

                # Plot the data under the column 'ep_rew_mean'
                total_ax.plot(df['rollout/ep_rew_mean'], label=subdir)

                # If there is a column called train/k_p, plot that as well along with train_k_i and train_k_d on a separate graph
                if 'train/k_p' in df.columns:
                    fig = plt.figure(figsize=(10, 4))
                    gs = fig.add_gridspec(nrows=1, ncols=3, width_ratios=[1,1,1], wspace=0.3, hspace=0.5)

                    for i, gain in enumerate(['k_p', 'k_i', 'k_d']):
                        ax = fig.add_subplot(gs[0, i])
                        ax.plot(df[f"train/{gain}"], label=f"train_{gain}")
                        ax.set_xlabel('Iteration')
                        ax.set_ylabel(gain)
                        ax.legend()

                    plt.suptitle(f"Adaptive Agent: {subdir}")

                    # Set square aspect ratio for each subplot
                    for ax in fig.axes:
                        ax.set_box_aspect(1)

                    # Center the subplots in a single row and move up to remove whitespace
                    gs.tight_layout(fig, rect=[0.05, 0.08, 0.95, 0.95])

                    plt.savefig(f"{directory}/gains_plot_{subdir}.png")
                    plt.close()

    # Set the title of the graph
    total_ax.set_title(f"{environment_name}:{seed}")
    # Set the x-axis label
    total_ax.set_xlabel("Episode")
    # Set the y-axis label
    total_ax.set_ylabel("Mean Reward")
    # Set the legend
    total_ax.legend()

    # Save and show the plot:
    total_fig.savefig(f"{directory}/plot")
    total_fig.show()


if __name__ == "__main__":
    seed = np.random.randint(0, 1000000)
    environment_name = ""
    directory = ""

    control_experiment()
    graph_experiment()