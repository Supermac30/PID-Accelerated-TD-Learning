import matplotlib.pyplot as plt
import hydra
import pandas as pd
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from wandb.integration.sb3 import WandbCallback
import wandb

from Experiments.ExperimentHelpers import *
from TabularPID.AgentBuilders.DQNBuilder import build_PID_DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import evaluation

@hydra.main(version_base=None, config_path="../../config/DQNExperiments", config_name="DQNExperiment")
def control_experiment(cfg):
    """Attempt to replicate results in figure 2 of PID Accelerated VI"""
    global seed, environment_name, directory, target_reward

    if cfg['seed'] != -1:
        seed = cfg['seed']
    logging.info(f"The chosen seed is: {seed}")

    # Create a prg with this seed
    prg = np.random.RandomState(seed)

    log_name = f"{cfg['kp']} {cfg['ki']} {cfg['kd']}{'*' if cfg['tabular_d'] else ''} {cfg['alpha']} {cfg['beta']} {cfg['d_tau']}" \
          + (f" {cfg['epsilon']} {cfg['meta_lr']}" if cfg['adapt_gains'] else "")

    env_cfg = next(iter(cfg['env'].values()))

    for i in range(cfg['num_runs']):
        agent, config = build_PID_DQN(
            cfg['kp'], cfg['ki'], cfg['kd'], cfg['alpha'], cfg['beta'], 
            env_cfg['env'], env_cfg['gamma'], env_cfg['optimizer'],
            env_cfg['replay_memory_size'], env_cfg['batch_size'], env_cfg['learning_rate'],
            env_cfg['tau'], env_cfg['initial_eps'], env_cfg['exploration_fraction'],
            env_cfg['minimum_eps'], env_cfg['gradient_steps'], env_cfg['train_freq'], env_cfg['target_update_interval'],
            cfg['d_tau'], env_cfg['inner_size'], cfg['slow_motion'], env_cfg['learning_starts'],
            tabular_d=cfg['tabular_d'], tensorboard_log=cfg['tensorboard_log'], seed=prg.randint(0, 2**32),
            adapt_gains=cfg['adapt_gains'], meta_lr=cfg['meta_lr'],
            epsilon=cfg['epsilon'], log_name=log_name, name_append=f"run {i}",
            should_stop=cfg['should_stop'], diagonal_adaptation=cfg['diagonal_adaptation'],
        )

        run = wandb.init(
            project="PID Accelerated VI",
            config=config,
            sync_tensorboard=True,
            moniter_gym=True,
            save_code=True,
        )

        agent = agent.learn(
            total_timesteps=env_cfg['num_iterations'],
            log_interval=cfg['log_interval'],
            progress_bar=cfg['progress_bar'],
            tb_log_name=log_name,
            callback=WandbCallback(
                model_save_path=f"{os.getcwd()}/models/{log_name}",
                verbose=2,
            ),
        )
        run.finish()

    agent.visualize_episode()

    # A small hack to get this info outside of hydra
    directory = os.getcwd()
    environment_name = env_cfg['env']
    target_reward = agent.stopping_criterion


class CustomEvalCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomEvalCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        if self.num_timesteps % 100 != 0:
            return True
        
        mean, std = evaluation.evaluate_policy(self.model, self.model.get_env(), n_eval_episodes=10, render=True)
        self.logger.record("eval/mean_reward", mean)
        self.logger.record("eval/std_reward", std)

        if self.model.update_gains:
            self.logger.record("eval/k_p", self.model.kp)
            self.logger.record("eval/k_i", self.model.ki)
            self.logger.record("eval/k_d", self.model.kd)


def graph_experiment(seed, environment_name, directory, target_reward):
    def add_arrays(array1, array2):
        if isinstance(array1, int):
            return array2
        # Get the sizes of the arrays
        size1 = array1.size
        size2 = array2.size

        # Truncate the larger array if needed
        if size1 > size2:
            array1 = array1[:size2]
        elif size2 > size1:
            array2 = array2[:size1]

        # Add the arrays
        result = array1 + array2

        return result

    def create_tensorboard_average(arr, directory, subdir, name):
        # If the directory doesn't exist, create it
        if not os.path.exists(f"{directory}/tensorboard"):
            os.makedirs(f"{directory}/tensorboard")

        # Create a tensorboard writer
        writer = SummaryWriter(f"{directory}/tensorboard/{subdir}")

        # Add the array to tensorboard by looping through it
        for i in range(arr.size):
            writer.add_scalar(f"{name}", arr[i], i)

    """Plots all the data in the directory tensorboard."""
    # Create a graph for plotting the rewards called fig and ax
    total_fig, total_ax = plt.subplots()

    # Loop through all directories in f"{directory}/tensorboard"
    for subdir in os.listdir(f"{directory}/tensorboard"):
        total_history = 0
        total_gain_history = {'k_p': 0, 'k_i': 0, 'k_d': 0}
        plot_gains = False
        count = 0
        for batch in os.listdir(f"{directory}/tensorboard/{subdir}"):
            for run in os.listdir(f"{directory}/tensorboard/{subdir}/{batch}"):
                if not run.endswith(".csv"):
                    continue
                df = pd.read_csv(f"{directory}/tensorboard/{subdir}/{batch}/{run}")
                
                # Make the x-axis jump up by 100s
                x_axis = np.arange(0, df['eval/mean_reward'].size * 100, 100)

                # Plot the gains
                if 'eval/k_p' in df.columns:
                    total_gain_history['k_p'] = add_arrays(total_gain_history['k_p'], np.array(df['eval/k_p']))
                    total_gain_history['k_i'] = add_arrays(total_gain_history['k_i'], np.array(df['eval/k_i']))
                    total_gain_history['k_d'] = add_arrays(total_gain_history['k_d'], np.array(df['eval/k_d']))
                    plot_gains = True

                count += 1
                total_history = add_arrays(total_history, np.array(df['eval/mean_reward']))

        # Truncate the x_axis to be the same size as total_history
        x_axis = x_axis[:total_history.size]

        smoothed_history = pd.DataFrame(total_history / count)
        total_ax.plot(smoothed_history[0].rolling(10).mean(), label=subdir)
        color = total_ax.lines[-1].get_color()
        total_ax.plot(x_axis, total_history / count, color=color, alpha=0.2)
        create_tensorboard_average(total_history / count, f"{directory}/average", subdir, "ep_rew")

        # Plot the gains, if they exist
        if plot_gains:
            fig = plt.figure(figsize=(10, 4))
            gs = fig.add_gridspec(nrows=1, ncols=3, width_ratios=[1,1,1], wspace=0.3, hspace=0.5)

            for i, gain in enumerate(['k_p', 'k_i', 'k_d']):
                ax = fig.add_subplot(gs[0, i])
                ax.plot(x_axis, total_gain_history[gain] / count, label=f"train_{gain}")
                ax.set_xlabel('Episode')
                ax.set_ylabel(gain)
                ax.legend()

                create_tensorboard_average(
                    total_gain_history[gain] / count,
                    f"{directory}/average",
                    subdir,
                    f"train_{gain}"
                )


            plt.suptitle(f"Adaptive Agent: {subdir}")

            # Set square aspect ratio for each subplot
            for ax in fig.axes:
                ax.set_box_aspect(1)

            # Center the subplots in a single row and move up to remove whitespace
            gs.tight_layout(fig, rect=[0.05, 0.08, 0.95, 0.95])

            plt.savefig(f"{directory}/gains_plot_{subdir}.png")
            plt.close()

    # Draw a dotted line at the target reward
    total_ax.axhline(y=target_reward, color='r', linestyle='--')

    # Set the title of the graph
    total_ax.set_title(f"{environment_name} seed:{seed}")
    # Set the x-axis label
    total_ax.set_xlabel("Episode")
    # Set the y-axis label
    total_ax.set_ylabel("Mean Reward")
    # Set the legend
    total_ax.legend(loc="lower right")

    # Save and show the plot:
    total_fig.savefig(f"{directory}/plot")
    total_fig.show()


if __name__ == "__main__":
    seed = np.random.randint(0, 1000000)
    environment_name = ""
    directory = ""
    target_reward = float("inf")

    control_experiment()
    graph_experiment(seed, environment_name, directory, target_reward)