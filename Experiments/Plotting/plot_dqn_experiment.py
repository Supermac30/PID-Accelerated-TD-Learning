import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import hydra

from torch.utils.tensorboard import SummaryWriter


@hydra.main(config_path="../../config/DQNExperiments", config_name="DQNExperiment")
def graph_experiment(cfg):
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

    directory = cfg['directory']
    target_reward = cfg['env']['target_reward']
    environment_name = cfg['env']['env']

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
    total_ax.set_title(f"{environment_name}")
    # Set the x-axis label
    total_ax.set_xlabel("Episode")
    # Set the y-axis label
    total_ax.set_ylabel("Mean Reward")
    # Set the legend
    total_ax.legend(loc="lower right")

    # Save and show the plot:
    total_fig.savefig(f"{directory}/plot")
    total_fig.show()
