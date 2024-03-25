import os
import numpy as np
import matplotlib.pyplot as plt
import hydra

from Experiments.ExperimentHelpers import *

@hydra.main(version_base=None, config_path="../../config", config_name="plot")
def create_plots(cfg):
    """Plot all of the data in the npy folder from the runs of the adaptive agent."""
    # Create a figure that will be used to plot the history of each agent
    fig0 = plt.figure()
    ax0 = fig0.add_subplot()

    # Min history
    min_final_history = np.inf
    min_history = None
    min_history_file = None
    min_std_dev = None

    max_y = 1
    sep = cfg['separation']
    x_axis = lambda n: np.arange(0, sep * len(n), sep)

    files = os.listdir(f"{cfg['save_dir']}/npy/mean")[::-1]
    # Change the order of files, so that if a file starts with 'TD' or with 'Q learning', it comes first
    files = sorted(files, key=lambda x: (not x.startswith('TD'), not x.startswith('Q learning')))

    # Iterate through all of the files in the npy folder
    for file in files:
        name = file[:-4]
        history = np.load(f"{cfg['save_dir']}/npy/mean/{file}")
        std_dev = np.load(f"{cfg['save_dir']}/npy/std_dev/{file}")
        # If the file starts with gain_history, plot it:
        if file.startswith("gain_history"):
            # GPT generated plotting code:
            # Create a figure and a single subplot
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            # Create a second y-axis for ki and kd (colored lines)
            ax2 = ax1.twinx()

            # Plot kp with its standard deviation on the left y-axis (colored line)
            gain_values_kp = history[:, 0]
            std_dev_values_kp = std_dev[:, 0]
            ax1.plot(x_axis(gain_values_kp), gain_values_kp, label='kp', color='blue')
            ax1.fill_between(x_axis(gain_values_kp), gain_values_kp - std_dev_values_kp, 
                            gain_values_kp + std_dev_values_kp, alpha=0.2, color='lightblue')
            ax1.set_xlabel('Steps')
            ax1.set_ylabel('kp', color='black')
            ax1.tick_params(axis='y', labelcolor='black')

            # Reduce the title font size
            ax1.set_title('PID Gain Values Over Time', fontsize=10)

            # Plot ki with its standard deviation on the right y-axis
            gain_values_ki = history[:, 1]
            std_dev_values_ki = std_dev[:, 1]
            ax2.plot(x_axis(gain_values_ki), gain_values_ki, label='ki', color='orange')
            ax2.fill_between(x_axis(gain_values_ki), gain_values_ki - std_dev_values_ki, 
                            gain_values_ki + std_dev_values_ki, alpha=0.2, color='moccasin')

            # Plot kd with its standard deviation on the right y-axis
            gain_values_kd = history[:, 2]
            std_dev_values_kd = std_dev[:, 2]
            ax2.plot(x_axis(gain_values_kd), gain_values_kd, label='kd', color='green')
            ax2.fill_between(x_axis(gain_values_kd), gain_values_kd - std_dev_values_kd, 
                            gain_values_kd + std_dev_values_kd, alpha=0.2, color='honeydew')

            ax2.set_ylabel('ki, kd', color='black')
            ax2.tick_params(axis='y', labelcolor='black')

            # Set square aspect ratio
            ax1.set_box_aspect(1)

            # Legend
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc='upper left')

            # Add grid lines
            ax1.margins(0)
            ax1.grid(True)
            #ax2.set_yticks(np.linspace(ax2.get_yticks()[0], ax2.get_yticks()[-1], len(ax1.get_yticks())))
            ax1.set_yticks(np.linspace(*(ax1.get_yticks()[[0,-1]]),6))
            ax2.set_yticks(np.linspace(*(ax2.get_yticks())[[0,-1]],6))
            ax2.grid(False)

            # Save the figure
            fig.savefig(f"{cfg['save_dir']}/gain_history_{name}.png")
            fig.savefig(f"{cfg['save_dir']}/gain_history_{name}.pdf")
            
        # Otherwise, plot the file on the history plot
        elif cfg['plot_best']:
            if history[-1] < min_final_history:
                min_final_history = history[-1]
                min_history_file = file
                min_history = history
                min_std_dev = std_dev
        else:
            if history[0] != 0:
                std_dev /= history[0]
            max_value = np.max(normalize(history) + std_dev)
            if max_value > 10:
                # Don't plot diverging runs
                continue
            max_y = max(max_y, max_value)
            ax0.plot(x_axis(history), normalize(history), label=name)
            ax0.fill_between(x_axis(history), normalize(history) - std_dev, normalize(history) + std_dev, alpha=0.2)

    if cfg['plot_best']:
        logging.info(f"Best final history: {min_history_file}")
        if cfg['is_double_q']:
            name = "Gain Adaptation Double Q Learning"
        elif cfg['is_q']:
            name = "Gain Adaptation Q Learning"
        else:
            name = "Gain Adaptation TD"
        if min_history[0] != 0:
            min_std_dev /= min_history[0]
        max_y = max(max_y, np.max(normalize(min_history) + min_std_dev))
        ax0.plot(x_axis(min_history), normalize(min_history), label=r"{}".format(name))
        ax0.fill_between(x_axis(min_history), normalize(min_history) - min_std_dev, normalize(min_history) + min_std_dev, alpha=0.2)

    ax0.title.set_text(f"{cfg['env'].title()}")
    ax0.set_xlabel('Steps')
    # ax0.set_ylim(0, min(2, max_y))
    ax0.legend()
    create_label(ax0, cfg['norm'], cfg['normalize'], cfg['is_q'], is_star=cfg['is_star'])

    if cfg['log_plot']:
        ax0.set_yscale('log')
    
    # Add grid lines
    ax0.grid()

    fig0.savefig(f"{cfg['save_dir']}/adaptive_agent_{cfg['env']}.pdf")
    fig0.savefig(f"{cfg['save_dir']}/adaptive_agent_{cfg['env']}.png")


if __name__ == "__main__":
    create_plots()