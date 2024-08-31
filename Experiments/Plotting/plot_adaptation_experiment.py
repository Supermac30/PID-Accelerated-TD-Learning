import os
import numpy as np
import matplotlib.pyplot as plt
import hydra
from matplotlib.ticker import FuncFormatter
from math import sqrt

from Experiments.ExperimentHelpers import *

# Made by ChatGPT: Function to format the tick labels
def thousands_formatter(x, pos):
    return f'{int(x/1000)}k' if x >= 1000 else int(x)

def sorting_mechanism(files):
    """Order files so that a file starting with "TD" or "Q Learning" or "Zap Q Learning" or "Speedy Q Learning" or "TIDBD" goes after anything else,
    and sort those files in the corresponding order they appear afterwards.
    """
    td_files = []
    q_learning_files = []
    zap_q_learning_files = []
    speedy_q_learning_files = []
    tidbd_files = []
    other_files = []

    dont_compare = False

    for file in files:
        if file.startswith("TD"):
            td_files.append(file)
        elif file.startswith("Q Learning"):
            q_learning_files.append(file)
        elif file.startswith("Zap Q Learning"):
            zap_q_learning_files.append(file)
        elif file.startswith("Speedy Q Learning"):
            if dont_compare: continue
            speedy_q_learning_files.append(file)
        elif file.startswith("TIDBD"):
            if dont_compare: continue
            tidbd_files.append(file)
        else:
            other_files.append(file)

    return other_files + td_files + q_learning_files + zap_q_learning_files + speedy_q_learning_files + tidbd_files


@hydra.main(version_base=None, config_path="../../config", config_name="plot")
def create_plots(cfg):
    """Plot all of the data in the npy folder from the runs of the adaptive agent."""
    # Create a figure that will be used to plot the history of each agent
    # Min history
    repeat = cfg['repeat']
    min_final_history = np.inf
    min_history = None
    min_history_file = None
    min_std_dev = None

    titlefontsize = 15
    plt.rc('font', size=14)          # controls default text sizes
    plt.rc('axes', titlesize=14)     # fontsize of the axes title
    plt.rc('axes', labelsize=14)     # fontsize of the x and y labels
    plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=13)    # fontsize of the tick labels

    fig0 = plt.figure()
    ax0 = fig0.add_subplot()

    max_y = 1
    sep = cfg['separation']
    x_axis = lambda n: np.arange(0, sep * len(n), sep)
    
    files = os.listdir(f"{cfg['save_dir']}/npy/mean")[::-1]
    files = sorting_mechanism(files)
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

            ax1.set_xlabel('Steps')
            ax1.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
            ax1.set_ylabel('Gain Value', color='black')
            ax1.tick_params(axis='y', labelcolor='black')

            # Plot kp with its standard deviation on the left y-axis (colored line)
            gain_values_kp = history[:, 0]
            std_dev_values_kp = std_dev[:, 0]
            ax1.plot(x_axis(gain_values_kp), gain_values_kp - 1, label='$\kappa_p - 1$', color='blue')
            ax1.fill_between(x_axis(gain_values_kp), gain_values_kp - 1 - std_dev_values_kp / sqrt(repeat), 
                            gain_values_kp - 1 + std_dev_values_kp / sqrt(repeat), alpha=0.2, color='lightblue')

            # Reduce the title font size
            ax1.set_title('PID Gain Values Over Time', fontsize=titlefontsize)

            # Plot ki with its standard deviation on the right y-axis
            gain_values_ki = history[:, 1]
            std_dev_values_ki = std_dev[:, 1]
            ax1.plot(x_axis(gain_values_ki), gain_values_ki, label='$\kappa_I$', color='orange')
            ax1.fill_between(x_axis(gain_values_ki), gain_values_ki - std_dev_values_ki / sqrt(repeat), 
                            gain_values_ki + std_dev_values_ki / sqrt(repeat), alpha=0.2, color='moccasin')

            # Plot kd with its standard deviation on the right y-axis
            gain_values_kd = history[:, 2]
            std_dev_values_kd = std_dev[:, 2]
            ax1.plot(x_axis(gain_values_kd), gain_values_kd, label='$\kappa_d$', color='green')
            ax1.fill_between(x_axis(gain_values_kd), gain_values_kd - std_dev_values_kd / sqrt(repeat), 
                            gain_values_kd + std_dev_values_kd / sqrt(repeat), alpha=0.2, color='honeydew')

            ax1.tick_params(axis='y', labelcolor='black')

            # Set square aspect ratio
            ax1.set_box_aspect(1)

            # Legend
            lines, labels = ax1.get_legend_handles_labels()
            ax1.legend(lines, labels, loc='upper left')

            # start, end = ax1.get_yticks()[[0, -1]]
            # endpoint = max(- start, end)
            # ax1.set_yticks(np.linspace(-endpoint, endpoint, 7))
            ax1.margins(0)
            ax1.grid(True)

            # Asymmetric but space efficient layout:
            # ax1.margins(0)
            # ax2.margins(0)
            # ax1.grid(True)
            # #ax2.set_yticks(np.linspace(ax2.get_yticks()[0], ax2.get_yticks()[-1], len(ax1.get_yticks())))
            # start, end = ax1.get_yticks()[[0, -1]]
            # num_at_bottom = int(7 * (1 - start) / (end - start))
            # if num_at_bottom == 0:
            #     ax1.set_yticks(np.linspace(1, end, 7))
            # else:
            #     step_size = (1 - start) / num_at_bottom
            #     ax1.set_yticks([start + step_size * i for i in range(7)])
            # start, end = ax2.get_yticks()[[0, -1]]
            # num_at_bottom = int(7 * (-start) / (end - start))
            # if num_at_bottom == 0:
            #     ax2.set_yticks(np.linspace(0, end, 7))
            # else:
            #     step_size = (-start) / num_at_bottom
            #     ax2.set_yticks([start + step_size * i for i in range(7)])
            # ax2.grid(False)


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
            # if max_value > 10:
            #     # Don't plot diverging runs
            #     continue
            max_y = max(max_y, max_value)
            if name == "Gain Adaptation":
                if cfg['is_q']:
                    plot_name = "PID Q-Learning (Gain Adaptation)"
                else:
                    plot_name = "PID TD (Gain Adaptation)"
            else:
                plot_name = name.replace("Q Learning", "Q-Learning").replace("kp=", "\kappa_p=").replace("ki=", "\kappa_I=").replace("kd=", "\kappa_d=")
            ax0.plot(x_axis(history), normalize(history), label=plot_name)
            ax0.fill_between(x_axis(history), normalize(history) - std_dev / sqrt(repeat), normalize(history) + std_dev / sqrt(repeat), alpha=0.2)

    if cfg['plot_best']:
        logging.info(f"Best final history: {min_history_file}")
        if cfg['is_double_q']:
            name = "Gain Adaptation Double Q-Learning"
        elif cfg['is_q']:
            name = "Gain Adaptation Q-Learning"
        else:
            name = "Gain Adaptation TD"
        if min_history[0] != 0:
            min_std_dev /= min_history[0]
        max_y = max(max_y, np.max(normalize(min_history) + min_std_dev))
        ax0.plot(x_axis(min_history), normalize(min_history), label=r"{}".format(name))
        ax0.fill_between(x_axis(min_history), normalize(min_history) - min_std_dev / sqrt(repeat), normalize(min_history) + min_std_dev / sqrt(repeat), alpha=0.2)

    ax0.set_title(f"{cfg['env'].title()}", fontsize=titlefontsize)
    ax0.set_xlabel('Steps (t)')
    ax0.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))

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