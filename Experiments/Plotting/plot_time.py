import os
import numpy as np
import matplotlib.pyplot as plt
import hydra
from matplotlib.ticker import FuncFormatter
from math import sqrt
import pickle

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

    time_fig = plt.figure()
    time_ax = time_fig.add_subplot()

    files = os.listdir(f"{cfg['save_dir']}/time")[::-1]
    files = sorting_mechanism(files)
    # Iterate through all of the files in the npy folder
    for file in files:
        name = file[:-4]
        time_data = pickle.load(f"{cfg['save_dir']}/time/{name}.pkl")
        states = time_data.keys()
        mean_time = [time_data[state][0] for state in states]
        std_dev_time = [time_data[state][1] for state in states]

        time_ax.plot(states, mean_time, label=name)
        time_ax.fill_between(states, mean_time - std_dev_time / sqrt(80), mean_time + std_dev_time / sqrt(80), alpha=0.2)

    time_ax.set_title("Garnets", fontsize=titlefontsize)
    time_ax.set_xlabel("Number of states")
    time_ax.set_ylabel("Wall Clock (ms)")
    time_ax.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))

    # ax0.set_ylim(0, min(2, max_y))
    time_ax.legend()

    if cfg['log_plot']:
        time_ax.set_yscale('log')
    
    # Add grid lines
    time_ax.grid()

    time_fig.savefig(f"{cfg['save_dir']}/time_figure.pdf")
    time_fig.savefig(f"{cfg['save_dir']}/time_figure.png")

if __name__ == "__main__":
    create_plots()