import os
import numpy as np
import matplotlib.pyplot as plt
import hydra
from matplotlib.ticker import FuncFormatter

from Experiments.ExperimentHelpers import *

# Made by ChatGPT: Function to format the tick labels
def thousands_formatter(x, pos):
    return f'{int(x/1000)}k' if x >= 1000 else int(x)

@hydra.main(version_base=None, config_path="../../config", config_name="plot")
def create_plots(cfg):
    """Plot all of the data in the npy folder from the runs of the adaptive agent."""
    # Create a figure that will be used to plot the history of each agent
    titlefontsize = 13
    plt.rc('font', size=12)          # controls default text sizes
    plt.rc('axes', titlesize=12)     # fontsize of the axes title
    plt.rc('axes', labelsize=12)     # fontsize of the x and y labels
    plt.rc('xtick', labelsize=11)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=11)    # fontsize of the tick labels

    fig0 = plt.figure()
    ax0 = fig0.add_subplot()

    # Iterate through all of the files in the npy folder. We want to average all of the history files together,
    # and plot an average history plot and an error curve
    all_histories = []
    if cfg['is_q']:
        file_loc = "Q"
    else:
        file_loc = "TD"
    
    assert(cfg['repeat'] > 0)
    for run in range(1, cfg['repeat'] + 1):
        # If the file starts with gain_history, plot it:
        # Take the file f"{cfg['save_dir']}/{file_loc}/{run}/npy/" and plot it on the history plot
        # I don't know what the file name is, so plot the only file in the folder
        file_folder = f"{cfg['save_dir']}/{file_loc}/{run}/npy/mean"
        if os.path.exists(file_folder):
            files = os.listdir(file_folder)
            if len(files) > 0:
                file = files[0]
                # Load f"{cfg['save_dir']}/TD/{run}/npy/{file}" and plot it on the history plot
                all_histories.append(np.load(f"{file_folder}/{file}"))
    
    # Average all of the histories together
    average_history = np.mean(all_histories, axis=0)
    standard_deviation = np.std(all_histories, axis=0)

    all_histories = []
    for run in range(1, cfg['repeat'] + 1):
        min_final_history = np.inf
        final_history = None
        file_folder = f"{cfg['save_dir']}/gain_adaptation/{run}/npy/mean"
        if os.path.exists(file_folder):
            for file in os.listdir(file_folder):
                if file.startswith("gain_history"):
                    continue

                history = np.load(f"{file_folder}/{file}")
                if history[-1] < min_final_history:
                    min_final_history = history[-1]
                    final_history = history
        else:
            raise Exception(f"{run}")
        
        all_histories.append(final_history)

    # Average all of the histories together
    new_average_history = np.mean(all_histories, axis=0)
    new_standard_deviation = np.std(all_histories, axis=0)

    # Plot the average history
    if cfg['is_q']:
        ax0.plot(normalize(new_average_history), label="PID Q-Learning (Gain Adaptation)")
    else:
        ax0.plot(normalize(new_average_history), label="PID TD (Gain Adaptation)")
    
    ax0.fill_between(np.arange(len(new_average_history)), normalize(new_average_history - new_standard_deviation), normalize(new_average_history + new_standard_deviation), alpha=0.2)

    # Plot the average history
    if cfg['is_q']:
        ax0.plot(normalize(average_history), label="Q-Learning")
    else:
        ax0.plot(normalize(average_history), label="TD")
    
    ax0.fill_between(np.arange(len(average_history)), normalize(average_history - standard_deviation), normalize(average_history + standard_deviation), alpha=0.2)

    ax0.set_title("Garnet", fontsize=titlefontsize)
    ax0.legend()
    # Add grid lines
    ax0.grid()
    ax0.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    ax0.set_xlabel('Steps (t)')
    create_label(ax0, cfg['norm'], cfg['normalize'], cfg['is_q'])

    fig0.savefig(f"{cfg['save_dir']}/adaptive_agent_{cfg['env']}.pdf")
    fig0.savefig(f"{cfg['save_dir']}/adaptive_agent_{cfg['env']}.png")


if __name__ == "__main__":
    create_plots()