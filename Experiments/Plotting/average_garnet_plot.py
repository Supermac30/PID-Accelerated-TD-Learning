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

    # Plot the average history
    if cfg['is_q']:
        ax0.plot(normalize(average_history), label="Q-learning")
    else:
        ax0.plot(normalize(average_history), label="TD")
    
    ax0.fill_between(np.arange(len(average_history)), normalize(average_history - standard_deviation), normalize(average_history + standard_deviation), alpha=0.2)

    all_histories = []
    for run in range(1, cfg['repeat'] + 1):
        min_final_history = np.inf
        final_history = None
        for file in os.listdir(f"{cfg['save_dir']}/gain_adaptation/{run}/npy/mean"):
            if file.startswith("gain_history"):
                continue

            history = np.load(f"{cfg['save_dir']}/gain_adaptation/{run}/npy/mean/{file}")
            if history[-1] < min_final_history:
                min_final_history = history[-1]
                final_history = history
        
        all_histories.append(final_history)

    # Average all of the histories together
    average_history = np.mean(all_histories, axis=0)
    standard_deviation = np.std(all_histories, axis=0)

    # Plot the average history
    if cfg['is_q']:
        ax0.plot(normalize(average_history), label="PID Q-learning + Gain Adaptation")
    else:
        ax0.plot(normalize(average_history), label="PID TD + Gain Adaptation")
    
    ax0.fill_between(np.arange(len(average_history)), normalize(average_history - standard_deviation), normalize(average_history + standard_deviation), alpha=0.2)

    ax0.title.set_text("Garnet")
    ax0.legend()
    # Add grid lines
    ax0.grid()
    ax0.set_xlabel('Steps')
    create_label(ax0, cfg['norm'], cfg['normalize'], cfg['is_q'])

    fig0.savefig(f"{cfg['save_dir']}/adaptive_agent_{cfg['env']}.pdf")
    fig0.savefig(f"{cfg['save_dir']}/adaptive_agent_{cfg['env']}.png")


if __name__ == "__main__":
    create_plots()