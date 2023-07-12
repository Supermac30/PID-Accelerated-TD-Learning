import os
import numpy as np
import matplotlib.pyplot as plt
import hydra

from Experiments.ExperimentHelpers import *

@hydra.main(config_path="../../config/AdaptationExperiments", config_name="AdaptiveAgentExperiment")
def create_plots(cfg):
    """Plot all of the data in the npy folder from the runs of the adaptive agent."""
    # Create a figure that will be used to plot the history of each agent
    fig0 = plt.figure()
    ax0 = fig0.add_subplot()

    # Iterate through all of the files in the npy folder
    for file in os.listdir("npy"):
        # If the file starts with gain_history, plot it:
        if file.startswith("gain_history"):
            gain_history = np.load(f"npy/{file}")
            fig = plt.figure(figsize=(10, 4))
            gs = fig.add_gridspec(nrows=1, ncols=3, width_ratios=[1,1,1], wspace=0.3, hspace=0.5)
            titles = ["kp", "ki", "kd"]
            for i in range(3):
                ax = fig.add_subplot(gs[0, i])
                ax.plot(gain_history[:, i])
                ax.set_xlabel('Iteration')
                ax.set_ylabel(titles[i])
                ax.legend()

            plt.suptitle(file)
            
            # Set square aspect ratio for each subplot
            for ax in fig.axes:
                ax.set_box_aspect(1)
                
            # Center the subplots in a single row and move up to remove whitespace
            gs.tight_layout(fig, rect=[0.05, 0.08, 0.95, 0.95])

            plt.savefig(f"plots/{file}.png")
            plt.close()

        # Otherwise, plot the file on the history plot
        else:
            history = np.load(f"npy/{file}")
            ax0.plot(history, label=file)

    ax0.title.set_text(f"Adaptive Agent: {cfg['env']}")
    ax0.legend()
    ax0.set_xlabel('Iteration')
    create_label(ax0, cfg['norm'], cfg['normalize'], False)
    if cfg['log_plot']:
        ax0.set_yscale('log')