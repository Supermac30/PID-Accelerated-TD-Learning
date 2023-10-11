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

    # Iterate through all of the files in the npy folder
    for file in os.listdir(f"{cfg['save_dir']}/npy"):
        name = file[:-4]
        history = np.load(f"{cfg['save_dir']}/npy/{file}")
        
        # Plot history
        ax0.plot(normalize(history), label=name)

    # Make the y label steps
    ax0.set_ylabel("Steps")

    if cfg['log_plot']:
        ax0.set_yscale('log')

    # Set the x label to be ||V - V^*||
    create_label(ax0, cfg['norm'], cfg['normalize'], cfg['is_q'])

    # Set the title
    ax0.set_title(f"{cfg['env'].title()}")

    # Add grid lines
    ax0.grid()

    # Set the legend
    ax0.legend()

    fig0.savefig(f"{cfg['save_dir']}/adaptive_agent_{cfg['env']}.pdf")
    fig0.savefig(f"{cfg['save_dir']}/adaptive_agent_{cfg['env']}.png")


if __name__ == "__main__":
    create_plots()