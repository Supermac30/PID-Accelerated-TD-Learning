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

    runs = cfg['repeat'] ** 0.5

    # Iterate through all of the files in the npy folder
    for file in os.listdir(f"{cfg['save_dir']}/npy/mean"):
        name = file[:-4]
        history = np.load(f"{cfg['save_dir']}/npy/mean/{file}")
        std_dev = np.load(f"{cfg['save_dir']}/npy/std_dev/{file}")
        # If the file starts with gain_history, plot it:
        if file.startswith("gain_history"):
            fig = plt.figure(figsize=(10, 4))
            gs = fig.add_gridspec(nrows=1, ncols=3, width_ratios=[1,1,1], wspace=0.3, hspace=0.5)
            titles = ["kp", "ki", "kd"]
            for i in range(3):
                ax = fig.add_subplot(gs[0, i])
                ax.plot(history[:, i])
                ax.fill_between(np.arange(len(history[:, i])), history[:, i] - (std_dev[:, i] / runs), history[:, i] + (std_dev[:, i] / runs), alpha=0.2)
                ax.set_xlabel('Steps')
                ax.title.set_text(titles[i])

            # Set square aspect ratio for each subplot
            for ax in fig.axes:
                ax.set_box_aspect(1)
                
            # Center the subplots in a single row and move up to remove whitespace
            gs.tight_layout(fig, rect=[0.05, 0.08, 0.95, 0.95])

            fig.savefig(f"{cfg['save_dir']}/{name}.png")
            fig.savefig(f"{cfg['save_dir']}/{name}.pdf")
            plt.close(fig)

        # Otherwise, plot the file on the history plot
        elif file.startswith("Adaptive Agent"):
            if cfg['plot_best']:
                if history[-1] < min_final_history:
                    min_final_history = history[-1]
                    min_history_file = file
                    min_history = history
                    min_std_dev = std_dev
            else:
                std_dev /= history[0]
                ax0.plot(normalize(history), label=name)
                ax0.fill_between(np.arange(len(history)), normalize(history) - (std_dev / runs), normalize(history) + (std_dev / runs), alpha=0.2)
        
        else:
            std_dev /= history[0]
            ax0.plot(normalize(history), label=f"{cfg['name']}")
            ax0.fill_between(np.arange(len(history)), normalize(history) - (std_dev / runs), normalize(history) + (std_dev / runs), alpha=0.2)

    if cfg['plot_best']:
        logging.info(f"Best final history: {min_history_file}")
        min_std_dev /= min_history[0]
        ax0.plot(normalize(min_history), label=f"PID {cfg['name']} + Gain Adaptation")
        ax0.fill_between(np.arange(len(min_history)), normalize(min_history) - (min_std_dev / runs), normalize(min_history) + (min_std_dev / runs), alpha=0.2)

    ax0.title.set_text(f"{cfg['env'].title()}")
    ax0.legend()
    ax0.set_xlabel('Steps')
    create_label(ax0, cfg['norm'], cfg['normalize'], False)
    if cfg['log_plot']:
        ax0.set_yscale('log')

    fig0.savefig(f"{cfg['save_dir']}/adaptive_agent_{cfg['env']}.pdf")
    fig0.savefig(f"{cfg['save_dir']}/adaptive_agent_{cfg['env']}.png")


if __name__ == "__main__":
    create_plots()