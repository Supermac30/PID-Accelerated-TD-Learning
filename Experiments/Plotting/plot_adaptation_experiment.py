import os
import numpy as np
import matplotlib.pyplot as plt
import hydra

from Experiments.ExperimentHelpers import *

@hydra.main(version_base=None, config_path="../../config", config_name="plot")
def create_plots(cfg):
    """Plot all of the data in the npy folder from the runs of the adaptive agent."""
    # Create a figure that will be used to plot the history of each agent
    fig0, ax0 = plt.subplots(figsize=(11, 7))

    # Min history
    min_final_history = np.inf
    min_history = None
    min_history_file = None
    min_std_dev = None

    max_y = 1
    sep = cfg['separation']
    x_axis = lambda n: np.arange(0, sep * len(n), sep)

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
                ax.plot(x_axis(history[i]), history[i])
                ax.fill_between(x_axis(history[i]), history[i] - std_dev[i], history[i] + std_dev[i], alpha=0.2)
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
                if history[0] != 0:
                    std_dev /= history[0]
                max_y = max(max_y, np.max(normalize(history) + std_dev))
                ax0.plot(x_axis(history), normalize(history), label=name)
                ax0.fill_between(x_axis(history), normalize(history) - std_dev, normalize(history) + std_dev, alpha=0.2)
        else:
            if file.startswith("TIDBD"):
                name = "TIDBD"
            elif file.startswith("TD"):
                name = "TD"
            elif file.startswith("speedy Q learning"):
                name = "Speedy Q Learning"
            elif file.startswith("zap Q learning"):
                name = "Zap Q Learning"
            elif file.startswith("Q learning"):
                name = "Q Learning"
            else:
                name = cfg['default_name']

            name = file[:-4]
            if history[0] != 0:
                std_dev /= history[0]
            max_y = max(max_y, np.max(normalize(history) + std_dev))
            ax0.plot(x_axis(history), normalize(history), label=name)
            ax0.fill_between(x_axis(history), normalize(history) - std_dev, normalize(history) + std_dev, alpha=0.2)

    if cfg['plot_best']:
        logging.info(f"Best final history: {min_history_file}")
        if cfg['is_double_q']:
            name = "Double Q Learning"
        elif cfg['is_q']:
            name = "Q Learning"
        else:
            name = "TD"
        if min_history[0] != 0:
            min_std_dev /= min_history[0]
        max_y = max(max_y, np.max(normalize(min_history) + min_std_dev))
        ax0.plot(x_axis(history), normalize(history), label=name)
        ax0.fill_between(x_axis(history), normalize(history) - std_dev, normalize(history) + std_dev, alpha=0.2)

    ax0.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=2)
    ax0.title.set_text(f"{cfg['env'].title()}")
    ax0.set_xlabel('Steps')
    ax0.set_ylim(0, min(2, max_y))
    # Place the legend outside the graph
    create_label(ax0, cfg['norm'], cfg['normalize'], cfg['is_q'])

    if cfg['log_plot']:
        ax0.set_yscale('log')

    # Force everything to fit
    fig0.tight_layout()

    fig0.savefig(f"{cfg['save_dir']}/adaptive_agent_{cfg['env']}.pdf")
    fig0.savefig(f"{cfg['save_dir']}/adaptive_agent_{cfg['env']}.png")


if __name__ == "__main__":
    create_plots()