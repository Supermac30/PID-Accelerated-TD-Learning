"""
Plot all npy files in the directory, with the label being the file name. Save the file
"""

import os
import numpy as np

import matplotlib.pyplot as plt

def plot_npy_files_in_dir(dir_path):
    """
    Plot all npy files in the directory, with the label being the file name
    """
    files = get_all_files_in_dir(dir_path)
    for file in files:
        if file.endswith(".npy"):
            data = np.load(os.path.join(dir_path, file))
            plt.plot(data, label=file)

    plt.ylabel("Reward")
    plt.xlabel("Episode")
    plt.legend()
    plt.show()
    print("Saving the file")

    # Save the file
    plt.savefig(os.path.join(dir_path, "plot.png"))
    print("File directory: ", os.path.join(dir_path, "plot.png"))

def get_all_files_in_dir(dir_path):
    """
    Get all files in the directory
    """
    return os.listdir(dir_path)

plot_npy_files_in_dir("outputs/DQNExperiment/2023-06-07/15-35-23/npy")

# Command to run this file from the root directory: python -m outputs.DQNExperiment.2023-06-07.15-35-23.npy.plot