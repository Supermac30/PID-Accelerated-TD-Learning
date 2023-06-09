"""ChatGPT output: run in a directory to create a plot for each npy file."""
import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import os

# create the argument parser
parser = argparse.ArgumentParser(description='Plot data from .npy files.')
parser.add_argument('ylabel', type=str, help='the label for the y-axis')

# parse the command-line arguments
args = parser.parse_args()

# find all .npy files in the current directory
files = glob.glob('*.npy')

# plot each file with its filename as the label
for file in files:
    data = np.load(file)
    label = os.path.splitext(os.path.basename(file))[0] # use the filename as the label
    plt.plot(data, label=label)

# set labels and save/show the plot
plt.legend()
plt.xlabel('Iteration')
plt.ylabel(args.ylabel)
plt.savefig("plot")
plt.show()
