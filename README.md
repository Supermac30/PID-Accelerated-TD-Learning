# PID-Accelerated-TD-Learning
This work is presented in https://arxiv.org/abs/2407.08803

This is an application of ideas from control theory to accelerate the dynamics of TD learning, building on the work of Farahmand and Ghavamzadeh [1] in an RL setting.

[1] A.M. Farahmand and Mohammad Ghavamzadeh, “PID Accelerated Value Iteration Algorithm,” International Conference on Machine Learning (ICML), 2021.

## Reproducibility Instructions:
0) Create a directory called outputs in the top level if one doesn't already exist.
1) Change the base directory in the slurm/setup.sh and globals.py files.
2) Change the learning rates to grid search through at the top of TabularPID/hyperparameter_tests.py if needed.
3) Each slurm/*.sh file is an experiment. Change the parameters at the top as desired. See the list below for some documentation on their meaning.
4) Just run the file and the output should be created in the outputs directory.

Important parameters:
- Repeat: The number of times an experiment is repeated on different seeds when calculating the results.

To run aggregated garnet results quicker, use run_aggregated_garnets.sh to split up the work across different nodes. Then, call slurm/plot_garnets.sh to plot the results.
