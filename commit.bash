#!/bin/bash

git add Experiments/*.py
git add Experiments/*/*.py
git add Experiments/optimal_learning_rates.pickle
git add makefile
git add commit.bash
git add config
git add slurm/dqn.sh
git add TabularPID/*.py
git add TabularPID/*/*.py
git add TabularPID/*/*/*.py
git add stable_baselines3
git add README.md
git add requirements.txt

git commit
