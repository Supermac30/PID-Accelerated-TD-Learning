#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1
#SBATCH --time=30:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=dqn_gain_sweep

source ~/.bashrc
source ~/newgym.nv
conda activate myenv

cd /h/bedaywim/PID-Accelerated-TD-Learning

make DQN_mountaincar