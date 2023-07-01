#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1
#SBATCH --time=30:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=pid_dqn
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/errors/%x_%j.err

source ~/.bashrc
source ~/newgym.nv
conda activate myenv

cd /h/bedaywim/PID-Accelerated-TD-Learning

# python3 -m Experiments.DQNExperiments.DQNExperiment \
#    env=cartpole name=cartpole \
#    hydra.mode=MULTIRUN \
#    kd=0,0.1,0.2,0.3,0.4,0.5 \
#    d_tau=0.9 \
#    tabular_d=True,False

# python3 -m Experiments.DQNExperiments.DQNExperiment \
#    env=mountaincar name=mountaincar \
#    hydra.mode=MULTIRUN \
#    adapt_gains=True,False \
#    meta_lr=0.01 \
#    epsilon=0.25 \
#    d_tau=0.001

python3 -m Experiments.DQNExperiments.DQNExperiment \
   env=mountaincar name=mountaincar \
   hydra.mode=MULTIRUN \
   adapt_gains=True \
   meta_lr=1e-2,1e-3 \
   epsilon=5,2,1,0.25 \
   d_tau=1e-1,1e-2,1e-3

# python3 -m Experiments.DQNExperiments.DQNExperiment \
#    env=acrobot name=acrobot \
#    hydra.mode=MULTIRUN \
#    kp=1,1.1,1.2 \
#    kd=0,0.1,0.2 \
#    ki=-0.1,0,0.1 \
#    d_tau=1,0.5,0.1
