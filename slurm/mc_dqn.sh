#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1
#SBATCH --time=30:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=mc_dqn
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/errors/%x_%j.err

source slurm/setup.sh

current_time=$(date "+%Y.%m.%d-%H.%M.%S")
env=Cartpole-v1
directory=models/$env
echo "Saving to ${directory}"
mkdir -p "$directory"

python3 -m Experiments.DQNEx


python3 -m Experiments.DQNExperiments.BuildMCNetwork \
   env=cartpole name=cartpole experiment_name="Build Cartpole MC Network"\
   hydra.mode=MULTIRUN \
   hydra.run.dir=$directory \
   hydra.sweep.dir=$directory \
   save_dir=$directory \
   seed=$RANDOM

python3 -m Experiments.Plotting.plot_dqn_experiment \
   hydra.run.dir=$directory \
   save_dir=$directory \
   hydra/job_logging=disabled