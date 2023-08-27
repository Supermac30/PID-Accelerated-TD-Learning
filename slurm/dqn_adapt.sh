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

source slurm/setup.sh

current_time=$(date "+%Y.%m.%d-%H.%M.%S")
env=Cartpole-v1
directory=outputs/dqn_experiment/${env}/$current_time
echo "Saving to ${directory}"
mkdir -p "$directory"

gain_adapter=SingleGainAdapter  # Options: NoGainAdapter, SingleGainAdapter, DiagonalGainAdapter, NetworkGainAdapter
use_previous_BRs=True


python3 -m Experiments.DQNExperiments.DQNExperiment \
   env=$env name=$env experiment_name="$env PBR Gain Sweep"\
   hydra.mode=MULTIRUN \
   hydra.run.dir=$directory \
   hydra.sweep.dir=$directory \
   save_dir=$directory \
   seed=$RANDOM \
   gain_adapter=$gain_adapter \
   adapt_gains=True \
   use_previous_BRs=$use_previous_BRs \
   d_tau=1,0.5,0.1 \
   epsilon=0,0.1,1 \
   meta_lr=0.5,0.1,0.01

python3 -m Experiments.Plotting.plot_dqn_experiment \
   hydra.run.dir=$directory \
   save_dir=$directory \
   hydra/job_logging=disabled