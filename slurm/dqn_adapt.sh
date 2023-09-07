#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1
#SBATCH --time=50:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=adapt_dqn
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/errors/%x_%j.err

source slurm/setup.sh

current_time=$(date "+%Y.%m.%d-%H.%M.%S")
env=Acrobot-v1
directory=outputs/dqn_adapt_experiment/${env}/$current_time
echo "Saving to ${directory}"
mkdir -p "$directory"

gain_adapter=SingleGainAdapter  # Options: NoGainAdapter, SingleGainAdapter, DiagonalGainAdapter, NetworkGainAdapter
tabular_d=True
use_previous_BRs=True
is_double=True

num_runs=5

seed=$RANDOM

python3 -m Experiments.DQNExperiments.DQNExperiment --multirun \
   env=$env name=$env experiment_name="$env Diagonal Adaptation Experiment"\
   hydra.mode=MULTIRUN \
   hydra.run.dir=$directory \
   hydra.sweep.dir=$directory \
   save_dir=$directory \
   seed=$seed \
   gain_adapter=$gain_adapter \
   adapt_gains=True \
   is_double=$is_double \
   use_previous_BRs=$use_previous_BRs \
   d_tau=1,0.5,0.1 \
   epsilon=1 \
   meta_lr=0.5,0.1,0.01 \
   tabular_d=$tabular_d \
   num_runs=$num_runs

python3 -m Experiments.DQNExperiments.DQNExperiment \
   env=$env name=$env experiment_name="$env Adaptation Experiment"\
   hydra.mode=MULTIRUN \
   hydra.run.dir=$directory \
   hydra.sweep.dir=$directory \
   save_dir=$directory \
   is_double=$is_double \
   seed=$seed \
   gain_adapter=$gain_adapter \
   adapt_gains=False \
   use_previous_BRs=$use_previous_BRs \
   num_runs=$num_runs

python3 -m Experiments.Plotting.plot_dqn_experiment \
   hydra.run.dir=$directory \
   save_dir=$directory \
   hydra/job_logging=disabled