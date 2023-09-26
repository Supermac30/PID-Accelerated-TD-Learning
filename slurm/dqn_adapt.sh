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
env=MountainCar-v0
directory=outputs/dqn_adapt_experiment/${env}/$current_time
echo "Saving to ${directory}"
mkdir -p "$directory"

gain_adapter=DiagonalGainAdapter  # Options: NoGainAdapter, SingleGainAdapter, DiagonalGainAdapter, NetworkGainAdapter
tabular_d=False
use_previous_BRs=True
is_double=True

num_runs=1

policy_evaluation=False
eval=True

seed=$RANDOM
experiment_name="$env new normalization test"

python3 -m Experiments.DQNExperiments.DQNExperiment --multirun \
   env="$env" name="$env" experiment_name="$experiment_name" \
   hydra.mode=MULTIRUN \
   hydra.run.dir=$directory \
   hydra.sweep.dir=$directory \
   save_dir=$directory \
   seed=$seed \
   run_name=$seed \
   gain_adapter=$gain_adapter \
   adapt_gains=True \
   is_double=$is_double \
   use_previous_BRs=$use_previous_BRs \
   d_tau=0.01 \
   epsilon=1 \
   meta_lr_p=0.01 \
   meta_lr_I=0.01 \
   meta_lr_d=0.01 \
   tabular_d=$tabular_d \
   num_runs=$num_runs \
   policy_evaluation=$policy_evaluation \
   eval=$eval

python3 -m Experiments.DQNExperiments.DQNExperiment \
   env="$env" name="$env" experiment_name="$experiment_name" \
   hydra.mode=MULTIRUN \
   hydra.run.dir=$directory \
   hydra.sweep.dir=$directory \
   save_dir=$directory \
   is_double=$is_double \
   seed=$seed \
   run_name=$seed \
   adapt_gains=False \
   num_runs=$num_runs \
   policy_evaluation=$policy_evaluation \
   eval=$eval

python3 -m Experiments.Plotting.plot_dqn_experiment \
   hydra.run.dir=$directory \
   save_dir=$directory \
   hydra/job_logging=disabled
