#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1
#SBATCH --time=50:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=pid_dqn
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/errors/%x_%j.err

source slurm/setup.sh

current_time=$(date "+%Y.%m.%d-%H.%M.%S")
env=Acrobot-v1
directory=outputs/dqn_experiment/${env}/$current_time
echo "Saving to ${directory}"
mkdir -p "$directory"

tabular_d=True
is_double=True
num_runs=5

eval=True
visualize=False

seed=$RANDOM

python3 -m Experiments.DQNExperiments.DQNExperiment --multirun \
   env=$env name=$env experiment_name="$env Double DQN tabular d"\
   hydra.mode=MULTIRUN \
   hydra.run.dir=$directory \
   hydra.sweep.dir=$directory \
   save_dir=$directory \
   seed=$seed \
   is_double=$is_double \
   kp=1 \
   kd=0.1,0.2 \
   ki=0 \
   d_tau=1,0.5,0.1 \
   tabular_d=$tabular_d \
   num_runs=$num_runs

python3 -m Experiments.DQNExperiments.DQNExperiment --multirun \
   env=$env name=$env experiment_name="$env "\
   hydra.mode=MULTIRUN \
   hydra.run.dir=$directory \
   hydra.sweep.dir=$directory \
   save_dir=$directory \
   seed=$seed \
   is_double=$is_double \
   kp=1 \
   kd=0 \
   ki=0 \
   num_runs=$num_runs \
   visualize=$visualize \
   eval=$eval \

python3 -m Experiments.Plotting.plot_dqn_experiment \
   hydra.run.dir=$directory \
   save_dir=$directory \
   hydra/job_logging=disabled
