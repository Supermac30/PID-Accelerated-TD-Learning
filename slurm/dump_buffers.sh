#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1
#SBATCH --time=5:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=dump_buffers
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/errors/%x_%j.err

source slurm/setup.sh

for env in "Cartpole-v1" "MountainCar-v0"  # "LunarLander-v2"  "Acrobot-v1"
do
   current_time=$(date "+%Y.%m.%d-%H.%M.%S")
   directory=outputs/build_true_Q_nets/$env/$current_time
   echo "Saving to $directory"
   mkdir -p "$directory"

   python3 -m Experiments.DQNExperiments.DQNExperiment \
      env=$env name=$env experiment_name="$env Dump Buffers"\
      hydra.mode=MULTIRUN \
      hydra.run.dir=$directory \
      hydra.sweep.dir=$directory \
      save_dir=$directory \
      seed=$RANDOM \
      kp=1 \
      kd=0 \
      ki=0 \
      d_tau=1 \
      dump_buffer=True \
      eval=False \
      num_runs=1
done