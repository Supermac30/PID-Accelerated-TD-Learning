#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1
#SBATCH --time=5:00:00
#SBATCH --mem=1GB
#SBATCH --job-name=pid_vi
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/errors/%x_%j.err

source slurm/setup.sh
current_time=$(date "+%Y.%m.%d/%H.%M.%S")
env="cliff walk"
gamma=0.99
seed=$RANDOM
num_iterations=1000
directory=outputs/vi_control_experiment/$env/$current_time
echo "Saving to $directory"
mkdir -p "$directory"

PID-Accelerated-TD-Learning/

python3 -m Experiments.VIExperiments.VIPolicyEvaluation --multirun \
    hydra.mode=MULTIRUN \
    hydra.run.dir="$directory" \
    hydra.sweep.dir="$directory" \
    save_dir="$directory" \
    seed=$seed \
    kp=1 \
    ki=0.2 \
    kd=0 \
    gamma=$gamma \
    env="$env" \
    num_iterations=$num_iterations \
    'name="PID VI with kp=1, ki=0.2, kd=0"'

python3 -m Experiments.VIExperiments.VIPolicyEvaluation --multirun \
    hydra.mode=MULTIRUN \
    hydra.run.dir="$directory" \
    hydra.sweep.dir="$directory" \
    save_dir="$directory" \
    seed=$seed \
    kp=1 \
    ki=0.3 \
    kd=0.2 \
    gamma=$gamma \
    env="$env" \
    num_iterations=$num_iterations \
    'name="PID VI with kp=1, ki=0.3, kd=0.2"'

python3 -m Experiments.VIExperiments.VIPolicyEvaluation --multirun \
    hydra.mode=MULTIRUN \
    hydra.run.dir="$directory" \
    hydra.sweep.dir="$directory" \
    save_dir="$directory" \
    seed=$seed \
    kp=1 \
    ki=0 \
    kd=0 \
    gamma=$gamma \
    env="$env" \
    num_iterations=$num_iterations \
    name="VI"


python3 -m Experiments.Plotting.plot_vi_experiment \
    hydra.run.dir="$directory" \
    save_dir="$directory" \
    repeat=$repeat \
    env="$env" \
    is_q=False \
    log_plot=True