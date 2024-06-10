#!/bin/bash
#SBATCH -N 1
#SBATCH -p cpu
#SBATCH --cpus-per-task=64
#SBATCH --tasks-per-node=1
#SBATCH --time=30:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=VI_GA
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/errors/%x_%j.err

source slurm/setup.sh
current_time=$(date "+%Y.%m.%d/%H.%M.%S")
env="cliff walk"
gamma=0.999
seed=$RANDOM
num_iterations=1000
directory=outputs/vi_control_adapt_experiment/$env/$current_time
echo "Saving to $directory"
mkdir -p "$directory"

python3 -m Experiments.AdaptationExperiments.AdaptiveAgentPAVIAControlExperiment --multirun \
    hydra.mode=MULTIRUN \
    hydra.run.dir="$directory" \
    hydra.sweep.dir="$directory" \
    save_dir="$directory" \
    seed=$seed \
    gamma=$gamma \
    env="$env" \
    meta_lr=1e-3 \
    num_iterations=$num_iterations \
    agent_name="Q planner" \
    'name="Gain Adaptation"'

python3 -m Experiments.VIExperiments.VIQControl --multirun \
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
    is_q=True \
    log_plot=True