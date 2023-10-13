#!/bin/bash
#SBATCH -N 1
#SBATCH -p cpu
#SBATCH --cpus-per-task=64
#SBATCH --tasks-per-node=1
#SBATCH --time=0:05:00
#SBATCH --mem=1GB
#SBATCH --job-name=visualize_adaptation
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/errors/%x_%j.err

ulimit -n 2048

source slurm/setup.sh

current_time=$(date "+%Y.%m.%d/%H.%M.%S")
env="baby chain walk"
gamma=0.1
seed=$RANDOM
num_iterations=10000
search_steps=10000
recompute_optimal=True
compute_optimal=True  # False when we need to debug, so there is no multiprocessing
get_optimal=True  # False when we need to debug with a specific learning rate

directory=outputs/visualize_adaptation/$env/$current_time
echo "Saving to ${directory}"
mkdir -p "$directory"

python3 -m Experiments.AdaptationExperiments.TrajectoryVisualization --multirun \
    hydra.mode=MULTIRUN \
    hydra.run.dir="$directory/Adaptive Agent" \
    hydra.sweep.dir="$directory" \
    seed=$seed \
    save_dir="$directory" \
    search_steps=$search_steps \
    recompute_optimal=$recompute_optimal \
    compute_optimal=$compute_optimal \
    get_optimal=$get_optimal \
    meta_lr=0.01 \
    epsilon=0.01 \
    env="$env" \
    gamma=$gamma \
    num_iterations=$num_iterations \
    agent_name="semi gradient updater" \
    +jump=1000