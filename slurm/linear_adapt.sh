#!/bin/bash
#SBATCH -N 1
#SBATCH -p cpu
#SBATCH --cpus-per-task=64
#SBATCH --tasks-per-node=1
#SBATCH --time=30:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=linear_adapt_PID
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/errors/%x_%j.err

source slurm/setup.sh

current_time=$(date "+%Y.%m.%d/%H.%M.%S")
env="chain walk"
gamma=0.99
repeat=20
is_q=False
order=20
type="fourier"  # "trivial", "fourier", "polynomial", "tile coding"
seed=$RANDOM
num_iterations=10000
search_steps=10000
separation=$((num_iterations/100))
directory=outputs/linear_adapt_experiment/$env/$current_time
echo "Saving to $directory"
mkdir -p "$directory"

recompute_optimal=True
compute_optimal=True
get_optimal=True

python3 -m Experiments.LinearFAExperiments.AdaptiveLinearFAExperiment --multirun \
    hydra.mode=MULTIRUN \
    hydra.run.dir="$directory" \
    hydra.sweep.dir="$directory" \
    save_dir="$directory" \
    seed=$seed \
    order=$order \
    gamma=$gamma \
    env="$env" \
    repeat=$repeat \
    num_iterations=$num_iterations \
    type="$type" \
    compute_optimal=$compute_optimal \
    get_optimal=$get_optimal \
    recompute_optimal=$recompute_optimal \
    search_steps=$search_steps \
    is_q=$is_q \
    epsilon=1e-2,2e-2,3e-2 \
    meta_lr=1e-2,2e-2,3e-2 \

python3 -m Experiments.LinearFAExperiments.LinearFAExperiment --multirun \
    hydra.mode=MULTIRUN \
    hydra.run.dir="$directory" \
    hydra.sweep.dir="$directory" \
    save_dir="$directory" \
    seed=$seed \
    kp=1 \
    ki=0 \
    kd=0 \
    order=$order \
    gamma=$gamma \
    env="$env" \
    repeat=$repeat \
    num_iterations=$num_iterations \
    type="$type" \
    compute_optimal=$compute_optimal \
    get_optimal=$get_optimal \
    recompute_optimal=$recompute_optimal \
    search_steps=$search_steps \
    is_q=$is_q

python3 -m Experiments.Plotting.plot_adaptation_experiment \
    hydra.run.dir="$directory" \
    save_dir="$directory" \
    repeat=$repeat \
    env="$env" \
    is_q=False \
    plot_best=False \
    separation=$separation