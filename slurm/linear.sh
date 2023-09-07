#!/bin/bash
#SBATCH -N 1
#SBATCH -p cpu
#SBATCH --cpus-per-task=64
#SBATCH --tasks-per-node=1
#SBATCH --time=30:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=linear_PID
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/errors/%x_%j.err

source slurm/setup.sh

current_time=$(date "+%Y.%m.%d/%H.%M.%S")
env="CartPole-v1"
gamma=0.99
repeat=20
order=3
type="fourier"  # "trivial", "fourier", "polynomial", "tile coding"
is_q=True
seed=$RANDOM
num_iterations=100
search_steps=100
separation=$((num_iterations/100))
directory=outputs/linear_experiment/$env/$current_time
echo "Saving to $directory"
mkdir -p "$directory"

recompute_optimal=False
compute_optimal=False
get_optimal=True

python3 -m Experiments.LinearFAExperiments.LinearFAExperiment --multirun \
    hydra.mode=MULTIRUN \
    hydra.run.dir="$directory" \
    hydra.sweep.dir="$directory" \
    save_dir="$directory" \
    seed=$seed \
    kp=1 \
    ki=0,-0.1,0.1 \
    kd=0,-0.1,0.1 \
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