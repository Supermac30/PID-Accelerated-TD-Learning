#!/bin/bash
#SBATCH -N 1
#SBATCH -p cpu
#SBATCH --cpus-per-task=64
#SBATCH --tasks-per-node=1
#SBATCH --time=00:01:00
#SBATCH --mem=8GB
#SBATCH --job-name=PLOT
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/errors/%x_%j.err

source slurm/setup.sh

directory="outputs/q_adaptation_experiment/chain walk/Final Result 0.999"
is_q=True
repeat=20
norm="fro"
env="chain walk"

python3 -m Experiments.Plotting.plot_adaptation_experiment \
    hydra.run.dir="$directory" \
    save_dir="$directory" \
    repeat=$repeat \
    env="$env" \
    is_q=$is_q \
    norm=$norm \
    plot_best=False \
    small_name=True \
    is_star=$is_q