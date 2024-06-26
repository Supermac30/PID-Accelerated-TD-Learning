#!/bin/bash
#SBATCH -N 1
#SBATCH -p cpu
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --mem=1GB
#SBATCH --job-name=garnet_plot
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/errors/%x_%j.err

source slurm/setup.sh

python3 -m Experiments.Plotting.average_garnet_plot \
    hydra.run.dir="$1" \
    save_dir="$1" \
    repeat="$2" \
    is_q="$3" \
    norm="$4" \
    hydra/job_logging=disabled