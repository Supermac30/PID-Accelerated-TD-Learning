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

current_time=$(date "+%Y.%m.%d/%H.%M.%S")
save_dir=outputs/averaged_garnet_results

python3 -m Experiments.Plotting.average_garnet_plot \
    hydra.run.dir="$1" \
    save_dir="$1" \
    repeat=100 \
    hydra/job_logging=disabled