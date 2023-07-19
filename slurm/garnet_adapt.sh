#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --tasks-per-node=1
#SBATCH --time=30:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=garnet_adapt
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/errors/%x_%j.err

source ~/.bashrc
source ~/newgym.nv
conda activate myenv

# CHANGE THIS TO YOUR OWN PATH
cd /h/bedaywim/PID-Accelerated-TD-Learning

current_time=$(date "+%Y.%m.%d/%H.%M.%S")
save_dir=$1
directory=$save_dir/${current_time}
echo "Saving to ${directory}"
mkdir -p "$directory"

for run in {$2..$3}
do
    mkdir -p "$save_dir/gain_adaptation/$run"
    mkdir -p "$save_dir/TD/$run"

	seed=$RANDOM
	garnet_seed=$RANDOM

    python3 -m Experiments.AdaptationExperiments.AdaptiveAgentExperiment \
        hydra.run.dir="${directory}/Adaptive Agent" \
        hydra.sweep.dir="$directory" \
        seed=$seed \
        save_dir="$save_dir/gain_adaptation/$run" \
        meta_lr=1e-1 \
        epsilon=1e-2 \
        env="garnet $garnet_seed 50" \
        gamma=0.999 \
        num_iterations=100000

    python3 -m Experiments.AdaptationExperiments.AdaptiveAgentExperiment \
        hydra.run.dir="${directory}/Adaptive Agent" \
        hydra.sweep.dir="$directory" \
        seed=$seed \
        save_dir="$save_dir/gain_adaptation/$run" \
        meta_lr=1e-2 \
        epsilon=1e-2 \
        env="garnet $garnet_seed 50" \
        gamma=0.999 \
        num_iterations=100000

    python3 -m Experiments.AdaptationExperiments.AdaptiveAgentExperiment \
        hydra.run.dir="${directory}/Adaptive Agent" \
        hydra.sweep.dir="$directory" \
        seed=$seed \
        save_dir="$save_dir/gain_adaptation/$run" \
        meta_lr=1e-3 \
        epsilon=1e-2 \
        env="garnet $garnet_seed 50" \
        gamma=0.999 \
        num_iterations=100000

	python3 -m Experiments.TDExperiments.SoftTDPolicyEvaluation \
        hydra.run.dir="${directory}/TD Agent" \
        save_dir="$save_dir/TD/$run" \
        seed=$seed \
        kp=1 \
        ki=0 \
        kd=0 \
        gamma=0.999 \
        env="garnet $garnet_seed 50" \
        num_iterations=100000
done
