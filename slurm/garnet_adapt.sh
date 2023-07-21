#!/bin/bash
#SBATCH -N 1
#SBATCH -p cpu
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
num_iterations=500000
directory=$save_dir/$current_time
echo "Saving to $directory"
mkdir -p "$directory"

for run in $(seq $2 $3)
do
    mkdir -p "$save_dir/gain_adaptation/$run"
    mkdir -p "$save_dir/TD/$run"

	seed=$RANDOM
	garnet_seed=$RANDOM

    python3 -m Experiments.AdaptationExperiments.AdaptiveAgentExperiment --multirun \
        hydra.run.dir="$directory/Adaptive Agent" \
        hydra.sweep.dir="$directory" \
        seed=$seed \
        save_dir="$save_dir/gain_adaptation/$run" \
        meta_lr=1e-1,1e-2,1e-3 \
        epsilon=1e-1,1e-2,1e-3 \
        env="garnet $garnet_seed 50" \
        gamma=0.999 \
        agent_name="diagonal semi gradient updater" \
        num_iterations=$num_iterations

	python3 -m Experiments.TDExperiments.SoftTDPolicyEvaluation \
        hydra.run.dir="$directory/TD Agent" \
        save_dir="$save_dir/TD/$run" \
        seed=$seed \
        kp=1 \
        ki=0 \
        kd=0 \
        gamma=0.999 \
        env="garnet $garnet_seed 50" \
        num_iterations=$num_iterations
done
