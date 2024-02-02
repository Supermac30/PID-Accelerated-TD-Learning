#!/bin/bash
#SBATCH -N 1
#SBATCH -p cpu
#SBATCH --cpus-per-task=32
#SBATCH --tasks-per-node=1
#SBATCH --time=20:00:00
#SBATCH --mem=1GB
#SBATCH --job-name=linear_garnet_tests
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/errors/%x_%j.err

source slurm/setup.sh

current_time=$(date "+%Y.%m.%d/%H.%M.%S")
save_dir=$1
num_iterations=5000
search_steps=5000
directory=$save_dir/$current_time
echo "Saving to $directory"
mkdir -p "$directory"

gamma=0.99

recompute_optimal=False
compute_optimal=True
get_optimal=True

is_q=$4

for run in $(seq $2 $3)
do
    mkdir -p "$save_dir/gain_adaptation/$run"
    mkdir -p "$save_dir/TD/$run"

	seed=$RANDOM
	garnet_seed=$RANDOM

    python3 -m Experiments.LinearFAExperiments.AdaptiveLinearFAExperiment --multirun \
        hydra.mode=MULTIRUN \
        hydra.run.dir="$directory/Adaptive Agent" \
        hydra.sweep.dir="$directory" \
        save_dir="$directory" \
        seed=$seed \
        order=$order \
        gamma=$gamma \
        env="garnet $garnet_seed 50" \
        repeat=$repeat \
        num_iterations=$num_iterations \
        type="$type" \
        compute_optimal=$compute_optimal \
        get_optimal=$get_optimal \
        recompute_optimal=$recompute_optimal \
        debug=$debug \
        search_steps=$search_steps \
        is_q=$is_q \
        epsilon=0.1 \
        meta_lr=2e-6 \
        name="Gain Adaptation <meta_lr> <epsilon>"

    python3 -m Experiments.LinearFAExperiments.LinearFAExperiment --multirun \
        hydra.mode=MULTIRUN \
        hydra.run.dir="$directory/TD Agent" \
        hydra.sweep.dir="$directory" \
        save_dir="$save_dir/gain_adaptation/$run" \
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
        debug=$debug \
        search_steps=$search_steps \
        is_q=$is_q \
        name="TD"
done

