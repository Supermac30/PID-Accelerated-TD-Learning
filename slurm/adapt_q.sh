#!/bin/bash
#SBATCH -N 1
#SBATCH -p cpu
#SBATCH --cpus-per-task=64
#SBATCH --tasks-per-node=1
#SBATCH --time=30:00:00
#SBATCH --mem=1GB
#SBATCH --job-name=adapt_Q
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/errors/%x_%j.err

source slurm/setup.sh
current_time=$(date "+%Y.%m.%d/%H.%M.%S")
env="cliff walk"
gamma=0.99
repeat=20
seed=$RANDOM
num_iterations=10000
search_steps=10000
directory=outputs/q_adaptation_experiment/$env/$current_time
echo "Saving to $directory"
mkdir -p "$directory"

recompute_optimal=True
compute_optimal=True
get_optimal=True
debug=False

python3 -m Experiments.AdaptationExperiments.AdaptiveQAgentExperiment --multirun \
    hydra.mode=MULTIRUN \
    hydra.run.dir="$directory/Adaptive Agent" \
    hydra.sweep.dir="$directory" \
    seed=$seed \
    save_dir="$directory" \
    meta_lr=1e-3,1e-4 \
    epsilon=0.1 \
    env="$env" \
    gamma=$gamma \
    repeat=$repeat \
    num_iterations=$num_iterations \
    search_steps=$search_steps \
    agent_name="semi gradient double Q updater" \
    recompute_optimal=$recompute_optimal \
    compute_optimal=$compute_optimal \
    get_optimal=$get_optimal \
    debug=$debug

python3 -m Experiments.QExperiments.PIDQLearning \
    hydra.run.dir="$directory/TD Agent" \
    save_dir="$directory" \
    seed=$seed \
    kp=1 \
    ki=0 \
    kd=0 \
    gamma=$gamma \
    env="$env" \
    repeat=$repeat \
    num_iterations=$num_iterations \
    search_steps=$search_steps \
    agent_name="Q learning" \
    recompute_optimal=$recompute_optimal \
    compute_optimal=$compute_optimal \
    get_optimal=$get_optimal \

python3 -m Experiments.Plotting.plot_adaptation_experiment \
    hydra.run.dir="$directory" \
    save_dir="$directory" \
    repeat=$repeat \
    env="$env" \
    is_q=False \
    is_double_q=True \
    plot_best=True \
    small_name=True