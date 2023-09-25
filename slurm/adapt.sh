#!/bin/bash
#SBATCH -N 1
#SBATCH -p cpu
#SBATCH --cpus-per-task=64
#SBATCH --tasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --mem=1GB
#SBATCH --job-name=adapt_SA
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/errors/%x_%j.err

ulimit -n 2048

source slurm/setup.sh

current_time=$(date "+%Y.%m.%d/%H.%M.%S")
env="garnet 150 50"
gamma=0.99
repeat=3
seed=$RANDOM
num_iterations=1000
search_steps=1000
recompute_optimal=True
compute_optimal=True  # False when we need to debug, so there is no multiprocessing
get_optimal=True  # False when we need to debug with a specific learning rate
debug=False

directory=outputs/adaptation_experiment/$env/$current_time
echo "Saving to ${directory}"
mkdir -p "$directory"

python3 -m Experiments.AdaptationExperiments.AdaptiveAgentExperiment --multirun \
    hydra.mode=MULTIRUN \
    hydra.run.dir="$directory/Adaptive Agent" \
    hydra.sweep.dir="$directory" \
    seed=$seed \
    save_dir="$directory" \
    search_steps=$search_steps \
    recompute_optimal=$recompute_optimal \
    compute_optimal=$compute_optimal \
    get_optimal=$get_optimal \
    meta_lr=1e-2 \
    epsilon=1e-2 \
    env="$env" \
    gamma=$gamma \
    repeat=$repeat \
    debug=$debug \
    num_iterations=$num_iterations \
    agent_name="semi gradient updater" \

python3 -m Experiments.TDExperiments.SoftTDPolicyEvaluation \
    hydra.run.dir="$directory/TD Agent" \
    save_dir="$directory" \
    seed=$seed \
    search_steps=$search_steps \
    recompute_optimal=$recompute_search \
    kp=1 \
    ki=0 \
    kd=0 \
    gamma=$gamma \
    env="$env" \
    repeat=$repeat \
    debug=$debug \
    num_iterations=$num_iterations \

python3 -m Experiments.Plotting.plot_adaptation_experiment \
    hydra.run.dir="$directory" \
    save_dir="$directory" \
    plot_best=True \
    repeat=$repeat \
    env="$env" \
    small_name=True