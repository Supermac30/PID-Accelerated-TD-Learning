#!/bin/bash
#SBATCH -N 1
#SBATCH -p cpu
#SBATCH --cpus-per-task=64
#SBATCH --tasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --mem=1GB
#SBATCH --job-name=past_work_comparison
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/errors/%x_%j.err

source slurm/setup.sh

current_time=$(date "+%Y.%m.%d/%H.%M.%S")
env="grid world"
seed=$RANDOM
gamma=0.99
repeat=20
norm="1"
directory=outputs/past_work_comparison/${env}/${current_time}
echo "Saving to ${directory}"
mkdir -p "$directory"
num_iterations=2500
search_steps=2500

recompute_optimal=False
compute_optimal=True
get_optimal=True
debug=False

python3 -m Experiments.TDExperiments.SoftTDPolicyEvaluation \
    hydra.run.dir="$directory/TD Agent" \
    save_dir="$directory" \
    seed=$seed \
    search_steps=$search_steps \
    recompute_optimal=$recompute_search \
    compute_optimal=$compute_optimal \
    get_optimal=$get_optimal \
    kp=1 \
    ki=0 \
    kd=0 \
    gamma=$gamma \
    env="$env" \
    repeat=$repeat \
    debug=$debug \
    num_iterations=$num_iterations \
    name="TD"

python3 -m Experiments.TDExperiments.PastWorkEvaluation \
    hydra.run.dir="${directory}/TD Agent" \
    save_dir="$directory" \
    seed=$seed \
    agent_name=TIDBD \
    gamma=$gamma \
    repeat=$repeat \
    env="$env" \
    search_steps=$search_steps \
    num_iterations=$num_iterations \
    recompute_optimal=$recompute_optimal \
    compute_optimal=$compute_optimal \
    get_optimal=$get_optimal \
    debug=$debug \
    is_q=False \
    norm=$norm \
    name="TIDBD"

python3 -m Experiments.AdaptationExperiments.AdaptiveAgentExperiment \
    hydra.run.dir="${directory}/TD Agent" \
    save_dir="$directory" \
    seed=$seed \
    meta_lr=1e-3 \
    epsilon=1e-3 \
    agent_name="semi gradient updater" \
    gamma=$gamma \
    repeat=$repeat \
    env="$env" \
    search_steps=$search_steps \
    num_iterations=$num_iterations \
    recompute_optimal=$recompute_optimal \
    compute_optimal=$compute_optimal \
    get_optimal=$get_optimal \
    debug=$debug \
    norm=$norm \
    name="Gain Adaptation"

python3 -m Experiments.Plotting.plot_adaptation_experiment \
    hydra.run.dir="$directory" \
    save_dir="$directory" \
    env="$env" \
    is_q=False \
    plot_best=False \
    norm=$norm \
    hydra/job_logging=disabled