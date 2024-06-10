#!/bin/bash
#SBATCH -N 1
#SBATCH -p cpu
#SBATCH --cpus-per-task=64
#SBATCH --tasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=adapt
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/errors/%x_%j.err

ulimit -n 2048

source slurm/setup.sh

current_time=$(date "+%Y.%m.%d/%H.%M.%S")
env="chain walk"
gamma=0.999
repeat=20
seed=$RANDOM
num_iterations=25000
search_steps=25000
recompute_optimal=True
compute_optimal=True  # False when we need to debug, so there is no multiprocessing
get_optimal=True  # False when we need to debug with a specific learning rate
debug=False

directory=outputs/adaptation_experiment/$env/$current_time
echo "Saving to ${directory}"
mkdir -p "$directory"

# python3 -m Experiments.TDExperiments.PastWorkEvaluation \
#     hydra.run.dir="${directory}/TD Agent" \
#     save_dir="$directory" \
#     seed=$seed \
#     agent_name=TIDBD \
#     gamma=$gamma \
#     repeat=$repeat \
#     env="$env" \
#     search_steps=$search_steps \
#     recompute_optimal=$recompute_optimal \
#     compute_optimal=$compute_optimal \
#     get_optimal=$get_optimal \
#     num_iterations=$num_iterations \
#     debug=$debug \
#     is_q=False \
#     name="TIDBD"

for meta_lr in 5e-8
do
for epsilon in 1e-3
do
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
    meta_lr=$meta_lr \
    epsilon=$epsilon  \
    alpha=0.95 \
    beta=0.05 \
    env="$env" \
    gamma=$gamma \
    repeat=$repeat \
    debug=$debug \
    num_iterations=$num_iterations \
    agent_name="semi gradient updater" \
    name="Gain Adaptation"
done
done

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

python3 -m Experiments.Plotting.plot_adaptation_experiment \
    hydra.run.dir="$directory" \
    save_dir="$directory" \
    repeat=$repeat \
    env="$env" \
    norm=1 \
    small_name=True \
    plot_best=False