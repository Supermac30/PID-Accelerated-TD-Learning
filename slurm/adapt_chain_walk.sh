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
gamma=0.99
repeat=6400
norm=1
seed=$RANDOM
num_iterations=50000
search_steps=50000
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

for meta_lr in 1e-7
do
for epsilon in 1e-1
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
    norm=$norm \
    env="$env" \
    gamma=$gamma \
    repeat=$repeat \
    debug=$debug \
    num_iterations=$num_iterations \
    agent_name="semi gradient updater" \
    measure_time=False \
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
    norm=$norm \
    gamma=$gamma \
    env="$env" \
    repeat=$repeat \
    debug=$debug \
    num_iterations=$num_iterations \
    measure_time=False \
    name="TD"

python3 -m Experiments.TDExperiments.SoftTDPolicyEvaluation \
    hydra.mode=MULTIRUN \
    hydra.run.dir="$directory" \
    hydra.sweep.dir="$directory" \
    save_dir="$directory" \
    seed=$seed \
    kp=1 \
    ki="-0.4" \
    kd=0 \
    gamma=$gamma \
    env="$env" \
    repeat=$repeat \
    num_iterations=$num_iterations \
    search_steps=$search_steps \
    recompute_optimal=$recompute_optimal \
    compute_optimal=$compute_optimal \
    get_optimal=$get_optimal \
    debug=$debug \
    norm=$norm \
    'name="PID TD with $\kappa_p=1$, $\kappa_I=-0.4$, $\kappa_d=0$"'

python3 -m Experiments.Plotting.plot_adaptation_experiment \
    hydra.run.dir="$directory" \
    save_dir="$directory" \
    repeat=$repeat \
    env="$env" \
    norm=$norm \
    small_name=True \
    is_q=False \
    plot_best=False