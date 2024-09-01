#!/bin/bash
#SBATCH -N 1
#SBATCH -p cpu
#SBATCH --cpus-per-task=64
#SBATCH --tasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=pid
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/errors/%x_%j.err

source slurm/setup.sh
current_time=$(date "+%Y.%m.%d/%H.%M.%S")
env="chain walk"
gamma=0.99
repeat=6400
norm=1
seed=$RANDOM
num_iterations=50000
search_steps=50000
directory=outputs/pid_experiment/$env/$current_time
echo "Saving to $directory"
mkdir -p "$directory"

recompute_optimal=True
compute_optimal=True
get_optimal=True
debug=False

python3 -m Experiments.TDExperiments.SoftTDPolicyEvaluation \
    hydra.run.dir="$directory/TD Agent" \
    save_dir="$directory" \
    seed=$seed \
    search_steps=$search_steps \
    recompute_optimal=$recompute_optimal \
    compute_optimal=$compute_optimal \
    get_optimal=$get_optimal \
    kp=1 \
    ki=0 \
    kd=0 \
    gamma=$gamma \
    env="$env" \
    repeat=$repeat \
    debug=$debug \
    norm=$norm \
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

# python3 -m Experiments.TDExperiments.SoftTDPolicyEvaluation --multirun \
#     hydra.mode=MULTIRUN \
#     hydra.run.dir="$directory" \
#     hydra.sweep.dir="$directory" \
#     save_dir="$directory" \
#     seed=$seed \
#     kp=1 \
#     ki=0 \
#     kd="0.15" \
#     gamma=$gamma \
#     env="$env" \
#     repeat=$repeat \
#     num_iterations=$num_iterations \
#     search_steps=$search_steps \
#     recompute_optimal=$recompute_optimal \
#     compute_optimal=$compute_optimal \
#     get_optimal=$get_optimal \
#     debug=$debug \
#     'name="PID TD with $kappa_p=1$, $kappa_I=0$, $kappa_d=0.15$"'

python3 -m Experiments.Plotting.plot_adaptation_experiment \
    hydra.run.dir="$directory" \
    save_dir="$directory" \
    repeat=$repeat \
    env="$env" \
    norm=$norm \
    is_q=False \
    is_star=False \
    small_name=False \
    plot_best=False