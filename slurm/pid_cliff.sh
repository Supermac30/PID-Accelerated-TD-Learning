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
env="cliff walk"
gamma=0.99
repeat=6400
seed=$RANDOM
num_iterations=20000
search_steps=20000
directory=outputs/pid_experiment/$env/$current_time
echo "Saving to $directory"
mkdir -p "$directory"

recompute_optimal=True
compute_optimal=True
get_optimal=True
debug=False

python3 -m Experiments.TDExperiments.SoftTDPolicyEvaluation --multirun \
    hydra.mode=MULTIRUN \
    hydra.run.dir="$directory" \
    hydra.sweep.dir="$directory" \
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
    recompute_optimal=$recompute_optimal \
    compute_optimal=$compute_optimal \
    get_optimal=$get_optimal \
    debug=$debug \
    name="TD"

python3 -m Experiments.TDExperiments.SoftTDPolicyEvaluation --multirun \
    hydra.mode=MULTIRUN \
    hydra.run.dir="$directory" \
    hydra.sweep.dir="$directory" \
    save_dir="$directory" \
    seed=$seed \
    kp=1 \
    ki="0.3" \
    kd="0.2" \
    gamma=$gamma \
    env="$env" \
    repeat=$repeat \
    num_iterations=$num_iterations \
    search_steps=$search_steps \
    recompute_optimal=$recompute_optimal \
    compute_optimal=$compute_optimal \
    get_optimal=$get_optimal \
    debug=$debug \
    'name="PID TD with $kp=1$, $ki=0.3$, $kd=0.2$"'

python3 -m Experiments.Plotting.plot_adaptation_experiment \
    hydra.run.dir="$directory" \
    save_dir="$directory" \
    repeat=$repeat \
    env="$env" \
    is_q=False \
    is_star=False \
    small_name=False \
    plot_best=False