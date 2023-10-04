#!/bin/bash
#SBATCH -N 1
#SBATCH -p cpu
#SBATCH --cpus-per-task=64
#SBATCH --tasks-per-node=1
#SBATCH --time=20:00:00
#SBATCH --mem=1GB
#SBATCH --job-name=pid_Q
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/errors/%x_%j.err

source slurm/setup.sh
current_time=$(date "+%Y.%m.%d/%H.%M.%S")
env="cliff walk"
gamma=0.9
repeat=20
norm="1"
seed=$RANDOM
num_iterations=2000
search_steps=2000
directory=outputs/pid_Q_experiment/$env/$current_time
echo "Saving to $directory"
mkdir -p "$directory"

recompute_optimal=True
compute_optimal=True
get_optimal=True
debug=False

python3 -m Experiments.QExperiments.PIDQLearning --multirun \
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
    norm="$norm" \
    agent_name="Q learning" \
    recompute_optimal=$recompute_optimal \
    compute_optimal=$compute_optimal \
    get_optimal=$get_optimal \
    debug=$debug

python3 -m Experiments.QExperiments.PIDQLearning --multirun \
    hydra.mode=MULTIRUN \
    hydra.run.dir="$directory" \
    hydra.sweep.dir="$directory" \
    save_dir="$directory" \
    seed=$seed \
    kp=1 \
    ki=0.2 \
    kd=0.2 \
    gamma=$gamma \
    env="$env" \
    repeat=$repeat \
    num_iterations=$num_iterations \
    search_steps=$search_steps \
    norm="$norm" \
    agent_name="Q learning" \
    recompute_optimal=$recompute_optimal \
    compute_optimal=$compute_optimal \
    get_optimal=$get_optimal \
    debug=$debug

python3 -m Experiments.QExperiments.PIDQLearning --multirun \
    hydra.mode=MULTIRUN \
    hydra.run.dir="$directory" \
    hydra.sweep.dir="$directory" \
    save_dir="$directory" \
    seed=$seed \
    kp=1 \
    ki=0.2 \
    kd=0.2 \
    gamma=$gamma \
    env="$env" \
    repeat=$repeat \
    num_iterations=$num_iterations \
    norm="$norm" \
    agent_name="true double Q learning" \
    recompute_optimal=True

python3 -m Experiments.Plotting.plot_adaptation_experiment \
    hydra.run.dir="$directory" \
    save_dir="$directory" \
    repeat=$repeat \
    env="$env" \
    is_q=True \
    plot_best=False \
    norm="$norm"