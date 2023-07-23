#!/bin/bash
#SBATCH -N 1
#SBATCH -p cpu
#SBATCH --cpus-per-task=32
#SBATCH --tasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --mem=1GB
#SBATCH --job-name=adapt_Q
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/errors/%x_%j.err

source ~/.bashrc
source ~/newgym.nv
conda activate myenv

# CHANGE THIS TO YOUR OWN PATH
cd /h/bedaywim/PID-Accelerated-TD-Learning

current_time=$(date "+%Y.%m.%d/%H.%M.%S")
env="cliff walk"
gamma=0.999
repeat=20
seed=$RANDOM
num_iterations=100000
directory=outputs/q_adaptation_experiment/$env/$current_time
echo "Saving to $directory"
mkdir -p "$directory"

python3 -m Experiments.AdaptationExperiments.AdaptiveQAgentExperiment --multirun \
    hydra.mode=MULTIRUN \
    hydra.run.dir="$directory/Adaptive Agent" \
    hydra.sweep.dir="$directory" \
    seed=$seed \
    save_dir="$directory" \
    meta_lr=1e-4 \
    epsilon=1e-1 \
    env="$env" \
    gamma=$gamma \
    repeat=$repeat \
    num_iterations=$num_iterations \
    agent_name="diagonal semi gradient Q updater" \
    recompute_optimal=True

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
    num_iterations=$num_iterations

python3 -m Experiments.Plotting.plot_adaptation_experiment \
    hydra.run.dir="$directory" \
    save_dir="$directory" \
    repeat=$repeat \
    env="$env" \
    is_q=True \
    plot_best=True \
