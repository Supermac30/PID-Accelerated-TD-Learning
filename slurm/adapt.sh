#!/bin/bash
#SBATCH -N 1
#SBATCH -p cpu
#SBATCH --cpus-per-task=16
#SBATCH --tasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --mem=1GB
#SBATCH --job-name=adapt_SA
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/errors/%x_%j.err

ulimit -n 2048

source ~/.bashrc
source ~/newgym.nv
conda activate myenv

# CHANGE THIS TO YOUR OWN PATH
cd /h/bedaywim/PID-Accelerated-TD-Learning

current_time=$(date "+%Y.%m.%d/%H.%M.%S")
env="chain walk"
gamma=0.999
repeat=20
seed=$RANDOM
num_iterations=30000
directory=outputs/adaptation_experiment/$env/$current_time
echo "Saving to ${directory}"
mkdir -p "$directory"

python3 -m Experiments.AdaptationExperiments.AdaptiveAgentExperiment --multirun \
    hydra.mode=MULTIRUN \
    hydra.run.dir="$directory/Adaptive Agent" \
    hydra.sweep.dir="$directory" \
    seed=$seed \
    save_dir="$directory" \
    meta_lr=1e-3,5e-4,1e-4 \
    epsilon=0.5 \
    env="$env" \
    gamma=$gamma \
    repeat=$repeat \
    num_iterations=$num_iterations

python3 -m Experiments.TDExperiments.SoftTDPolicyEvaluation \
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
    plot_best=True \
    repeat=$repeat \
    name="TD" \
    env="$env"