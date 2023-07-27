#!/bin/bash
#SBATCH -N 1
#SBATCH -p cpu
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --mem=1GB
#SBATCH --job-name=past_work_comparison_Q
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/errors/%x_%j.err

source slurm/setup.sh

current_time=$(date "+%Y.%m.%d/%H.%M.%S")
env="cliff walk"
seed=$RANDOM
gamma=0.999
repeat=20
directory=outputs/past_work_comparison_Q/${env}/${current_time}
echo "Saving to ${directory}"
mkdir -p "$directory"

python3 -m Experiments.TDExperiments.PastWorkEvaluation \
    hydra.run.dir="${directory}/Speedy Agent" \
    save_dir="$directory" \
    seed=$seed \
    agent_name="speedy Q learning" \
    gamma=0.999 \
    repeat=$repeat \
    env="$env" \
    is_q=True \
    recompute_optimal=True

python3 -m Experiments.TDExperiments.PastWorkEvaluation \
    hydra.run.dir="${directory}/Zip Agent" \
    save_dir="$directory" \
    agent_name="zap Q learning" \
    gamma=$gamma \
    seed=$seed \
    repeat=$repeat \
    env="$env" \
    is_q=True \
    recompute_optimal=True

python3 -m Experiments.AdaptationExperiments.AdaptiveQAgentExperiment \
    hydra.run.dir="${directory}/PID Agent" \
    save_dir="$directory" \
    seed=$seed \
    meta_lr=1e-3 \
    epsilon=10 \
    agent_name="diagonal semi gradient Q updater" \
    gamma=$gamma \
    repeat=$repeat \
    env="$env" \
    recompute_optimal=True

python3 -m Experiments.Plotting.plot_adaptation_experiment \
    hydra.run.dir="$directory" \
    save_dir="$directory" \
    env="$env" \
    is_q=True \
    hydra/job_logging=disabled
