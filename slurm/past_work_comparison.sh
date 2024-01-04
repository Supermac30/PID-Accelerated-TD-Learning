#!/bin/bash
#SBATCH -N 1
#SBATCH -p cpu
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --mem=1GB
#SBATCH --job-name=past_work_comparison
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/errors/%x_%j.err

source slurm/setup.sh

current_time=$(date "+%Y.%m.%d/%H.%M.%S")
env="cliff walk"
seed=$RANDOM
gamma=0.999
repeat=20
directory=outputs/past_work_comparison/${env}/${current_time}
echo "Saving to ${directory}"
mkdir -p "$directory"

python3 -m Experiments.TDExperiments.PastWorkEvaluation \
    hydra.run.dir="${directory}/TD Agent" \
    save_dir="$directory" \
    seed=$seed \
    agent_name=TIDBD \
    gamma=$gamma \
    repeat=$repeat \
    env="$env" \
    recompute_optimal=True \
    is_q=False \
    name="TIDBD"

python3 -m Experiments.AdaptationExperiments.AdaptiveAgentExperiment \
    hydra.run.dir="${directory}/TD Agent" \
    save_dir="$directory" \
    seed=$seed \
    meta_lr=0.1 \
    epsilon=0.1 \
    agent_name="semi gradient updater" \
    gamma=$gamma \
    repeat=$repeat \
    env="$env" \
    recompute_optimal=True \
    name="PID TD"

python3 -m Experiments.Plotting.plot_adaptation_experiment \
    hydra.run.dir="$directory" \
    save_dir="$directory" \
    env="$env" \
    hydra/job_logging=disabled
