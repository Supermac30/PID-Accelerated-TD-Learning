#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --mem=1GB
#SBATCH --job-name=past_work_comparison
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/errors/%x_%j.err

source ~/.bashrc
source ~/newgym.nv
conda activate myenv

# CHANGE THIS TO YOUR OWN PATH
cd /h/bedaywim/PID-Accelerated-TD-Learning

current_time=$(date "+%Y.%m.%d/%H.%M.%S")
env="cliff walk"
seed=$RANDOM
directory=outputs/adaptation_experiment/${env}/${current_time}
echo "Saving to ${directory}"
mkdir -p "$directory"

python3 -m Experiments.TDExperiments.PastWorkEvaluation \
    hydra.run.dir="${directory}/TD Agent" \
    save_dir="$directory" \
    seed=$seed \
    agent_name=TIDBD \
    gamma=0.999 \
    env="$env"

python3 -m Experiments.TDExperiments.PastWorkEvaluation \
    hydra.run.dir="${directory}/TD Agent" \
    save_dir="$directory" \
    seed=$seed \
    agent_name="speedy Q learning" \
    gamma=0.999 \
    env="$env"

python3 -m Experiments.TDExperiments.PastWorkEvaluation \
    hydra.run.dir="${directory}/TD Agent" \
    save_dir="$directory" \
    agent_name="zap Q learning" \
    gamma=0.999 \
    env="$env"

python3 -m Experiments.TDExperiments.SoftTDPolicyEvaluation \
    hydra.run.dir="${directory}/TD Agent" \
    save_dir="$directory" \
    seed=$seed \
    kp=1 \
    ki=0 \
    kd=0 \
    gamma=0.999 \
    env="$env"

python3 -m Experiments.Plotting.plot_adaptation_experiment \
    hydra.run.dir="$directory" \
    save_dir="$directory" \
    hydra/job_logging=disabled
