#!/bin/bash
#SBATCH -N 1
#SBATCH -p cpu
#SBATCH --cpus-per-task=64
#SBATCH --tasks-per-node=1
#SBATCH --time=100:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=garnet_adapt
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/errors/%x_%j.err

source slurm/setup.sh

current_time=$(date "+%Y.%m.%d/%H.%M.%S")
save_dir=$1
num_iterations=5000
search_steps=5000
directory=$save_dir/$current_time
echo "Saving to $directory"
mkdir -p "$directory"

gamma=0.99

recompute_optimal=False
compute_optimal=True
get_optimal=True

is_q=$4

for run in $(seq $2 $3)
do
    mkdir -p "$save_dir/gain_adaptation/$run"
    mkdir -p "$save_dir/TD/$run"

	seed=$RANDOM
	garnet_seed=$RANDOM

    # if is_q == 'False'
    if [ "$is_q" = "False" ]; then
        python3 -m Experiments.AdaptationExperiments.AdaptiveAgentExperiment --multirun \
            hydra.run.dir="$directory/Adaptive Agent" \
            hydra.sweep.dir="$directory" \
            seed=$seed \
            save_dir="$save_dir/gain_adaptation/$run" \
            meta_lr=1e-1,1e-2,1e-3,1e-4 \
            epsilon=1e-1,1e-2 \
            env="garnet $garnet_seed 50" \
            gamma=$gamma \
            agent_name="semi gradient updater" \
            num_iterations=$num_iterations \
            search_steps=$search_steps \
            recompute_optimal=$recompute_optimal \
            compute_optimal=$compute_optimal \
            get_optimal=$get_optimal

        python3 -m Experiments.TDExperiments.SoftTDPolicyEvaluation \
            hydra.run.dir="$directory/TD Agent" \
            save_dir="$save_dir/TD/$run" \
            seed=$seed \
            kp=1 \
            ki=0 \
            kd=0 \
            gamma=$gamma \
            env="garnet $garnet_seed 50" \
            num_iterations=$num_iterations \
            search_steps=$search_steps \
            recompute_optimal=$recompute_optimal \
            compute_optimal=$compute_optimal \
            get_optimal=$get_optimal
    else
        python3 -m Experiments.AdaptationExperiments.AdaptiveQAgentExperiment --multirun \
            hydra.mode=MULTIRUN \
            hydra.run.dir="$directory/Adaptive Agent" \
            hydra.sweep.dir="$directory" \
            seed=$seed \
            save_dir="$directory" \
            meta_lr_p=1e-1,1e-2 \
            meta_lr_I=1e-1,1e-2 \
            meta_lr_d=1e-3,1e-4 \
            epsilon=0.0001 \
            env="$env" \
            gamma=$gamma \
            repeat=$repeat \
            num_iterations=$num_iterations \
            search_steps=$search_steps \
            agent_name="diagonal semi gradient Q updater" \
            recompute_optimal=$recompute_optimal \
            compute_optimal=$compute_optimal \
            get_optimal=$get_optimal \
            debug=$debug

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
            num_iterations=$num_iterations \
            search_steps=$search_steps \
            agent_name="Q learning" \
            recompute_optimal=$recompute_optimal \
            compute_optimal=$compute_optimal \
            get_optimal=$get_optimal \
    fi
done

