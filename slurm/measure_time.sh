#!/bin/bash
#SBATCH -N 1
#SBATCH -p cpu
#SBATCH --cpus-per-task=64
#SBATCH --tasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=measure_time
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/errors/%x_%j.err

ulimit -n 2048

source slurm/setup.sh

current_time=$(date "+%Y.%m.%d/%H.%M.%S")
gamma=0.99
repeat=20
num_iterations=20000
search_steps=20000
recompute_optimal=True
compute_optimal=True  # False when we need to debug, so there is no multiprocessing
get_optimal=True  # False when we need to debug with a specific learning rate
debug=False

directory=outputs/time_experiment/$env/$current_time
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

for states in 10 20 30 40 50 60 70 80 90 100
do
    seed=$RANDOM
	garnet_seed=$RANDOM
    python3 -m Experiments.TDExperiments.PastWorkEvaluation \
        hydra.run.dir="${directory}/Zap Agent" \
        save_dir="$directory" \
        agent_name="zap Q learning" \
        gamma=$gamma \
        seed=$seed \
        repeat=$repeat \
        env="garnet $garnet_seed $states" \
        norm=$norm \
        is_q=True \
        recompute_optimal=$recompute_optimal \
        compute_optimal=$compute_optimal \
        get_optimal=$get_optimal \
        debug=$debug \
        num_iterations=$num_iterations \
        search_steps=$search_steps \
        measure_time=True \
        name="Zap Q Learning"

    python3 -m Experiments.AdaptationExperiments.AdaptiveQAgentExperiment --multirun \
        hydra.run.dir="$directory/Adaptive Q Agent" \
        hydra.sweep.dir="$directory" \
        seed=$seed \
        save_dir="$save_dir/gain_adaptation/$run" \
        env="garnet $garnet_seed $states" \
        gamma=$gamma \
        num_iterations=$num_iterations \
        search_steps=$search_steps \
        recompute_optimal=$recompute_optimal \
        compute_optimal=$compute_optimal \
        get_optimal=$get_optimal \
        meta_lr=1e-5 \
        epsilon=0.01 \
        lambda=0 \
        alpha=0.95 \
        beta=0.05 \
        agent_name="semi gradient Q updater" \
        measure_time=True \
        name="Gain Adaptation"

    python3 -m Experiments.QExperiments.PIDQLearning \
        hydra.run.dir="$directory/Q Agent" \
        save_dir="$save_dir/Q/$run" \
        seed=$seed \
        kp=1 \
        ki=0 \
        kd=0 \
        repeat=$repeat \
        agent_name="Q learning" \
        seed=$seed \
        gamma=$gamma \
        env="garnet $garnet_seed $states" \
        num_iterations=$num_iterations \
        search_steps=$search_steps \
        recompute_optimal=$recompute_optimal \
        compute_optimal=$compute_optimal \
        get_optimal=$get_optimal \
        measure_time=True \
        name="Q Learning"
done

python3 -m Experiments.Plotting.plot_time \
    hydra.run.dir="$directory" \
    save_dir="$directory" \
    repeat=$repeat \
    env="$env" \
    norm=1 \
    small_name=True \
    plot_best=False