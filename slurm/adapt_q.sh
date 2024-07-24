#!/bin/bash
#SBATCH -N 1
#SBATCH -p cpu
#SBATCH --cpus-per-task=64
#SBATCH --tasks-per-node=1
#SBATCH --time=40:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=adapt_q
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/errors/%x_%j.err

source slurm/setup.sh
current_time=$(date "+%Y.%m.%d/%H.%M.%S.%3N")
env="zap MDP"
gamma=0.99
repeat=20
seed=$RANDOM
norm="BR"  # 1, 2, inf, fro, BR
num_iterations=100000
search_steps=100000
directory=outputs/q_adaptation_experiment/$env/$current_time
echo "Saving to $directory"
mkdir -p "$directory"

recompute_optimal=False
compute_optimal=True
get_optimal=True
debug=False

python3 -m Experiments.TDExperiments.PastWorkEvaluation \
    hydra.run.dir="${directory}/Zap Agent" \
    save_dir="$directory" \
    agent_name="zap Q learning" \
    gamma=$gamma \
    seed=$seed \
    repeat=$repeat \
    env="$env" \
    norm=$norm \
    is_q=True \
    recompute_optimal=$recompute_optimal \
    compute_optimal=$compute_optimal \
    get_optimal=$get_optimal \
    debug=$debug \
    num_iterations=$num_iterations \
    search_steps=$search_steps \
    name="Zap Q Learning"

# python3 -m Experiments.TDExperiments.PastWorkEvaluation \
#     hydra.run.dir="${directory}/Speedy Agent" \
#     save_dir="$directory" \
#     seed=$seed \
#     agent_name="speedy Q learning" \
#     gamma=$gamma \
#     repeat=$repeat \
#     env="$env" \
#     norm=$norm \
#     is_q=True \
#     recompute_optimal=$recompute_optimal \
#     compute_optimal=$compute_optimal \
#     get_optimal=$get_optimal \
#     num_iterations=$num_iterations \
#     search_steps=$search_steps \
#     debug=$debug \
#     name="Speedy Q Learning"

# for meta_lr in 8e-9
# do
# for epsilon in 1e-3
# do
# python3 -m Experiments.AdaptationExperiments.AdaptiveQAgentExperiment --multirun \
#     hydra.mode=MULTIRUN \
#     hydra.run.dir="$directory/Adaptive Agent" \
#     hydra.sweep.dir="$directory" \
#     seed=$seed \
#     save_dir="$directory" \
#     meta_lr_p=$meta_lr \
#     meta_lr_I=$meta_lr \
#     meta_lr_d=$meta_lr \
#     alpha=0.95 \
#     beta=0.05 \
#     epsilon=$epsilon \
#     lambda=0 \
#     env="$env" \
#     gamma=$gamma \
#     repeat=$repeat \
#     num_iterations=$num_iterations \
#     search_steps=$search_steps \
#     agent_name="semi gradient Q updater" \
#     recompute_optimal=$recompute_optimal \
#     compute_optimal=$compute_optimal \
#     get_optimal=$get_optimal \
#     norm=$norm \
#     debug=$debug \
#     name="Gain Adaptation"
# done
# done

# python3 -m Experiments.QExperiments.PIDQLearning \
#     hydra.run.dir="$directory/TD Agent" \
#     save_dir="$directory" \
#     seed=$seed \
#     kp=1 \
#     ki=0 \
#     kd=0 \
#     gamma=$gamma \
#     env="$env" \
#     repeat=$repeat \
#     num_iterations=$num_iterations \
#     search_steps=$search_steps \
#     agent_name="Q learning" \
#     recompute_optimal=$recompute_optimal \
#     compute_optimal=$compute_optimal \
#     get_optimal=$get_optimal \
#     debug=$debug \
#     norm=$norm \
#     name="Q Learning"

python3 -m Experiments.Plotting.plot_adaptation_experiment \
    hydra.run.dir="$directory" \
    save_dir="$directory" \
    repeat=$repeat \
    env="$env" \
    is_q=True \
    norm=$norm \
    normalize=False \
    plot_best=False \
    small_name=True