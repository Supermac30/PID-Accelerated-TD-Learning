#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1
#SBATCH --time=50:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=kd_final
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/errors/%x_%j.err
#SBATCH --account=deadline
#SBATCH --qos=deadline 

# regular dqn with atari presets

source slurm/setup.sh

current_time=$(date "+%Y.%m.%d-%H.%M.%S")
env=PongNoFrameskip-v4 # envs include PongNoFrameskip-v4, MsPacmanNoFrameskip-v4, SpaceInvadersNoFrameskip-v4, BreakoutNoFrameskip-v4 
directory=outputs/dqn_experiment/${env}/$current_time
echo "Saving to ${directory}"
mkdir -p "$directory"

tabular_d=False
is_double=True
num_runs=3

policy_evaluation=False
eval=True

adapt_gains=True
gain_adapter=SingleGainAdapter  # Options: NoGainAdapter, SingleGainAdapter, DiagonalGainAdapter, NetworkGainAdapter
meta_lr=1e-5
epsilon=0.1
d_tau=0.001

seed=$RANDOM

run_comparison=True  # Set this to false if the control is already computed
experiment_name="$env Atari DQN kd experiment"

if [ "$adapt_gains" = True ];
then
   python3 -m Experiments.DQNExperiments.DQNExperiment --multirun \
      env="$env" name="$env" experiment_name="$experiment_name" \
      hydra.mode=MULTIRUN \
      hydra.run.dir=$directory \
      hydra.sweep.dir=$directory \
      save_dir=$directory \
      seed=$seed \
      run_name=$seed \
      gain_adapter=$gain_adapter \
      adapt_gains=True \
      is_double=$is_double \
      use_previous_BRs=$use_previous_BRs \
      d_tau=$d_tau \
      epsilon=$epsilon \
      meta_lr=$meta_lr \
      tabular_d=$tabular_d \
      num_runs=$num_runs \
      policy_evaluation=$policy_evaluation \
      eval=$eval
else
   python3 -m Experiments.DQNExperiments.DQNExperiment --multirun \
      env="$env" name="$env" experiment_name="$experiment_name" \
      hydra.mode=MULTIRUN \
      hydra.run.dir=$directory \
      hydra.sweep.dir=$directory \
      save_dir=$directory \
      seed=$seed \
      run_name=$seed \
      is_double=$is_double \
      kp=1 \
      kd=0.001,0.01,0.1 \
      ki=0 \
      d_tau=0.1,0.01,0.001 \
      tabular_d=$tabular_d \
      num_runs=$num_runs \
      policy_evaluation=$policy_evaluation \
      eval=$eval
fi

if [ "$run_comparison" = True ];
then
   python3 -m Experiments.DQNExperiments.DQNExperiment --multirun \
      env="$env" name="$env" experiment_name="$experiment_name" \
      hydra.mode=MULTIRUN \
      hydra.run.dir=$directory \
      hydra.sweep.dir=$directory \
      save_dir=$directory \
      seed=$seed \
      run_name=$seed \
      is_double=$is_double \
      kp=1 \
      kd=0 \
      ki=0 \
      num_runs=$num_runs \
      visualize=$visualize \
      policy_evaluation=$policy_evaluation \
      eval=$eval \
      slow_motion=$slow_motion
fi

python3 -m Experiments.Plotting.plot_dqn_experiment \
   hydra.run.dir=$directory \
   save_dir=$directory \
   hydra/job_logging=disabled
