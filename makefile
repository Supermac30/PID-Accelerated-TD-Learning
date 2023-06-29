adapt:
	python3 -m Experiments.AdaptationExperiments.AdaptiveAgentExperiment
true_adapt:
	python3 -m Experiments.AdaptationExperiments.AdaptiveAgentPAVIAExperiment
Q_adapt:
	python3 -m Experiments.AdaptationExperiments.AdaptiveQAgentExperiment
TD:
	python3 -m Experiments.TDExperiments.SoftTDPolicyEvaluation
params:
	python3 -m Experiments.HyperparameterTests
Q:
	python3 -m Experiments.QExperiments.PIDQLearning
true_Q:
	python3 -m Experiments.VIExperiments.VIQControl

DQN_cartpole:
	python3 -m Experiments.DQNExperiments.DQNExperiment env=cartpole name=cartpole
DQN_lunarlander:
	python3 -m Experiments.DQNExperiments.DQNExperiment env=lunarlander name=lunarlander
DQN_mountaincar:
	python3 -m Experiments.DQNExperiments.DQNExperiment env=mountaincar name=mountaincar
DQN_acrobot:
	python3 -m Experiments.DQNExperiments.DQNExperiment env=acrobot name=acrobot
DQN_atari:
	python3 -m Experiments.DQNExperiments.DQNExperiment env=atari name=atari

DQN_slurm:
	sbatch slurm/dqn.sh