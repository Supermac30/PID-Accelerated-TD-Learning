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
	python3 -m Experiments.DQNExperiments.DQNExperiment --config-name CartPole
DQN_lunarlander:
	python3 -m Experiments.DQNExperiments.DQNExperiment --config-name LunarLander
DQN_mountaincar:
	python3 -m Experiments.DQNExperiments.DQNExperiment --config-name MountainCar
DQN_acrobot:
	python3 -m Experiments.DQNExperiments.DQNExperiment --config-name Acrobot
DQN_atari:
	python3 -m Experiments.DQNExperiments.DQNExperiment --config-name Atari