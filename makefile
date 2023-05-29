adapt:
	python3 -m Experiments.AdaptationExperiments.AdaptiveAgentExperiment
true_adapt:
	python3 -m Experiments.AdaptationExperiments.AdaptiveAgentPAVIAExperiment

TD:
	python3 -m Experiments.TDExperiments.SoftTDPolicyEvaluation
params:
	python3 -m Experiments.HyperparameterTests
Q:
	python3 -m Experiments.QExperiments.PIDQLearning
true_Q:
	python3 -m Experiments.VIExperiments.VIQControl
