import matplotlib.pyplot as plt
import numpy as np
import hydra

from Environments import ChainWalk
from Agents import ControlledTDLearning, SoftControlledTDLearning
from Experiments.ExperimentHelpers import *

@hydra.main(version_base=None, config_path="../../config/ComparisonExperiments", config_name="HardSoftComparison")
def hard_soft_convergence_experiment(cfg):
    """Experiment with the convergence rate of hard and soft derivative updates."""
    num_states = 50
    num_actions = 2
    env = ChainWalk(num_states)
    policy = np.zeros((num_states, num_actions))
    for i in range(num_states):
        policy[i,0] = 1
    soft = SoftControlledTDLearning(
        env,
        policy,
        0.99,
        learning_rate_function(1, 0),
        learning_rate_function(1, 0)
    )
    hard = ControlledTDLearning(
        env,
        policy,
        0.99,
        learning_rate_function(1, 0)
    )

    V_pi = find_Vpi(env, policy)
    test_function = lambda V, Vp, BR: np.max(np.abs(V - V_pi))

    fig, (ax1, ax2) = plt.subplots(2)

    for kp, kd, ki in zip(cfg['kp'], cfg['kd'], cfg['ki']):
        soft_history, soft_rates = find_optimal_pid_learning_rates(
            soft,
            kp,
            kd,
            ki,
            test_function,
            cfg['num_iterations'],
            cfg['threshold'],
            True
        )
        hard_history, hard_rates = find_optimal_pid_learning_rates(
            hard,
            kp,
            kd,
            ki,
            test_function,
            cfg['num_iterations'],
            cfg['threshold'],
            False
        )

        save_array(soft_history, f"soft: {kp=} {kd=} {ki=} {soft_rates}", ax1)
        save_array(hard_history, f"hard: {kp=} {kd=} {ki=} {hard_rates}", ax2)

    plot_comparison(fig, ax1, ax2, 'Soft Updates', 'Hard Updates', '$||V_k - V^\pi||_\infty$')


if __name__ == '__main__':
    hard_soft_convergence_experiment()