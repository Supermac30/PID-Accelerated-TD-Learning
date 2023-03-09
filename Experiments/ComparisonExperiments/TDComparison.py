"""Compare TD Learning with PID-Accelerated TD Learning"""

import matplotlib.pyplot as plt
import numpy as np
import hydra

from Environments import ChainWalk
from Agents import ControlledTDLearning, SoftControlledTDLearning
from Experiments.ExperimentHelpers import *

@hydra.main(version_base=None, config_path="../../config/ComparisonExperiments", config_name="TDComparison")
def TD_comparison_experiment(cfg):
    """Compare convergence rate of PID-TD and PID-VI"""
    num_states = 50
    num_actions = 2
    env = ChainWalk(num_states, cfg['seed'])
    policy = np.zeros((num_states, num_actions))
    for i in range(num_states):
        policy[i,0] = 1

    PID_TDagent = ControlledTDLearning(
        env,
        policy,
        0.99,
        learning_rate_function(1, 0)
    )
    TDagent = ControlledTDLearning(
        env,
        policy,
        0.99,
        learning_rate_function(1, 0)
    )

    V_pi = find_Vpi(env, policy)
    test_function = lambda V, Vp, BR: np.linalg.norm(V - V_pi, 1)

    TD_history, td_rates = \
        find_optimal_pid_learning_rates(TDagent, 1, 0, 0, test_function, cfg['num_iterations'], False)
    save_array(TD_history, f"Regular TD {td_rates}", plt)
    for kp, kd, ki in zip(cfg['kp'], cfg['kd'], cfg['ki']):
        PID_TD_history, pid_td_rates = \
            find_optimal_pid_learning_rates(PID_TDagent, kp, kd, ki, test_function, cfg['num_iterations'], False)

        save_array(PID_TD_history, f"{kp=} {kd=} {ki=} {pid_td_rates}", plt)

    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('$||V_k - V^\pi||_1$')
    plt.savefig("plot")
    plt.show()

if __name__ == '__main__':
    TD_comparison_experiment()