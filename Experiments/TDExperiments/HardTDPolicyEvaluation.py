import matplotlib.pyplot as plt
import numpy as np
import hydra
import logging

from Environments import ChainWalk
from Agents import ControlledTDLearning
from Experiments.ExperimentHelpers import *

@hydra.main(version_base=None, config_path="../../config/TDExperiments", config_name="HardTDPolicyEvaluation")
def policy_evaluation_experiment(cfg):
    """Experiments with policy evaluation and TD"""
    logger = logging.getLogger(__name__)
    set_seed(cfg['seed'], logger)

    num_states = 50
    num_actions = 2
    env = ChainWalk(num_states)
    policy = np.zeros((num_states, num_actions))
    for i in range(num_states):
        policy[i,0] = 1
    agent = ControlledTDLearning(
        env,
        policy,
        0.99,
        learning_rate_function(cfg['alpha'], cfg['N'])
    )

    V_pi = find_Vpi(env, policy)
    test_function=lambda V, Vp, BR: np.max(np.abs(V - V_pi))

    for kp, kd, ki in zip(cfg['kp'], cfg['kd'], cfg['ki']):
        history = run_TD_experiment(agent, kp, kd, ki, test_function)
        save_array(history, f"{kp=} {kd=} {ki=}", plt)


    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('$||V_k - V^\pi||_\infty$')
    plt.show()
    plt.savefig("plot")


if __name__ == "__main__":
    policy_evaluation_experiment()