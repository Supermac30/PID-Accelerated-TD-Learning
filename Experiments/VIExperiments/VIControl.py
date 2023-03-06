import matplotlib.pyplot as plt
import numpy as np
import hydra
import logging

from Environments import ChainWalk
from MDP import Control
from Experiments.ExperimentHelpers import *

@hydra.main(version_base=None, config_path="../../config/VIExperiments", config_name="VIControl")
def control_experiment(cfg):
    """Attempt to replicate results in figure 2 of PID Accelerated VI"""
    logger = logging.getLogger(__name__)
    set_seed(cfg['seed'], logger)

    num_states = 50
    num_actions = 2
    env = ChainWalk(num_states)
    agent = Control(
        env.num_states,
        env.num_actions,
        env.build_reward_matrix(),
        env.build_probability_transition_kernel(),
        0.99
    )

    V_star = find_Vstar(env)
    test_function=lambda V, Vp, BR: np.max(np.abs(V - V_star))

    for kp, kd, ki in zip(cfg['kp'], cfg['kd'], cfg['ki']):
        history = run_VI_experiment(agent, kp, kd, ki, test_function)
        save_array(history, f"{kp=} {kd=} {ki=}", plt)

    plt.legend()
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('$||V_k - V^*||$')

    plt.savefig("plot")
    plt.show()


if __name__ == "__main__":
    control_experiment()