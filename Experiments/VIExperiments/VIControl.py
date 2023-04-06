import matplotlib.pyplot as plt
import numpy as np
import hydra

from Environments import ChainWalk
from MDP import Control
from Experiments.ExperimentHelpers import *

@hydra.main(version_base=None, config_path="../../config/VIExperiments", config_name="VIControl")
def control_experiment(cfg):
    """Attempt to replicate results in figure 2 of PID Accelerated VI"""
    env, _ = get_env_policy(cfg['env'], cfg['seed'])
    agent = Control(
        env.num_states,
        env.num_actions,
        env.build_reward_matrix(),
        env.build_probability_transition_kernel(),
        0.99
    )

    V_star = find_Vstar(env)
    test_function=build_test_function(cfg['norm'], V_star)

    for kp, kd, ki in zip(cfg['kp'], cfg['kd'], cfg['ki']):
        history = run_VI_experiment(agent, kp, kd, ki, test_function, num_iterations=cfg['num_iterations'])
        save_array(history, f"{kp=} {kd=} {ki=}", plt)

    plt.title(f"VI Control: {cfg['env']}")
    plt.legend()
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel(f"$||V_k - V^*||_{cfg['norm']}$")
    plt.savefig("plot")
    plt.show()


if __name__ == "__main__":
    control_experiment()