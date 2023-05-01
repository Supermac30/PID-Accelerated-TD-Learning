import matplotlib.pyplot as plt
import numpy as np
import hydra

from Environments import ChainWalk
from MDP import PolicyEvaluation
from Experiments.ExperimentHelpers import *

@hydra.main(version_base=None, config_path="../../config/VIExperiments", config_name="VIPolicyEvaluation")
def policy_evaluation_experiment(cfg):
    """Attempt to replicate results in figure 1 of PID Accelerated VI"""
    env, policy = get_env_policy(cfg['env'], cfg['seed'])
    agent = PolicyEvaluation(
        env.num_states,
        env.num_actions,
        env.build_policy_reward_vector(policy),
        env.build_policy_probability_transition_kernel(policy),
        0.99
    )

    V_pi = find_Vpi(env, policy)

    test_function=build_test_function(cfg['norm'], V_pi)

    for kp, kd, ki in zip(cfg['kp'], cfg['kd'], cfg['ki']):
        history = run_VI_experiment(agent, kp, kd, ki, test_function, num_iterations=cfg['num_iterations'])
        save_array(history, f"kp={kp} kd={kd} ki={ki}", plt)


    plt.title(f"Policy Evaluation: {cfg['env']}")
    plt.legend()
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel(f"$||V_k - V^\pi||_{cfg['norm']}$")
    plt.savefig("plot")
    plt.show()

if __name__ == "__main__":
    policy_evaluation_experiment()