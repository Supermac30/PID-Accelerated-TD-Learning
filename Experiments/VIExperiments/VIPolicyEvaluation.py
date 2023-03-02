import matplotlib.pyplot as plt
import numpy as np
import hydra

from Environments import ChainWalk
from MDP import PolicyEvaluation
from Experiments.ExperimentHelpers import *

@hydra.main(version_base=None, config_path="../../config/VIExperiments", config_name="VIPolicyEvaluation")
def policy_evaluation_experiment(cfg):
    """Attempt to replicate results in figure 1 of PID Accelerated VI"""
    num_states = 50
    num_actions = 2
    env = ChainWalk(num_states)
    policy = np.zeros((num_states, num_actions))
    for i in range(num_states):
        policy[i,0] = 1
    agent = PolicyEvaluation(
        env.num_states,
        env.num_actions,
        env.build_policy_reward_vector(policy),
        env.build_policy_probability_transition_kernel(policy),
        0.99
    )

    V_pi = find_Vpi(env, policy)
    test_function=lambda V, Vp, BR: np.max(np.abs(V - V_pi))

    for kp, kd, ki in zip(cfg['kp'], cfg['kd'], cfg['ki']):
        history = run_VI_experiment(agent, kp, kd, ki, test_function)
        save_array(history, f"{kp=} {kd=} {ki=}", plt)


    plt.legend()
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('$||V_k - V^\pi||$')
    plt.savefig("plot")
    plt.show()

if __name__ == "__main__":
    policy_evaluation_experiment()