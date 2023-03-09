import matplotlib.pyplot as plt
import numpy as np
import hydra

from Environments import ChainWalk
from Agents import ControlledTDLearning
from Experiments.ExperimentHelpers import *

@hydra.main(version_base=None, config_path="../../config/TDExperiments", config_name="AverageUpdateTheorem")
def find_average_update_experiment(cfg):
    """
    Experiment with what the expectation of

    \frac{\kappa_p}{\kappa_p - \kappa_d}\Bhat V_k - \frac{\kappa_d}{\kappa_p - \kappa_d}V_{p(k)}

    looks like. This experimentally verifies theorem 2.1 in the manuscript.
    """
    num_states = 50
    num_actions = 2
    gamma = 0.99
    env = ChainWalk(num_states, cfg['seed'])
    policy = np.zeros((num_states, num_actions))
    for i in range(num_states):
        policy[i,0] = 1
    agent = ControlledTDLearning(
        env,
        policy,
        gamma,
        learning_rate_function(cfg['alpha'], cfg['N'])
    )

    R = env.build_policy_reward_vector(policy)
    P = env.build_policy_probability_transition_kernel(policy)

    V_pi = find_Vpi(env, policy)

    def bellman(V):
        return R.reshape((-1, 1)) + gamma * P @ V

    for kp, kd in zip(cfg['kp'], cfg['kd']):
        test_function=lambda V, Vp, BR: np.max(np.abs(V_pi - bellman(V) * (kp / (kp - kd)) + Vp * (kd / (kp - kd))))
        history = run_TD_experiment(agent, kp, kd, 0, test_function)

        save_array(history, f"{kp=} {kd=}", plt)

    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('$||V^{\pi} - ((\kappa_p)/(\kappa_p - \kappa_d)T^{\pi} V_k - (\kappa_d)/(\kappa_p - \kappa_d) V_{p(k)})||_\infty$')
    plt.show()
    plt.savefig("plot")


if __name__ == "__main__":
    find_average_update_experiment()