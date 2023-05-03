import matplotlib.pyplot as plt
import numpy as np
import hydra

from Environments import ChainWalk
from Agents import Hard_PID_TD
from Experiments.ExperimentHelpers import *

@hydra.main(version_base=None, config_path="../../config/TDExperiments", config_name="AverageUpdateTheorem")
def find_average_update_experiment(cfg):
    """
    Experiment with what the expectation of

    \frac{\kappa_p}{\kappa_p - \kappa_d}\Bhat V_k - \frac{\kappa_d}{\kappa_p - \kappa_d}V_{p(k)}

    looks like. This experimentally verifies theorem 2.1 in the manuscript.
    """
    gamma = 0.99
    env, policy = get_env_policy(cfg['env'], cfg['seed'])
    agent = Hard_PID_TD(
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
        history = run_PID_TD_experiment(agent, kp, kd, 0, test_function)

        save_array(history, f"{kp=} {kd=}", plt)

    plt.title(f"Average Update: {cfg['env']}")
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('$||V^{\pi} - ((\kappa_p)/(\kappa_p - \kappa_d)T^{\pi} V_k - (\kappa_d)/(\kappa_p - \kappa_d) V_{p(k)})||_\infty$')
    plt.savefig("plot")
    plt.show()


if __name__ == "__main__":
    find_average_update_experiment()