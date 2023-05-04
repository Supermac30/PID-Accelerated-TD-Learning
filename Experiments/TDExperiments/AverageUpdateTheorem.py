import matplotlib.pyplot as plt
import numpy as np
import hydra

from AgentBuilder import build_agent_and_env
from Experiments.ExperimentHelpers import *

@hydra.main(version_base=None, config_path="../../config/TDExperiments", config_name="AverageUpdateTheorem")
def find_average_update_experiment(cfg):
    """
    Experiment with what the expectation of

    \frac{\kappa_p}{\kappa_p - \kappa_d}\Bhat V_k - \frac{\kappa_d}{\kappa_p - \kappa_d}V_{p(k)}

    looks like. This experimentally verifies theorem 2.1 in the manuscript.
    """

    for kp, kd in zip(cfg['kp'], cfg['kd']):
        agent, env, policy = build_agent_and_env(("TD", kp, 0, kd, 0.05, 0.95), cfg['env'], False, cfg['seed'], cfg['gamma'])
        R = env.get_reward_matrix(policy)
        P = env.get_transition_matrix(policy)
        def bellman(V):
            return R.reshape((-1, 1)) + cfg['gamma'] * P @ V

        V_pi = find_Vpi(env, policy)
        test_function=lambda V, Vp, BR: np.max(np.abs(V_pi - bellman(V) * (kp / (kp - kd)) + Vp * (kd / (kp - kd))))
        history, _ = agent.estimate_value_function(num_iterations=cfg['num_iterations'], test_function=test_function)

        save_array(history, f"{kp=} {kd=}", plt)

    plt.title(f"Average Update: {cfg['env']}")
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('$||V^{\pi} - ((\kappa_p)/(\kappa_p - \kappa_d)T^{\pi} V_k - (\kappa_d)/(\kappa_p - \kappa_d) V_{p(k)})||_\infty$')
    plt.savefig("plot")
    plt.show()


if __name__ == "__main__":
    find_average_update_experiment()