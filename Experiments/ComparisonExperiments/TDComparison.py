"""Compare TD Learning with PID-Accelerated TD Learning"""

import matplotlib.pyplot as plt
import hydra

from Experiments.ExperimentHelpers import *
from Experiments.AgentBuilder import build_agent_and_env

@hydra.main(version_base=None, config_path="../../config/ComparisonExperiments", config_name="TDComparison")
def TD_comparison_experiment(cfg):
    """Compare convergence rate of PID-TD and Regular TD"""
    TDagent, env, policy = build_agent_and_env(("TD", 1, 0, 0, 0, 0), cfg['env'], cfg['get_optimal'], cfg['seed'], cfg['gamma'])
    V_pi = find_Vpi(env, policy, cfg['gamma'])
    test_function = build_test_function(cfg['norm'], V_pi)

    TD_history, _ = TDagent.estimate_value_function(num_iterations=cfg['num_iterations'], test_function=test_function, follow_trajectory=cfg['follow_trajectory'])
    save_array(TD_history, f"Regular TD", plt)
    for kp, kd, ki, alpha, beta in zip(cfg['kp'], cfg['kd'], cfg['ki'], cfg['alpha'], cfg['beta']):
        PID_TDagent, env, policy = build_agent_and_env(("TD", kp, ki, kd, alpha, beta), cfg['env'], cfg['get_optimal'], cfg['seed'], cfg['gamma'])
        PID_TD_history, _ = PID_TDagent.estimate_value_function(num_iterations=cfg['num_iterations'], test_function=test_function, follow_trajectory=cfg['follow_trajectory'])

        save_array(PID_TD_history, f"kp={kp} kd={kd} ki={ki} alpha={alpha} beta={beta}", plt)

    plt.title(f"TD Comparison: {cfg['env']}")
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel(f'$||V_k - V^\pi||_{cfg["norm"]}$')
    plt.savefig("plot")
    plt.show()

if __name__ == '__main__':
    TD_comparison_experiment()