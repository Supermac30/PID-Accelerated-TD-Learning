import matplotlib.pyplot as plt
import hydra

from Experiments.ExperimentHelpers import *
from TabularPID.AgentBuilders.AgentBuilder import build_agent_and_env

@hydra.main(version_base=None, config_path="../../config/ComparisonExperiments", config_name="ConvergenceRateComparison")
def convergence_rate_VI_experiment(cfg):
    """Compare convergence rate of PID-TD and PID-VI"""
    seed = pick_seed(cfg['seed'])
    fig, (ax1, ax2) = plt.subplots(2)

    for kp, kd, ki, alpha, beta in zip(cfg['kp'], cfg['kd'], cfg['ki'], cfg['alpha'], cfg['beta']):
        TDagent, env, policy = build_agent_and_env((cfg['agent_description'], kp, ki, kd, alpha, beta), cfg['env'], cfg['get_optimal'], seed, cfg['gamma'])
        TD_history, _ = TDagent.estimate_value_function(num_iterations=cfg['num_iterations'], test_function=test_function, follow_trajectory=cfg['follow_trajectory'])

        V_pi = find_Vpi(env, policy, cfg['gamma'])
        test_function = build_test_function(cfg['norm'], V_pi)

        VIagent, env, policy = build_agent_and_env(("VI", kp, ki, kd, alpha, beta), cfg['env'], cfg['get_optimal'], seed, cfg['gamma'])
        VI_history = VIagent.estimate_value_function(num_iterations=cfg['num_iterations'], test_function=test_function, follow_trajectory=cfg['follow_trajectory'])

        save_array(TD_history, f"kp={kp} kd={kd} ki={ki} alpha={alpha} beta={beta}", ax2, cfg['normalize'])
        save_array(VI_history, f"kp={kp} kd={kd} ki={ki} alpha={alpha} beta={beta}", ax1, cfg['normalize'])

    plot_comparison(fig, ax1, ax2, f"PID Accelerated VI: {cfg['env']}", f"PID Accelerated TD: {cfg['env']}", f"$||V_k - V^\pi||_{cfg['norm']}$")

if __name__ == '__main__':
    convergence_rate_VI_experiment()