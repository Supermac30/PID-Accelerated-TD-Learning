import matplotlib.pyplot as plt
import hydra

from Experiments.ExperimentHelpers import *
from Experiments.AgentBuilder import build_agent_and_env
from Experiments.HyperparameterTests import get_optimal_pid_rates

@hydra.main(version_base=None, config_path="../../config/TDExperiments", config_name="SoftTDPolicyEvaluation")
def soft_policy_evaluation_experiment(cfg):
    """Experiments with policy evaluation and TD"""
    for agent_name, kp, ki, kd, alpha, beta in zip(cfg['agent_name'], cfg['kp'], cfg['ki'], cfg['kd'], cfg['alpha'], cfg['beta']):
        if cfg['compute_optimal']:
            get_optimal_pid_rates(agent_name, cfg['env'], kp, ki, kd, alpha, beta, cfg['gamma'], cfg['recompute_optimal'])
        agent, env, policy = build_agent_and_env((agent_name, kp, ki, kd, alpha, beta), cfg['env'], cfg['get_optimal'], cfg['seed'], cfg['gamma'])
        V_pi = find_Vpi(env, policy, cfg['gamma'])
        test_function = build_test_function(cfg['norm'], V_pi)
        total_history = 0
        for _ in range(cfg['num_repeats']):
            history, _ = agent.estimate_value_function(
                num_iterations=cfg['num_iterations'],
                test_function=test_function,
                follow_trajectory=cfg['follow_trajectory'],
                stop_if_diverging=cfg['stop_if_diverging']
            )
            total_history += history
        total_history /= cfg['num_repeats']
        save_array(total_history, f"{agent_name} kp={kp} ki={ki} kd={kd} alpha={alpha} beta={beta}", plt)

    plt.title(f"PID-TD: {cfg['env']} gamma={cfg['gamma']}")
    plt.legend()
    plt.xlabel('Iteration')
    # Maybe code a function that returns this label to be used in all files?
    if type(cfg['norm']) == type("") and cfg['norm'][:4] == 'diff':
        index = cfg['norm'][5:]
        plt.ylabel(f"$V_k[{index}] - V^\pi[{index}]$")
        # Draw a dotted line at 0
        plt.axhline(y=0, color='k', linestyle='--')
    else:
        plt.ylabel(f"$||V_k - V^\pi||_{{{cfg['norm']}}}$")
    if cfg['log_plot']:
        plt.yscale('log')
    plt.savefig("plot")
    plt.show()

if __name__ == "__main__":
    soft_policy_evaluation_experiment()