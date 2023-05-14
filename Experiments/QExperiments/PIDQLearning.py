import matplotlib.pyplot as plt
import hydra

from Experiments.ExperimentHelpers import *
from Experiments.AgentBuilder import build_agent_and_env
from Experiments.HyperparameterTests import get_optimal_pid_rates

@hydra.main(version_base=None, config_path="../../config/QExperiments", config_name="PIDQLearning")
def soft_policy_evaluation_experiment(cfg):
    """Experiments with policy evaluation and TD"""
    for agent_name, kp, ki, kd, alpha, beta in zip(cfg['agent_name'], cfg['kp'], cfg['ki'], cfg['kd'], cfg['alpha'], cfg['beta']):
        if cfg['compute_optimal']:
            get_optimal_pid_rates(agent_name, cfg['env'], kp, ki, kd, alpha, beta, cfg['gamma'], cfg['recompute_optimal'])
        agent, env, policy = build_agent_and_env((agent_name, kp, ki, kd, alpha, beta), cfg['env'], cfg['get_optimal'], cfg['seed'], cfg['gamma'])
        Q_star = find_Qstar(env, cfg['gamma'])
        test_function = build_test_function(cfg['norm'], Q_star)
        total_history = 0
        for _ in range(cfg['num_repeats']):
            history, _ = agent.estimate_value_function(num_iterations=cfg['num_iterations'], test_function=test_function, follow_trajectory=cfg['follow_trajectory'])
            total_history += history
        total_history /= cfg['num_repeats']
        save_array(total_history, f"{agent_name} kp={kp} ki={ki} kd={kd} alpha={alpha} beta={beta}", plt)

    plt.title(f"Q Learning: {cfg['env']} gamma={cfg['gamma']}")
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel(f"$||V_k - V^\pi||_{{{cfg['norm']}}}$")
    plt.savefig("plot")
    plt.show()

if __name__ == "__main__":
    soft_policy_evaluation_experiment()