import matplotlib.pyplot as plt
import hydra

from Experiments.ExperimentHelpers import *
from Experiments.AgentBuilder import build_agent_and_env

@hydra.main(version_base=None, config_path="../../config/TDExperiments", config_name="HardTDPolicyEvaluation")
def policy_evaluation_experiment(cfg):
    """Experiments with policy evaluation and TD"""
    for kp, kd, ki, alpha, beta in zip(cfg['kp'], cfg['kd'], cfg['ki'], cfg['alpha'], cfg['beta']):
        agent, env, policy = build_agent_and_env(("hard TD", kp, ki, kd, alpha, beta), cfg['env'], cfg['get_optimal'], cfg['seed'], cfg['gamma'])
        V_pi = find_Vpi(env, policy, cfg['gamma'])
        test_function = build_test_function(cfg['norm'], V_pi)
        history, _ = agent.estimate_value_function(num_iterations=cfg['num_iterations'], test_function=test_function)
        save_array(history, f"kp={kp} kd={kd} ki={ki} alpha={alpha} beta={beta}", plt)


    plt.title(f"Hard TD Updates: {cfg['env']}")
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel(f"$||V_k - V^\pi||_{{{cfg['norm']}}}$")
    plt.savefig("plot")
    plt.show()


if __name__ == "__main__":
    policy_evaluation_experiment()