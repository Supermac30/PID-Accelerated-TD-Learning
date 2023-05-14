import matplotlib.pyplot as plt
import hydra

from Experiments.ExperimentHelpers import *
from Experiments.AgentBuilder import build_agent_and_env

@hydra.main(version_base=None, config_path="../../config/VIExperiments", config_name="VIControl")
def control_experiment(cfg):
    """Attempt to replicate results in figure 2 of PID Accelerated VI"""

    for kp, kd, ki, alpha, beta in zip(cfg['kp'], cfg['kd'], cfg['ki'], cfg['alpha'], cfg['beta']):
        agent, env, _ = build_agent_and_env(("VI Q control", kp, ki, kd, alpha, beta), cfg['env'], False, cfg['seed'], cfg['gamma'])
        Q_star = find_Qstar(env, cfg['gamma'])
        history, _ = agent.value_iteration(num_iterations=cfg['num_iterations'], test_function=build_test_function(cfg['norm'], Q_star))
        save_array(history, f"kp={kp} kd={kd} ki={ki} alpha={alpha} beta={beta}", plt)

    plt.title(f"VI Control: {cfg['env']}")
    plt.legend()
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel(f"$||Q_k - Q^*||_{{{cfg['norm']}}}$")
    plt.savefig("plot")
    plt.show()


if __name__ == "__main__":
    control_experiment()