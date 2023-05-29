import matplotlib.pyplot as plt
import hydra

from Experiments.ExperimentHelpers import *
from Experiments.AgentBuilder import build_agent_and_env

@hydra.main(version_base=None, config_path="../../config/VIExperiments", config_name="VIQControl")
def control_experiment(cfg):
    """Attempt to replicate results in figure 2 of PID Accelerated VI"""
    seed = pick_seed(cfg['seed'])
    fig, ax = plt.subplots(1, 1)
    for kp, kd, ki, alpha, beta in zip(cfg['kp'], cfg['kd'], cfg['ki'], cfg['alpha'], cfg['beta']):
        agent, env, _ = build_agent_and_env(("VI Q control", kp, ki, kd, alpha, beta), cfg['env'], False, seed, cfg['gamma'])
        Q_star = find_Qstar(env, cfg['gamma'])
        history, _ = agent.value_iteration(num_iterations=cfg['num_iterations'], test_function=build_test_function(cfg['norm'], Q_star))
        save_array(history, f"kp={kp} kd={kd} ki={ki} alpha={alpha} beta={beta}", ax)

    ax.title.set_text(f"VI Control: {cfg['env']}")
    ax.legend()
    ax.set_yscale('log')
    ax.set_xlabel('Iteration')
    create_label(ax, cfg['norm'], cfg['normalize'], True)
    fig.savefig("plot")
    fig.show()


if __name__ == "__main__":
    control_experiment()