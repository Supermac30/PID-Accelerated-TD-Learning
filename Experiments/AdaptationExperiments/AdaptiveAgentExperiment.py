"""Test the formulation of the adaptive agent without learning rates"""

import matplotlib.pyplot as plt
import hydra

from Experiments.AdaptiveAgentBuilder import build_adaptive_agent_and_env
from Experiments.AgentBuilder import build_agent_and_env
from Experiments.ExperimentHelpers import *

@hydra.main(version_base=None, config_path="../../config/AdaptationExperiments", config_name="AdaptiveAgentExperiment")
def adaptive_agent_experiment(cfg):
    """Visualize the behavior of adaptation without learning rates."""
    fig0 = plt.figure()
    ax0 = fig0.add_subplot()

    TDagent, env, policy = build_agent_and_env(
        ("TD", 1, 0, 0, 0, 0),
        cfg['env'],
        get_optimal=cfg['get_optimal'],
        seed=cfg['seed'],
        gamma=cfg['gamma']
    )
    V_pi = find_Vpi(env, policy, cfg['gamma'])
    test_function = build_test_function(cfg['norm'], V_pi)
    TDhistory, _ = TDagent.estimate_value_function(num_iterations=cfg['num_iterations'], test_function=test_function, follow_trajectory=cfg['follow_trajectory'])
    save_array(TDhistory, f"TD Agent", ax0)

    for agent_name, meta_lr, delay in zip(cfg['agent_name'], cfg['meta_lr'], cfg['delay']):
        agent, _, _ = build_adaptive_agent_and_env(
            agent_name,
            cfg['env'],
            meta_lr,
            get_optimal=cfg['get_optimal'],
            seed=cfg['seed'],
            gamma=cfg['gamma'],
            delay=delay,
            kp=cfg['kp'],
            ki=cfg['ki'],
            kd=cfg['kd'],
            alpha=cfg['alpha'],
            beta=cfg['beta']
        )
        fig1 = plt.figure()
        ax1 = fig1.add_subplot()

        _, gain_history, history = agent.estimate_value_function(cfg['num_iterations'], test_function, follow_trajectory=cfg['follow_trajectory'])

        save_array(history, f"Adaptive Agent: {agent_name} {meta_lr} {delay}", ax0)

        save_array(gain_history[:, 0], f"kp", ax1)
        save_array(gain_history[:, 1], f"ki", ax1)
        save_array(gain_history[:, 2], f"kd", ax1)
        save_array(gain_history[:, 3], f"alpha", ax1)
        save_array(gain_history[:, 4], f"beta", ax1)

        ax1.title.set_text(f"Adaptive Agent: {agent_name}, {cfg['env']}, delay={delay}, meta_lr={meta_lr}")
        ax1.legend()
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel(f'Gain Value')
        fig1.savefig(f"gains_plot {agent_name} {str(meta_lr).replace('.', '-')} {delay}")

    ax0.title.set_text(f"Adaptive Agent: {cfg['env']}")
    ax0.legend()
    ax0.set_xlabel('Iteration')
    ax0.set_ylabel(f'$||V_k - V^\pi||_{{{cfg["norm"]}}}$')
    fig0.savefig("history_plot")


if __name__ == '__main__':
    adaptive_agent_experiment()