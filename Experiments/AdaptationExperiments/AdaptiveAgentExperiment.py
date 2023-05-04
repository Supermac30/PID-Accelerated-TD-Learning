"""Test the formulation of the adaptive agent without learning rates"""

import matplotlib.pyplot as plt
import hydra

from Experiments.AdaptiveAgentBuilder import build_adaptive_agent_and_env
from Experiments.AgentBuilder import build_agent_and_env
from Experiments.ExperimentHelpers import *

@hydra.main(version_base=None, config_path="../../config/AdaptationExperiments", config_name="AdaptiveAgentExperiment")
def adaptive_agent_experiment(cfg):
    """Visualize the behavior of adaptation without learning rates."""
    agent, env, policy = build_adaptive_agent_and_env(cfg['agent_name'], cfg['env'], cfg['get_optimal'], meta_lr_value=cfg['meta_lr'], seed=cfg['seed'], gamma=cfg['gamma'])

    V_pi = find_Vpi(env, policy)
    test_function = build_test_function(cfg['norm'], V_pi)

    _, gain_history, history = agent.estimate_value_function(cfg['num_iterations'], test_function)

    fig = plt.figure()
    ax = fig.add_subplot()

    save_array(history, f"Adaptive Agent: {cfg['agent_name']}", ax)

    TDagent, _, _ = build_agent_and_env(("TD", 1, 0, 0, 0, 0), cfg['env'], seed=cfg['seed'], gamma=cfg['gamma'])
    TDhistory, _ = TDagent.estimate_value_function(num_iterations=cfg['num_iterations'], test_function=test_function)
    save_array(TDhistory, f"TD Agent", ax)

    ax.title.set_text(f"Adaptive Agent: {cfg['env']}")
    ax.legend()
    ax.set_xlabel('Iteration')
    ax.set_ylabel(f'$||V_k - V^\pi||_{{{cfg["norm"]}}}$')
    fig.savefig("history_plot")

    fig = plt.figure()
    ax = fig.add_subplot()

    save_array(gain_history[:, 0], f"kp", ax)
    save_array(gain_history[:, 1], f"ki", ax)
    save_array(gain_history[:, 2], f"kd", ax)
    save_array(gain_history[:, 3], f"alpha", ax)
    save_array(gain_history[:, 4], f"beta", ax)

    ax.title.set_text(f"Adaptive Agent: {cfg['env']}")
    ax.legend()
    ax.set_xlabel('Iteration')
    ax.set_ylabel(f'Gain Value')
    fig.savefig("gains_plot")


if __name__ == '__main__':
    adaptive_agent_experiment()