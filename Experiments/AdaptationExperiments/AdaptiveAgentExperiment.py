"""Test the formulation of the adaptive agent without learning rates"""

import matplotlib.pyplot as plt
import numpy as np
import hydra

from AdaptiveAgents import AdaptiveAgent
from Agents import ControlledTDLearning
from MDP import PolicyEvaluation
from Experiments.ExperimentHelpers import *

@hydra.main(version_base=None, config_path="../../config/AdaptationExperiments", config_name="AdaptiveAgentExperiment")
def adaptive_agent_experiment(cfg):
    """Visualize the behavior of adaptation without learning rates."""
    env, policy = get_env_policy(cfg['env'], cfg['seed'])

    if cfg['debug']:
        transitions = env.build_policy_probability_transition_kernel(policy)
        rewards = env.build_policy_reward_vector(policy)
    else:
        transitions = None
        rewards = None

    agent = AdaptiveAgent(
        (learning_rate_function(cfg['alpha_P'], cfg['N_P']),
         learning_rate_function(cfg['alpha_I'], cfg['N_I']),
         learning_rate_function(cfg['alpha_D'], cfg['N_D'])
        ),
        cfg['meta_lr'],
        env,
        policy,
        0.99,
        cfg['sample_size'],
        transitions,
        rewards,
        cfg['planning']
    )

    V_pi = find_Vpi(env, policy)
    test_function = build_test_function(cfg['norm'], V_pi)

    _, history, gain_history = agent.estimate_value_function(cfg['num_iterations'], test_function)

    fig = plt.figure()
    ax = fig.add_subplot()

    save_array(history, f"Adaptive Agent", ax)

    if cfg['planning']:
        VIAgent = PolicyEvaluation(
            env.num_states,
            env.num_actions,
            env.build_policy_reward_vector(policy),
            env.build_policy_probability_transition_kernel(policy),
            0.99
        )
        p_controller = P_Controller(np.identity(env.num_states))
        VIHistory, _ = VIAgent.value_iteration(p_controller, num_iterations=cfg['num_iterations'], test_function=test_function)
        save_array(VIHistory, f"VI Agent", ax)

        ax.set_
    else:
        TDagent = ControlledTDLearning(env, policy, 0.99, learning_rate_function(cfg['alpha_P'], cfg['N_P']))
        TDhistory = run_PID_TD_experiment(TDagent, 1, 0, 0, test_function, cfg['num_iterations'])
        save_array(TDhistory, f"TD Agent", ax)

    ax.title(f"Adaptive Agent: {cfg['env']}")
    ax.legend()
    ax.set_xlabel('Iteration')
    ax.set_ylabel(f'$||V_k - V^\pi||_{cfg["norm"]}$')
    fig.savefig("history_plot")

    fig = plt.figure()
    ax = fig.add_subplot()

    save_array(gain_history[:, 0], f"kp", ax)
    save_array(gain_history[:, 1], f"ki", ax)
    save_array(gain_history[:, 2], f"kd", ax)
    save_array(gain_history[:, 3], f"alpha", ax)
    save_array(gain_history[:, 4], f"beta", ax)

    ax.title(f"Adaptive Agent: {cfg['env']}")
    ax.legend()
    ax.set_xlabel('Iteration')
    ax.set_ylabel(f'Gain Value')
    fig.savefig("gains_plot")


if __name__ == '__main__':
    adaptive_agent_experiment()