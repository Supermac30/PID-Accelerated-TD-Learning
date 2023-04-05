"""Test the formulation of the adaptive agent without learning rates"""

import matplotlib.pyplot as plt
import numpy as np
import hydra

from AdaptiveAgents import AdaptiveAgent
from Agents import ControlledTDLearning
from Experiments.ExperimentHelpers import *

@hydra.main(version_base=None, config_path="../../config/AdaptationExperiments", config_name="AdaptiveAgentExperiment")
def adaptive_agent_experiment(cfg):
    """Visualize the behavior of adaptation without learning rates."""
    seed = cfg['seed']
    if cfg['env'] == 'chain walk':
        env, policy = chain_walk_left(50, 2, seed)
    elif cfg['env'] == 'garnet':
        env, policy = PAVIA_garnet_settings(cfg['seed'])

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
    if cfg['norm'] == 'inf':
        test_function = lambda V, Vp, BR: np.max(np.abs(V - V_pi))
    else:
        test_function = lambda V, Vp, BR: np.linalg.norm(V - V_pi, cfg['norm'])

    _, history, gain_history = agent.estimate_value_function(cfg['num_iterations'], test_function)

    fig = plt.figure()
    ax = fig.add_subplot()

    save_array(history, f"Adaptive Agent", ax)

    if not cfg['planning']:
        TDagent = ControlledTDLearning(env, policy, 0.99, learning_rate_function(cfg['alpha_P'], cfg['N_P']))
        TDhistory = run_PID_TD_experiment(TDagent, 1, 0, 0, test_function, cfg['num_iterations'])
        save_array(TDhistory, f"TD Agent", ax)

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

    ax.legend()
    ax.set_xlabel('Iteration')
    ax.set_ylabel(f'Gain Value')
    fig.savefig("gains_plot")


if __name__ == '__main__':
    adaptive_agent_experiment()