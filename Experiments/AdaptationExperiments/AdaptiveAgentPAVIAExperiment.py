import matplotlib.pyplot as plt
import hydra

from Experiments.AdaptiveAgentBuilder import build_adaptive_agent
from Experiments.ExperimentHelpers import *

@hydra.main(version_base=None, config_path="../../config/AdaptationExperiments", config_name="AdaptiveAgentPAVIAExperiment")
def adaptive_agent_experiment(cfg):
    """Visualize the behavior of adaptation without learning rates."""
    env, policy = get_env_policy(cfg['env'], cfg['seed'])

    agent = build_adaptive_agent("planner", cfg['env'], env, policy, meta_lr_value=cfg['meta_lr'], gamma=cfg['gamma'])

    V_pi = find_Vpi(env, policy, cfg['gamma'])
    test_function = build_test_function(cfg['norm'], V_pi)

    _, gain_history, history = agent.estimate_value_function(cfg['num_iterations'], test_function)

    fig = plt.figure()
    ax = fig.add_subplot()

    save_array(history, "VI Gain Adaptation", ax)

    reward = env.build_policy_reward_vector(policy)
    transition = env.build_policy_probability_transition_kernel(policy)

    VIagent = PolicyEvaluation(env.num_states, env.num_actions, reward, transition, cfg['gamma'])
    VIhistory = run_VI_experiment(VIagent, 1, 0, 0, test_function, cfg['num_iterations'])
    save_array(VIhistory, f"VI Agent", ax)

    ax.title.set_text(f"Adaptive Agent: {cfg['env']}")
    ax.legend()
    ax.set_xlabel('Iteration')
    ax.set_yscale('log')
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

"""
Where is the circular import:
- It is in the file:
"""