import matplotlib.pyplot as plt
import hydra

from AdaptiveAgentBuilder import build_adaptive_agent_and_env
from AgentBuilder import build_agent_and_env
from Experiments.ExperimentHelpers import *

@hydra.main(version_base=None, config_path="../../config/AdaptationExperiments", config_name="AdaptiveAgentPAVIAExperiment")
def adaptive_agent_experiment(cfg):
    """Visualize the behavior of adaptation without learning rates."""
    seed = pick_seed(cfg['seed'])
    agent, env, policy = build_adaptive_agent_and_env(cfg['agent_name'], cfg['env'], cfg['meta_lr'], 0, 1, seed=seed, gamma=cfg['gamma'])

    V_pi = find_Vpi(env, policy, cfg['gamma'])
    test_function = build_test_function(cfg['norm'], V_pi)

    _, gain_history, history = agent.estimate_value_function(cfg['num_iterations'], test_function, follow_trajectory=cfg['follow_trajectory'])
    agent.gain_updater.plot()

    fig = plt.figure()
    ax = fig.add_subplot()

    save_array(history, "VI Gain Adaptation", ax)

    reward = env.build_policy_reward_vector(policy)
    transition = env.build_policy_probability_transition_kernel(policy)

    VIagent, env, policy = build_agent_and_env(("VI", 1, 0, 0, 0, 0), cfg['env'], gamma=cfg['gamma'], seed=seed)
    VIhistory, _ = VIagent.value_iteration(num_iterations=cfg['num_iterations'], test_function=test_function)
    save_array(VIhistory, f"VI Agent", ax)

    ax.title.set_text(f"Adaptive Agent: {cfg['env']}")
    ax.legend()
    ax.set_xlabel('Iteration')
    if cfg['log_plot']:
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
    create_label(ax, cfg['norm'], cfg['normalize'], False)
    fig.savefig("gains_plot")


if __name__ == '__main__':
    adaptive_agent_experiment()