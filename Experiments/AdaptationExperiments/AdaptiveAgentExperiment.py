"""Test the formulation of the adaptive agent without learning rates"""

import matplotlib.pyplot as plt

import hydra

from Experiments.AdaptiveAgentBuilder import build_adaptive_agent_and_env
from Experiments.AgentBuilder import build_agent_and_env
from Experiments.ExperimentHelpers import *
from Experiments.HyperparameterTests import get_optimal_pid_rates, get_optimal_adaptive_rates
import logging

@hydra.main(version_base=None, config_path="../../config/AdaptationExperiments", config_name="AdaptiveAgentExperiment")
def adaptive_agent_experiment(cfg):
    """Visualize the behavior of adaptation without learning rates."""
    seed = pick_seed(cfg['seed'])
    fig0 = plt.figure()
    ax0 = fig0.add_subplot()

    if cfg['compute_optimal_TD']:
        get_optimal_pid_rates("TD", cfg['env'], 1, 0, 0, 0.05, 0.95, cfg['gamma'], cfg['recompute_optimal_TD'])

    TDagent, env, policy = build_agent_and_env(
        ("TD", 1, 0, 0, 0.05, 0.95),
        cfg['env'],
        get_optimal=cfg['get_optimal_TD'],
        seed=seed,
        gamma=cfg['gamma']
    )
    V_pi = find_Vpi(env, policy, cfg['gamma'])
    test_function = build_test_function(cfg['norm'], V_pi)
    average_history = np.zeros((cfg['num_iterations'],))
    for i in range(cfg['repeat']):
        TDhistory, V = TDagent.estimate_value_function(
            num_iterations=cfg['num_iterations'],
            test_function=test_function,
            follow_trajectory=cfg['follow_trajectory'],
            stop_if_diverging=cfg['stop_if_diverging']
        )
        average_history += TDhistory
    
    average_history /= cfg['repeat']
    save_array(average_history, f"TD Agent", ax0, normalize=cfg['normalize'])

    for agent_name, meta_lr, delay, lambd, alpha, beta in zip(cfg['agent_name'], cfg['meta_lr'], cfg['delay'], cfg['lambda'], cfg['alphas'], cfg['betas']):
        if cfg['compute_optimal']:
            get_optimal_adaptive_rates(agent_name, cfg['env'], meta_lr, cfg['gamma'], lambd, delay, alpha, beta, recompute=cfg['recompute_optimal'])
        agent, _, _ = build_adaptive_agent_and_env(
            agent_name,
            cfg['env'],
            meta_lr,
            lambd,
            delay,
            get_optimal=cfg['get_optimal'],
            seed=seed,
            gamma=cfg['gamma'],
            kp=cfg['kp'],
            ki=cfg['ki'],
            kd=cfg['kd'],
            alpha=alpha,
            beta=beta
        )
        # Run the following agent.estimate_value_function 20 times and take an average of the histories
        average_history = np.zeros((cfg['num_iterations'],))
        for i in range(cfg['repeat']):
            V, gain_history, history = agent.estimate_value_function(
                cfg['num_iterations'],
                test_function,
                follow_trajectory=cfg['follow_trajectory'],
                stop_if_diverging=cfg['stop_if_diverging']
            )
            average_history += history

        average_history /= cfg['repeat']

        logging.info(V - V_pi)

        save_array(average_history, f"Adaptive Agent: {agent_name} {meta_lr} {delay} {lambd}", ax0, normalize=cfg['normalize'])

        fig = plt.figure(figsize=(10, 4))
        gs = fig.add_gridspec(nrows=1, ncols=3, width_ratios=[1,1,1], wspace=0.3, hspace=0.5)
        titles = ["kp", "ki", "kd"]

        for i in range(3):
            ax = fig.add_subplot(gs[0, i])
            save_array(gain_history[:, i], titles[i], ax)
            ax.set_xlabel('Iteration')
            ax.set_ylabel(titles[i])
            ax.legend()

        plt.suptitle(f"Adaptive Agent: {agent_name}, {cfg['env']}, delay={delay}, meta_lr={meta_lr}, lambda={lambd}")

        # Set square aspect ratio for each subplot
        for ax in fig.axes:
            ax.set_box_aspect(1)

        # Center the subplots in a single row and move up to remove whitespace
        gs.tight_layout(fig, rect=[0.05, 0.08, 0.95, 0.95])

        plt.savefig(f"gains_plot_{agent_name}_{str(meta_lr).replace('.', '-')}_{delay}_{str(lambd).replace('.','-')}.png")
        plt.close()

        if cfg['plot_updater']:
            agent.plot()

    ax0.title.set_text(f"Adaptive Agent: {cfg['env']}")
    ax0.legend()
    ax0.set_xlabel('Iteration')
    create_label(ax0, cfg['norm'], cfg['normalize'], False)
    if cfg['log_plot']:
        ax0.set_yscale('log')
    fig0.savefig("history_plot")

    plt.show()



if __name__ == '__main__':
    adaptive_agent_experiment()