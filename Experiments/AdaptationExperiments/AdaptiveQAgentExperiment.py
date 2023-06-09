"""Test the formulation of the adaptive agent without learning rates"""

import matplotlib.pyplot as plt

import hydra
import logging

from TabularPID.AgentBuilders.AdaptiveAgentBuilder import build_adaptive_agent_and_env
from TabularPID.AgentBuilders.AgentBuilder import build_agent_and_env
from TabularPID.OptimalRates.HyperparameterTests import get_optimal_pid_q_rates, get_optimal_adaptive_rates
from Experiments.ExperimentHelpers import *

@hydra.main(version_base=None, config_path="../../config/AdaptationExperiments", config_name="AdaptiveQAgentExperiment")
def adaptive_agent_experiment(cfg):
    """Visualize the behavior of adaptation without learning rates."""
    seed = pick_seed(cfg['seed'])
    fig0 = plt.figure()
    ax0 = fig0.add_subplot()

    if cfg['compute_optimal_Q']:
        get_optimal_pid_q_rates("Q learning", cfg['env'], 1, 0, 0, 0.05, 0.95, cfg['gamma'], cfg['recompute_optimal_Q'])

    Qagent, env, policy = build_agent_and_env(
        ("Q learning", 1, 0, 0, 0.05, 0.95, 1),
        cfg['env'],
        get_optimal=cfg['get_optimal_Q'],
        seed=seed,
        gamma=cfg['gamma']
    )
    Q_star = find_Qstar(env, cfg['gamma'])
    test_function = build_test_function(cfg['norm'], Q_star)
    average_history = np.zeros((cfg['num_iterations'],))
    for i in range(cfg['repeat']):
        Qhistory, V = Qagent.estimate_value_function(
            num_iterations=cfg['num_iterations'],
            test_function=test_function,
            follow_trajectory=cfg['follow_trajectory'],
            stop_if_diverging=cfg['stop_if_diverging'],
            reset_environment=False
        )
        average_history += Qhistory
    
    average_history /= cfg['repeat']
    save_array(average_history, f"Q-learner", ax0, normalize=cfg['normalize'])

    for meta_lr, delay, lambd, alpha, beta, epsilon in zip(cfg['meta_lr'], cfg['delay'], cfg['lambda'], cfg['alphas'], cfg['betas'], cfg['epsilon']):
        if cfg['compute_optimal']:
            get_optimal_adaptive_rates('semi gradient Q updater', cfg['env'], meta_lr, cfg['gamma'], lambd, delay, alpha, beta, recompute=cfg['recompute_optimal'], epsilon=epsilon, norm=cfg['norm'], is_q=True)
        agent, _, _ = build_adaptive_agent_and_env(
            "semi gradient Q updater",
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
            beta=beta,
            epsilon=epsilon
        )
        # Run the following agent.estimate_value_function 20 times and take an average of the histories
        average_history = np.zeros((cfg['num_iterations'],))
        for i in range(cfg['repeat']):
            Q, gain_history, history = agent.estimate_value_function(
                cfg['num_iterations'],
                test_function,
                follow_trajectory=cfg['follow_trajectory'],
                stop_if_diverging=cfg['stop_if_diverging'],
                reset_environment=False
            )
            average_history += history

        average_history /= cfg['repeat']

        logging.info(Q - Q_star)

        save_array(average_history, f"Adaptive Q Agent: {meta_lr} {delay} {lambd} {epsilon}", ax0, normalize=cfg['normalize'])

        fig = plt.figure(figsize=(10, 4))
        gs = fig.add_gridspec(nrows=1, ncols=3, width_ratios=[1,1,1], wspace=0.3, hspace=0.5)
        titles = ["kp", "ki", "kd"]

        for i in range(3):
            ax = fig.add_subplot(gs[0, i])
            save_array(gain_history[:, i], titles[i], ax)
            ax.set_xlabel('Iteration')
            ax.set_ylabel(titles[i])
            ax.legend()

        plt.suptitle(f"Adaptive Q Agent: {cfg['env']}, meta_lr={meta_lr}, epsilon={epsilon}")

        # Set square aspect ratio for each subplot
        for ax in fig.axes:
            ax.set_box_aspect(1)

        # Center the subplots in a single row and move up to remove whitespace
        gs.tight_layout(fig, rect=[0.05, 0.08, 0.95, 0.95])

        plt.savefig(f"gains_plot_{str(meta_lr).replace('.', '-')}_{delay}_{str(lambd).replace('.','-')}.png")
        plt.close()

        if cfg['plot_updater']:
            agent.plot()

    ax0.title.set_text(f"Adaptive Q Agent: {cfg['env']}")
    ax0.legend()
    ax0.set_xlabel('Iteration')
    create_label(ax0, cfg['norm'], cfg['normalize'], True)
    if cfg['log_plot']:
        ax0.set_yscale('log')
    fig0.savefig("history_plot")

    plt.show()



if __name__ == '__main__':
    adaptive_agent_experiment()