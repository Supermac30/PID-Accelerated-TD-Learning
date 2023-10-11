"""Test the formulation of the adaptive agent without learning rates"""

import hydra
import matplotlib.pyplot as plt

from TabularPID.AgentBuilders.AdaptiveAgentBuilder import build_adaptive_agent_and_env
from TabularPID.OptimalRates.HyperparameterTests import get_optimal_adaptive_rates
from Experiments.ExperimentHelpers import *


def plot_gains(history, cfg):
    fig = plt.figure(figsize=(8, 4))
    gs = fig.add_gridspec(nrows=1, ncols=3, width_ratios=[1,1,1])
    titles = ["kp", "ki", "kd"]
    for i in range(3):
        gain_values = history[:,i]
        ax = fig.add_subplot(gs[0, i])
        ax.plot(gain_values)
        ax.set_xlabel('Steps')
        ax.title.set_text(titles[i])

    # Set square aspect ratio for each subplot
    for ax in fig.axes:
        ax.set_box_aspect(1)
        
    # Center the subplots in a single row and move up to remove whitespace
    gs.tight_layout(fig, rect=[0.05, 0.08, 0.95, 0.95])

    fig.savefig(f"{cfg['save_dir']}/gain_history.png")
    fig.savefig(f"{cfg['save_dir']}/gain_history.pdf")
    plt.close(fig)


@hydra.main(version_base=None, config_path="../../config/AdaptationExperiments", config_name="AdaptiveAgentExperiment")
def adaptive_agent_experiment(cfg):
    """Visualize the behavior of adaptation without learning rates."""
    seed = pick_seed(cfg['seed'])

    min_x = float("inf")
    max_x = float("-inf")
    min_y = float("inf")
    max_y = float("-inf")

    # Create a figure and axis
    fig, ax = plt.subplots()
    
    for run in range(2):
        agent_name, lambd, delay, alpha, beta, epsilon = cfg['agent_name'], cfg['lambda'], cfg['delay'], cfg['alpha'], cfg['beta'], cfg['epsilon']
        if run == 0:
            meta_lr = cfg['meta_lr']
        else:
            meta_lr = 0

        if cfg['compute_optimal']:
            get_optimal_adaptive_rates(agent_name, cfg['env'], meta_lr, cfg['gamma'], lambd, delay, alpha, beta, recompute=cfg['recompute_optimal'], epsilon=epsilon, search_steps=cfg['search_steps'])
        agent, env, policy = build_adaptive_agent_and_env(
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
            beta=beta,
            epsilon=epsilon
        )
        V_pi = find_Vpi(env, policy, cfg['gamma'])
        test_function = build_test_function(cfg['norm'], V_pi)

        agent.set_seed(seed)
        V_trajectory, gain_history = agent.estimate_value_function(
            cfg['num_iterations'],
            test_function,
            follow_trajectory=cfg['follow_trajectory'],
            stop_if_diverging=cfg['stop_if_diverging'],
            visualize=True
        )

        if run == 0:
            plot_gains(gain_history, cfg)
        
        ###### GPT generated code below ######

        # Extract x and y coordinates separately
        x = V_trajectory[:, 0]
        y = V_trajectory[:, 1]

        max_x = max(max_x, max(x))
        min_x = min(min_x, min(x))
        max_y = max(max_y, max(y))
        min_y = min(min_y, min(y))

        jump = 5
        # Plot the trajectory with arrows
        for i in range(0, len(V_trajectory) - jump, jump):
            color = 'red' if run == 0 else 'blue'
            ax.annotate('', xy=(x[i + jump], y[i + jump]), xytext=(x[i], y[i]), arrowprops={'arrowstyle': '->', 'color': color})

    ax.plot(x[0], y[0], 'go', label='Start')
    ax.plot(V_pi[0], V_pi[1], 'y*', markersize=12, label='End')

    max_x = max(max_x, V_pi[0], x[0])
    min_x = min(min_x, V_pi[0], x[0])
    max_y = max(max_y, V_pi[1], y[0])
    min_y = min(min_y, V_pi[1], y[0])

    # Set axis limits, labels, and legend
    ax.set_xlim(min_x - (max_x - min_x) * 0.1, max_x + (max_x - min_x) * 0.1)
    ax.set_ylim(min_y - (max_y - min_y) * 0.1, max_y + (max_y - min_y) * 0.1)
    ax.set_xlabel('V(0)')
    ax.set_ylabel('V(1)')
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color='red', lw=2, label='Gain Adaptation'),
            plt.Line2D([0], [0], color='blue', lw=2, label='TD Learning'),
            plt.Line2D([0], [0], marker='o', color='green', markersize=10, linestyle='', label='Initial Value'),
            plt.Line2D([0], [0], marker='*', color='yellow', markersize=10, linestyle='', label='Optimal Value')
        ],
        loc='upper left'
    )

    # Save the figure as a PNG and a PDF
    fig.savefig(f"{cfg['save_dir']}/trajectory.png")
    fig.savefig(f"{cfg['save_dir']}/trajectory.pdf")

if __name__ == '__main__':
    adaptive_agent_experiment()