import numpy as np
import matplotlib.pyplot as plt
from Controllers import P_Controller, D_Controller, I_Controller
from MDP import PolicyEvaluation, Control

import logging
import os


# def create_env_policy()


def learning_rate_function(alpha, N):
    """Return the learning rate function alpha(k) parameterized by alpha and N"""
    return lambda k: min(alpha, N/(k + 1))


def find_optimal_learning_rates(agent, value_function_estimator, isSoft):
    """Run a grid search for values of N and alpha that makes the
    value_function_estimator reach an error of at most the threshold as fast as possible.

    agent should be an Agent object.
    value_function_estimator should be a function that runs agent.estimate_value_function with
        the correct parameters when called with a threshold, and does not plot

    isSoft is True when we update the update_rate as well.

    Return the optimal parameters and the associated history
    """
    # A dictionary from alpha to possible Ns
    learning_rates = {
        0.2: {10, 100},
        0.15: {10, 100},
        0.09: {10, 100, 1000, 10000},
        0.05: {10, 100, 1000, 10000},
        0.01: {1000, 10000},
        0.001: {1000, 10000},
        0.0001: {1000, 10000}
    }

    # Store for later restoration to avoid spooky action at a distance
    original_learning_rate = agent.learning_rate

    minimum_length = float('inf')
    minimum_params = None
    minimum_history = None

    for alpha in learning_rates:
        for N in learning_rates[alpha]:
            agent.learning_rate = learning_rate_function(alpha, N)
            if isSoft:
                # For now, the update and learning rates are the same in grid search
                # We should try separate learning rates for more extensive experiments later.
                agent.update_rate = learning_rate_function(alpha, N)
            history = value_function_estimator()
            if history.shape[0] < minimum_length or (history.shape[0] == minimum_length and history[-1] < minimum_history[-1]):
                minimum_length = history.shape[0]
                minimum_params = (N, alpha)
                minimum_history = history


    # Restore original learning rate
    agent.learning_rate = original_learning_rate

    return minimum_history, minimum_params


def find_optimal_pid_learning_rates(agent, kp, kd, ki, test_function, num_iterations, threshold, isSoft):
    """Runs the find_optimal_learning_rates function for a agent that uses a PID controller."""

    return find_optimal_learning_rates(
        agent,
        lambda: run_TD_experiment(agent, kp, kd, ki, test_function, num_iterations, threshold),
        isSoft
    )


def run_TD_experiment(agent, kp, kd, ki, test_function, num_iterations=5000, threshold=None):
    """Have the agent estimate the value function using some choice of control gains,
    and graph the value of test_function during training.

    test_function should be a function that takes in V, Vp, BR and returns a real number.
    If graph is None, nothing is plotted.

    Return the average history of test_function during training.
    """
    p_controller = P_Controller(kp * np.identity(agent.num_states))
    d_controller = D_Controller(kd * np.identity(agent.num_states))
    i_controller = I_Controller(0.05, 0.95, ki * np.identity(agent.num_states))

    average_history = [0]
    min_size = float('inf')
    for _ in range(10):
        history, V = agent.estimate_value_function(
            p_controller,
            d_controller,
            i_controller,
            test_function=test_function,
            num_iterations=num_iterations,
            threshold=threshold
        )

        # To deal with different sizes arrays, we sum the common indices only
        min_size = min(min_size, history.shape[0])
        average_history = average_history[:min_size] + history[:min_size]
    average_history /= 10

    return average_history


def run_VI_experiment(agent, kp, kd, ki, test_function, num_iterations=500):
    """Have the agent estimate the value function using some choice of control gains,
    and graph the value of test_function during training.

    test_function should be a function that takes in V, Vp, BR and returns a real number.
    If graph is None, nothing is plotted.

    Return the history of test_function during training.
    """
    p_controller = P_Controller(kp * np.identity(agent.num_states))
    d_controller = D_Controller(kd * np.identity(agent.num_states))
    i_controller = I_Controller(0.05, 0.95, ki * np.identity(agent.num_states))
    history, V = agent.value_iteration(
        p_controller,
        d_controller,
        i_controller,
        test_function=test_function,
        num_iterations=num_iterations
    )

    return history


def find_Vpi(env, policy):
    """Find a good approximation of the value function of policy in an environment.
    """
    oracle = PolicyEvaluation(
        env.num_states,
        env.num_actions,
        env.build_policy_reward_vector(policy),
        env.build_policy_probability_transition_kernel(policy),
        0.99
    )

    p_controller = P_Controller(np.identity(env.num_states))
    return oracle.value_iteration(p_controller, num_iterations=10000)


def find_Vstar(env):
    """Find a good approximation of the value function of the optimal policy in an environment.
    """
    oracle = Control(
        env.num_states,
        env.num_actions,
        env.build_reward_matrix(),
        env.build_probability_transition_kernel(),
        0.99
    )

    p_controller = P_Controller(np.identity(env.num_states))
    return oracle.value_iteration(p_controller, num_iterations=10000)


def save_array(nparr, name, graph=None):
    """Save nparr in a file with name name. If graph is not None, this is plotted on graph.
    Creates the npy and txt files if they don't exist to store the numpy arrays.
    """
    if not os.path.isdir('npy'):
        os.mkdir('npy')
    if not os.path.isdir('txt'):
        os.mkdir('txt')

    np.save("npy/" + name + ".npy", nparr)
    np.savetxt("txt/" + name + ".txt", nparr)

    if graph is not None:
        graph.plot(nparr, label=name)


def plot_comparison(fig, ax1, ax2, title1, title2, ylabel, show_fig=True, fig_name="plot"):
    """Configure and plot a comparison between the learning
    of two algorithms given pyplot objects"""
    plt.subplots_adjust(hspace=0.7)
    fig.set_figheight(10)
    fig.set_figwidth(10)

    ax1.set_title(title1)
    ax2.set_title(title2)

    ax1.legend()
    ax2.legend()
    ax1.set(xlabel='Iteration', ylabel=ylabel)
    ax2.set(xlabel='Iteration', ylabel=ylabel)

    fig.savefig(fig_name)
    if show_fig: plt.show()