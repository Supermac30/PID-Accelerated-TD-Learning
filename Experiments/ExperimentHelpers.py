import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
import logging

from MDP import PolicyEvaluation, Control, Control_Q
from Environments import ChainWalk, Garnet, CliffWalk, IdentityEnv

def build_test_function(norm, V_pi):
    """Return the function that tests how far away our current estimate is from V_pi
    in the format that the Agent class expects.
    """
    if norm == "inf":
        return lambda V, Vp, BR: np.max(np.abs(V - V_pi))
    else:
        return lambda V, Vp, BR: np.linalg.norm(V - V_pi, norm)

def get_env_policy(name, seed):
    """Return an environment, policy tuple given its string name as input. A seed is inputted as well
    for reproducibility.
    The environments are as follows:
        - "garnet": The garnet problem with the default settings in PAVIA
        - "chain walk": The chain walk problem with 50 states, and a policy that always moves left
        - "chain walk random": The chain walk problem with 50 states, and a policy that takes random choices
        - "cliff walk": The Cliff walk problem as implemented in OS-Dyna
    """
    if name[:6] == "garnet":
        # The name is of the form garnet <seed> <num_states>
        seed, num_states = map(int, name[7:].split(" "))
        return garnet_problem(num_states, 3, 5, 10, seed)
    elif name == "chain walk":
        return chain_walk_left(50, 2, seed)
    elif name == "chain walk random":
        return chain_walk_random(50, 2, seed)
    elif name == "cliff walk":
        return cliff_walk(seed)
    elif name == "identity":
        return identity(seed)
    else:
        raise Exception("Environment not indexed")

def cliff_walk(seed):
    """Return the CliffWalk Environment with the policy that moves randomly"""
    env = CliffWalk(0.9, seed)
    policy = np.zeros((env.num_states, env.num_actions))
    for i in range(env.num_states):
        policy[i, :] = 1 / env.num_actions

    return env, policy


def garnet_problem(num_states, num_actions, bP, bR, seed):
    """Return the Garnet Environment with the policy that chooses the first move at each iteration"""
    env = Garnet(
        num_states, num_actions, bP, bR, seed
    )
    policy = np.zeros((num_states, num_actions))
    for i in range(num_states):
        policy[i, :] = 1/num_actions

    return env, policy


def PAVIA_garnet_settings(seed):
    """Return Garnet used in the experiments of PAVIA."""
    return garnet_problem(50, 1, 3, 5, seed)


def chain_walk_random(num_states, num_actions, seed):
    """Return the chain walk environment with the policy that takes random moves."""
    env = ChainWalk(num_states, seed)
    policy = np.zeros((num_states, num_actions))
    for i in range(num_states):
        policy[i,:] = 1/num_actions
    return env, policy


def chain_walk_left(num_states, num_actions, seed):
    """Return the chain walk environment with the policy that always moves left."""
    env = ChainWalk(num_states, seed)
    policy = np.zeros((num_states, num_actions))
    for i in range(num_states):
        policy[i,0] = 1
        policy[i,1] = 0
    return env, policy

def identity(seed):
    """Return the identity environment the policy that picks the only available action."""
    env = IdentityEnv(1, seed)
    policy = np.zeros((1, 1))
    for i in range(env.num_states):
        policy[i, 0] = 1
    return env, policy


def learning_rate_function(alpha, N):
    """Return the learning rate function alpha(k) parameterized by alpha and N.
    If N is infinity, return a constant function that outputs alpha.
    """
    if N == 'inf':
        return lambda k: alpha
    return lambda k: min(alpha, N/(k + 1))


def find_optimal_learning_rates(agent, value_function_estimator, learning_rates={}, update_I_rates={}, update_D_rates={}, verbose=False):
    """Run a grid search for values of N and alpha that makes the
    value_function_estimator have the lowest possible error.

    agent should be an Agent object.
    value_function_estimator should be a function that runs agent.estimate_value_function with
        the correct parameters, and does not plot

    Return the optimal parameters and the associated history
    """
    # Store for later restoration to avoid spooky action at a distance
    original_learning_rate = agent.learning_rate

    minimum_params = None
    minimum_history = [float('inf')]

    def try_params(params):
        """Run the current value function with the parameters set, and return
        the optimal length, optimal params, and optimal history.
        params: an object representing the parameters we initialized the value_function_estimator to
        """
        if verbose:
            logging.info(f"trying {params}")
        history = value_function_estimator()
        if verbose:
            logging.info(f"final value: {history[-1]}")
        if history[-1] < minimum_history[-1]:
            return params, history
        return minimum_params, minimum_history

    for alpha, beta, gamma in itertools.product(learning_rates, update_I_rates, update_D_rates):
        for N, M, L in itertools.product(learning_rates[alpha], update_I_rates[beta], update_D_rates[gamma]):
            agent.learning_rate = learning_rate_function(alpha, N)
            agent.update_I_rate = learning_rate_function(beta, M)
            agent.update_D_rate = learning_rate_function(gamma, L)
            minimum_params, minimum_history = try_params((alpha, N, beta, M, gamma, L))

    # Restore original learning rate
    agent.learning_rate = original_learning_rate

    return minimum_history, minimum_params


def repeat_experiment(agent, num_times, **kwargs):
    """Run the experiment num_times times and return the average history.
    Take as input the parameters to the estimate_value_function function.
    """
    average_history = 0

    for _ in range(num_times):
        history = agent.estimate_value_function(kwargs)
        average_history += np.array(history)

    average_history /= num_times

    return average_history


def find_Vpi(env, policy, gamma=0.99):
    """Find a good approximation of the value function of policy in an environment.
    """
    oracle = PolicyEvaluation(
        env.num_states,
        env.num_actions,
        env.build_policy_reward_vector(policy),
        env.build_policy_probability_transition_kernel(policy),
        1,0,0,0,0,
        gamma
    )
    return oracle.value_iteration(num_iterations=10000)


def find_Vstar(env, gamma=0.99):
    """Find a good approximation of the value function of the optimal policy in an environment.
    """
    oracle = Control_Q(
        env.num_states,
        env.num_actions,
        env.build_reward_matrix(),
        env.build_probability_transition_kernel(),
        1,0,0,0,0,
        gamma
    )
    return oracle.value_iteration(num_iterations=10000)


def find_Qstar(env, gamma=0.99):
    """Find a good approximation of the value function of the optimal policy in an environment.
    """
    oracle = Control(
        env.num_states,
        env.num_actions,
        env.build_reward_matrix(),
        env.build_probability_transition_kernel(),
        1,0,0,0,0,
        gamma
    )
    return oracle.value_iteration(num_iterations=10000)


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


def plot_comparison(fig, ax1, ax2, title1, title2, ylabel, show_fig=True, fig_name="plot", is_log=False):
    """Configure and plot a comparison between the learning
    of two algorithms given pyplot objects"""
    plt.subplots_adjust(hspace=0.7)
    fig.set_figheight(10)
    fig.set_figwidth(10)

    ax1.set_title(title1)
    ax2.set_title(title2)

    if is_log:
        ax1.set_yaxis('log')

    ax1.legend()
    ax2.legend()
    ax1.set(xlabel='Iteration', ylabel=ylabel)
    ax2.set(xlabel='Iteration', ylabel=ylabel)

    fig.savefig(fig_name)
    if show_fig: plt.show()