import numpy as np
import matplotlib.pyplot as plt
from Controllers import P_Controller, D_Controller, I_Controller
from MDP import PolicyEvaluation, Control

def run_TD_experiment(agent, kp, kd, ki, test_function, graph):
    """Have the agent estimate the value function using some choice of control gains,
    and graph the value of test_function during training.

    test_function should be a function that takes in V, Vp, BR and returns a real number.
    If graph is None, nothing is plotted.
    """
    p_controller = P_Controller(kp * np.identity(agent.num_states))
    d_controller = D_Controller(kd * np.identity(agent.num_states))
    i_controller = I_Controller(0.05, 0.95, ki * np.identity(agent.num_states))

    total_history = 0
    for _ in range(10):
        history, V = agent.estimate_value_function(
            p_controller,
            d_controller,
            i_controller,
            test_function=test_function,
            num_iterations=5000
        )
        total_history += history
    total_history /= 10

    if graph is not None:
        graph.plot(total_history, label=f"(k_p, k_i, k_d) = ({kp}, {ki}, {kd})")

def run_VI_experiment(agent, kp, kd, ki, test_function, graph):
    """Have the agent estimate the value function using some choice of control gains,
    and graph the value of test_function during training.

    test_function should be a function that takes in V, Vp, BR and returns a real number.
    If graph is None, nothing is plotted.
    """
    p_controller = P_Controller(kp * np.identity(agent.num_states))
    d_controller = D_Controller(kd * np.identity(agent.num_states))
    i_controller = I_Controller(0.05, 0.95, kd * np.identity(agent.num_states))
    history, V = agent.value_iteration(
        p_controller,
        d_controller,
        i_controller,
        test_function=test_function
    )
    if graph is not None:
        graph.plot(history, label=f"(k_p, k_i, k_d) = ({kp}, {ki}, {kd})")

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

def plot_comparison(fig, ax1, ax2, title1, title2, ylabel):
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
    plt.show()