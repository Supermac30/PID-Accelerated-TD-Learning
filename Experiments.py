import matplotlib.pyplot as plt
import numpy as np

from Environments import Garnet, ChainWalk
from Controllers import P_Controller, D_Controller, I_Controller
from MDP import Control, PolicyEvaluation

def PolicyEvaluationExperiment():
    """Attempt to replicate results in figure 1 of PID Accelerated VI"""
    num_states = 50
    num_actions = 2
    env = ChainWalk(num_states)
    policy = np.zeros((num_states, num_actions))
    for i in range(num_states):
        policy[i,0] = 1
    agent = PolicyEvaluation(
        env.num_states,
        env.num_actions,
        env.build_policy_reward_vector(policy),
        env.build_policy_probability_transition_kernel(policy),
        0.99
    )

    # Vanilla Value Iteration is simply a P-controller with a gain equal to I
    p_controller = P_Controller(np.identity(num_states))
    V_pi = agent.value_iteration(p_controller, num_iterations=10000)

    # Plot vanilla VI
    agent.value_iteration(p_controller, V=V_pi, label="VI (conventional)")

    p_controller = P_Controller(1.1 * np.identity(num_states))
    d_controller = D_Controller(0 * np.identity(num_states))
    i_controller = I_Controller(0.05, 0.95, 0 * np.identity(num_states))
    agent.value_iteration(p_controller, d_controller, i_controller, V=V_pi, label="(k_p, k_i, k_d) = (1.1, 0, 0)")

    p_controller = P_Controller(1 * np.identity(num_states))
    d_controller = D_Controller(0.15 * np.identity(num_states))
    i_controller = I_Controller(0.05, 0.95, 0 * np.identity(num_states))
    agent.value_iteration(p_controller, d_controller, i_controller, V=V_pi, label="(k_p, k_i, k_d) = (1, 0, 0.15)")

    p_controller = P_Controller(1 * np.identity(num_states))
    d_controller = D_Controller(0 * np.identity(num_states))
    i_controller = I_Controller(0.05, 0.95, -0.4 * np.identity(num_states))
    agent.value_iteration(p_controller, d_controller, i_controller, V=V_pi, label="(k_p, k_i, k_d) = (1, -0.4, 0)")

    plt.legend()
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('$||V_k - V^\pi||$')
    plt.show()

def ControlExperiment():
    """Attempt to replicate results in figure 2 of PID Accelerated VI"""
    num_states = 50
    num_actions = 2
    env = ChainWalk(num_states)
    agent = Control(
        env.num_states,
        env.num_actions,
        env.build_reward_matrix(),
        env.build_probability_transition_kernel(),
        0.99
    )

    # Vanilla Value Iteration is simply a P-controller with a gain equal to I
    p_controller = P_Controller(np.identity(num_states))
    V_pi = agent.value_iteration(p_controller, num_iterations=10000)

    # Plot vanilla VI
    agent.value_iteration(p_controller, V=V_pi, label="VI (conventional)")

    p_controller = P_Controller(1.2 * np.identity(num_states))
    d_controller = D_Controller(0 * np.identity(num_states))
    i_controller = I_Controller(0.05, 0.95, 0 * np.identity(num_states))
    agent.value_iteration(p_controller, d_controller, i_controller, V=V_pi, label="(k_p, k_i, k_d) = (1.2, 0, 0)")

    p_controller = P_Controller(1 * np.identity(num_states))
    d_controller = D_Controller(0.4 * np.identity(num_states))
    i_controller = I_Controller(0.05, 0.95, 0 * np.identity(num_states))
    agent.value_iteration(p_controller, d_controller, i_controller, V=V_pi, label="(k_p, k_i, k_d) = (1, 0, 0.4)")

    p_controller = P_Controller(1 * np.identity(num_states))
    d_controller = D_Controller(0 * np.identity(num_states))
    i_controller = I_Controller(0.05, 0.95, 0.75 * np.identity(num_states))
    agent.value_iteration(p_controller, d_controller, i_controller, V=V_pi, label="(k_p, k_i, k_d) = (1, 0.75, 0)")

    p_controller = P_Controller(1 * np.identity(num_states))
    d_controller = D_Controller(0.4 * np.identity(num_states))
    i_controller = I_Controller(0.05, 0.95, 0.75* np.identity(num_states))
    agent.value_iteration(p_controller, d_controller, i_controller, V=V_pi, label="(k_p, k_i, k_d) = (1, 0.75, 0.4)")

    p_controller = P_Controller(1 * np.identity(num_states))
    d_controller = D_Controller(0.2 * np.identity(num_states))
    i_controller = I_Controller(0.05, 0.95, 0.7 * np.identity(num_states))
    agent.value_iteration(p_controller, d_controller, i_controller, V=V_pi, label="(k_p, k_i, k_d) = (1, 0.7, 0.2)")

    plt.legend()
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('$||V_k - V^*||$')
    plt.show()


if __name__ == "__main__":
    # PolicyEvaluationExperiment()
    ControlExperiment()