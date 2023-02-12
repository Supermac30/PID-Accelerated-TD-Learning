import matplotlib.pyplot as plt
import numpy as np

from Environments import Garnet, ChainWalk
from Controllers import P_Controller, D_Controller, I_Controller
from Agents import ControlledTDLearning
from MDP import PolicyEvaluation

def PolicyEvaluationExperiment():
    """Experiments with policy evaluation and TD"""
    num_states = 50
    num_actions = 2
    env = ChainWalk(num_states)
    policy = np.zeros((num_states, num_actions))
    for i in range(num_states):
        policy[i,0] = 1
    agent = ControlledTDLearning(
        env,
        policy,
        0.99,
        lambda k: min(0.02, 10/k)
    )

    oracle = PolicyEvaluation(
        env.num_states,
        env.num_actions,
        env.build_policy_reward_vector(policy),
        env.build_policy_probability_transition_kernel(policy),
        0.99
    )

    # Vanilla Value Iteration is simply a P-controller with a gain equal to I
    p_controller = P_Controller(np.identity(num_states))
    V_pi = oracle.value_iteration(p_controller, num_iterations=10000)

    # Plot vanilla VI
    total_history = 0
    for _ in range(10):
        history, V = agent.estimate_value_function(p_controller, V=V_pi)
        total_history += history
    total_history /= 10
    plt.plot(total_history, label="VI (Conventional)")

    p_controller = P_Controller(1 * np.identity(num_states))
    d_controller = D_Controller(0.25 * np.identity(num_states))
    i_controller = I_Controller(0.05, 0.95, 0 * np.identity(num_states))

    total_history = 0
    for _ in range(10):
        history, V = agent.estimate_value_function(p_controller, d_controller, i_controller, V=V_pi)
        total_history += history
    total_history /= 10
    plt.plot(total_history, label="(k_p, k_i, k_d) = (1, 0, 0.25)")

    p_controller = P_Controller(1 * np.identity(num_states))
    d_controller = D_Controller(0.2 * np.identity(num_states))
    i_controller = I_Controller(0.05, 0.95, 0 * np.identity(num_states))

    total_history = 0
    for _ in range(10):
        history, V = agent.estimate_value_function(p_controller, d_controller, i_controller, V=V_pi)
        total_history += history
    total_history /= 10
    plt.plot(total_history, label="(k_p, k_i, k_d) = (1, 0, 0.2)")

    p_controller = P_Controller(1 * np.identity(num_states))
    d_controller = D_Controller(0.2 * np.identity(num_states))
    i_controller = I_Controller(0.05, 0.95, 0.7 * np.identity(num_states))
    total_history = 0
    for _ in range(10):
        history, V = agent.estimate_value_function(p_controller, d_controller, i_controller, V=V_pi)
        total_history += history
    total_history /= 10
    plt.plot(total_history, label="(k_p, k_i, k_d) = (1, 0.7, 0.2)")

    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('$||V_k - V^\pi||_\infty$')
    plt.show()

if __name__ == "__main__":
    PolicyEvaluationExperiment()