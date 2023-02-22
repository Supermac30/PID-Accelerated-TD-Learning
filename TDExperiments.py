import matplotlib.pyplot as plt
import numpy as np

from Environments import Garnet, ChainWalk
from Controllers import P_Controller, D_Controller, I_Controller
from Agents import ControlledTDLearning, SoftControlledTDLearning
from MDP import PolicyEvaluation
from ExperimentHelpers import *


def policy_evaluation_experiment():
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
        lambda k: min(0.03, 10/k)
    )

    V_pi = find_Vpi(env, policy)
    test_function=lambda V, Vp, BR: np.max(np.abs(V - V_pi))

    def run_pid_experiment(kp, kd, ki):
        run_TD_experiment(agent, kp, kd, ki, test_function, plt)

    # Plot vanilla VI
    # run_pid_experiment(1, 0, 0)

    # for i in range(1, 8):
    #     run_pid_experiment(1, i/10, 0)

    run_pid_experiment(1, 0.15, 0)
    # run_pid_experiment(1, 0.2, 0)
    # run_pid_experiment(1, 0.7, 0.2)


    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('$||V_k - V^\pi||_\infty$')
    plt.show()

def soft_policy_evaluation_experiment():
    """Experiments with policy evaluation and TD"""
    num_states = 50
    num_actions = 2
    env = ChainWalk(num_states)
    policy = np.zeros((num_states, num_actions))
    for i in range(num_states):
        policy[i,0] = 1
    agent = SoftControlledTDLearning(
        env,
        policy,
        0.99,
        lambda k: min(0.02, 10/k)
    )

    V_pi = find_Vpi(env, policy)
    test_function=lambda V, Vp, BR: np.max(np.abs(V - V_pi))

    def run_pid_experiment(kp, kd, ki):
        run_TD_experiment(agent, kp, kd, ki, test_function, plt)

    # Plot vanilla VI
    # run_pid_experiment(1, 0, 0)

    # for i in range(1, 8):
    #     run_pid_experiment(1, i/10, 0)

    run_pid_experiment(1, 0.15, 0)
    # run_pid_experiment(1, 0.2, 0)
    # run_pid_experiment(1, 0.7, 0.2)

    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('$||V_k - V^\pi||_\infty$')
    plt.show()


def find_average_update_experiment():
    """
    Experiment with what the expectation of

    \frac{\kappa_p}{\kappa_p - \kappa_d}\Bhat V_k - \frac{\kappa_d}{\kappa_p - \kappa_d}V_{p(k)}

    looks like. This experimentally verifies theorem 2.1 in the manuscript.
    """
    num_states = 50
    num_actions = 2
    gamma = 0.99
    env = ChainWalk(num_states)
    policy = np.zeros((num_states, num_actions))
    for i in range(num_states):
        policy[i,0] = 1
    agent = ControlledTDLearning(
        env,
        policy,
        gamma,
        lambda k: min(0.03, 10/k)
    )

    R = env.build_policy_reward_vector(policy)
    P = env.build_policy_probability_transition_kernel(policy)

    V_pi = find_Vpi(env, policy)

    def bellman(V):
        return R.reshape((-1, 1)) + gamma * P @ V

    def run_experiment(kp, kd):
        test_function=lambda V, Vp, BR: np.max(np.abs(V_pi - bellman(V) * (kp / (kp - kd)) + Vp * (kd / (kp - kd))))
        run_TD_experiment(agent, kp, kd, 0, test_function, plt)

    run_TD_experiment(1, 0.2)

    # run_pid_experiment(1, 0.25, 0)
    # run_pid_experiment(1, 0.2, 0)
    # run_pid_experiment(1, 0.7, 0.2)

    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('$||V^{\pi} - ((\kappa_p)/(\kappa_p - \kappa_d)T^{\pi} V_k - (\kappa_d)/(\kappa_p - \kappa_d) V_{p(k)})||_\infty$')
    plt.show()

if __name__ == "__main__":
    soft_policy_evaluation_experiment()