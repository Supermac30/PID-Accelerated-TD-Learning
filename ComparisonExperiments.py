import matplotlib.pyplot as plt
import numpy as np

from Environments import Garnet, ChainWalk
from Controllers import P_Controller, D_Controller, I_Controller
from Agents import ControlledTDLearning, SoftControlledTDLearning
from MDP import PolicyEvaluation
from ExperimentHelpers import *


def convergence_rate_VI_experiment():
    """Compare convergence rate of PID-TD and PID-VI"""
    num_states = 50
    num_actions = 2
    env = ChainWalk(num_states)
    policy = np.zeros((num_states, num_actions))
    for i in range(num_states):
        policy[i,0] = 1
    TDagent = ControlledTDLearning(
        env,
        policy,
        0.99,
        lambda k: min(0.03, 10/k)
    )
    VIagent = PolicyEvaluation(
        env.num_states,
        env.num_actions,
        env.build_policy_reward_vector(policy),
        env.build_policy_probability_transition_kernel(policy),
        0.99
    )

    V_pi = find_Vpi(env, policy)
    test_function = lambda V, Vp, BR: np.max(np.abs(V - V_pi))

    fig, (ax1, ax2) = plt.subplots(2)

    def TD_agent_experiment(kp, kd, ki):
        run_TD_experiment(TDagent, kp, kd, ki, test_function, ax2)
    def VI_agent_experiment(kp, kd, ki):
        run_VI_experiment(VIagent, kp, kd, ki, test_function, ax1)

    TD_agent_experiment(TDagent, 1.2, 0, 0, test_function, ax2)
    VI_agent_experiment(VIagent, 1.2, 0, 0, test_function, ax1)

    plot_comparison(fig, ax1, ax2, 'PID Accelerated VI', 'PID Accelerated TD', '$||V_k - V^\pi||_\infty$')


def hard_soft_convergence_experiment():
    """Experiment with the convergence rate of hard and soft derivative updates."""
    num_states = 50
    num_actions = 2
    env = ChainWalk(num_states)
    policy = np.zeros((num_states, num_actions))
    for i in range(num_states):
        policy[i,0] = 1
    soft = SoftControlledTDLearning(
        env,
        policy,
        0.99,
        lambda k: min(0.005, 1000/k)
    )
    hard = ControlledTDLearning(
        env,
        policy,
        0.99,
        lambda k: min(0.02, 10/k)
    )

    V_pi = find_Vpi(env, policy)
    test_function = lambda V, Vp, BR: np.max(np.abs(V - V_pi))

    fig, (ax1, ax2) = plt.subplots(2)

    def run_soft_experiment(kp, kd, ki):
        run_TD_experiment(soft, kp, kd, ki, test_function, ax1)

    def run_hard_experiment(kp, kd, ki):
        run_TD_experiment(hard, kp, kd, ki, test_function, ax2)

    run_soft_experiment(1, 0.1, 0)
    run_hard_experiment(1, 0.1, 0)

    plot_comparison(fig, ax1, ax2, 'PID Accelerated VI', 'PID Accelerated TD', '$||V_k - V^\pi||_\infty$')


if __name__ == '__main__':
    hard_soft_convergence_experiment()