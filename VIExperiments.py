import matplotlib.pyplot as plt
import numpy as np

from Environments import Garnet, ChainWalk
from Controllers import P_Controller, D_Controller, I_Controller
from MDP import Control, PolicyEvaluation
from ExperimentHelpers import *

def policy_evaluation_experiment():
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

    V_pi = find_Vpi(env, policy)
    test_function=lambda V, Vp, BR: np.max(np.abs(V - V_pi))
    def run_pid_experiment(kp, kd, ki):
        run_VI_experiment(agent, kp, kd, ki, test_function, plt)

    # Plot vanilla VI
    run_pid_experiment(1, 0, 0)
    run_pid_experiment(1, 0.5, 0)

    # run_pid_experiment(1.1, 0, 0)
    # run_pid_experiment(1, 0.15, 0)
    # run_pid_experiment(1, 0, -0.4)


    plt.legend()
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('$||V_k - V^\pi||$')
    plt.show()

def control_experiment():
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

    V_star = find_Vstar(env)
    test_function=lambda V, Vp, BR: np.max(np.abs(V - V_star))
    def run_pid_experiment(kp, kd, ki):
        run_VI_experiment(agent, kp, kd, ki, test_function, plt)

    # Plot vanilla VI
    run_pid_experiment(1, 0, 0)

    # run_pid_experiment(1.2, 0, 0)
    run_pid_experiment(1, 0.4, 0)
    run_pid_experiment(1, 0, 0.75)
    run_pid_experiment(1, 0.4, 0.75)
    run_pid_experiment(1, 0.2, 0.7)

    plt.legend()
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('$||V_k - V^*||$')
    plt.show()


if __name__ == "__main__":
    policy_evaluation_experiment()
    #control_experiment()