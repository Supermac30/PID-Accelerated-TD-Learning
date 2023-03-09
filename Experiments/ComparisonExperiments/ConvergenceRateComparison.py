import matplotlib.pyplot as plt
import numpy as np
import hydra

from Environments import ChainWalk
from Agents import ControlledTDLearning, SoftControlledTDLearning
from MDP import PolicyEvaluation
from Experiments.ExperimentHelpers import *

@hydra.main(version_base=None, config_path="../../config/ComparisonExperiments", config_name="ConvergenceRateComparison")
def convergence_rate_VI_experiment(cfg):
    """Compare convergence rate of PID-TD and PID-VI"""
    num_states = 50
    num_actions = 2
    env = ChainWalk(num_states, cfg['seed'])
    policy = np.zeros((num_states, num_actions))
    for i in range(num_states):
        policy[i,0] = 1

    if cfg['isSoft']:
        TDalg = SoftControlledTDLearning
    else:
        TDalg = ControlledTDLearning

    TDagent = TDalg(
        env,
        policy,
        0.99,
        learning_rate_function(1, 0)
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

    for kp, kd, ki in zip(cfg['kp'], cfg['kd'], cfg['ki']):
        TD_history, td_rates = find_optimal_pid_learning_rates(TDagent, kp, kd, ki, test_function, cfg['num_iterations'], cfg['isSoft'])
        VI_history = run_VI_experiment(VIagent, kp, kd, ki, test_function)

        save_array(TD_history, f"{kp=} {kd=} {ki=} {td_rates}", ax2)
        save_array(VI_history, f"{kp=} {kd=} {ki=}", ax1)

    plot_comparison(fig, ax1, ax2, 'PID Accelerated VI', 'PID Accelerated TD', '$||V_k - V^\pi||_\infty$')

if __name__ == '__main__':
    convergence_rate_VI_experiment()