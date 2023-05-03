import matplotlib.pyplot as plt
import numpy as np
import hydra

from Environments import ChainWalk
from Agents import Hard_PID_TD, PID_TD
from Experiments.ExperimentHelpers import *

@hydra.main(version_base=None, config_path="../../config/ComparisonExperiments", config_name="HardSoftComparison")
def hard_soft_convergence_experiment(cfg):
    """Experiment with the convergence rate of hard and soft derivative updates."""
    env, policy = get_env_policy(cfg['env'], cfg['seed'])
    soft = PID_TD(
        env,
        policy,
        0.99,
        learning_rate_function(1, 0),
        learning_rate_function(1, 0)
    )
    hard = Hard_PID_TD(
        env,
        policy,
        0.99,
        learning_rate_function(1, 0)
    )

    V_pi = find_Vpi(env, policy)
    test_function = build_test_function(cfg['norm'], V_pi)

    fig, (ax1, ax2) = plt.subplots(2)

    for kp, kd, ki in zip(cfg['kp'], cfg['kd'], cfg['ki']):
        soft_history, soft_rates = find_optimal_pid_learning_rates(
            soft, kp, kd, ki, test_function, cfg['num_iterations'], True
        )
        hard_history, hard_rates = find_optimal_pid_learning_rates(
            hard, kp, kd, ki, test_function, cfg['num_iterations'], False
        )

        save_array(soft_history, f"soft kp={kp} kd={kd} ki={ki} {soft_rates}", ax1)
        save_array(hard_history, f"hard kp={kp} kd={kd} ki={ki} {hard_rates}", ax2)

    plot_comparison(fig, ax1, ax2, 'Soft Updates', 'Hard Updates', f"$||V_k - V^\pi||_{cfg['norm']}$")


if __name__ == '__main__':
    hard_soft_convergence_experiment()