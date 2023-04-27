import matplotlib.pyplot as plt
import numpy as np
import hydra

from Agents import FarSightedControlledTD
from Experiments.ExperimentHelpers import *

@hydra.main(version_base=None, config_path="../../config/TDExperiments", config_name="FarSightedUpdate")
def far_sighted_update_experiment(cfg):
    """Experiments with policy evaluation and TD"""
    env, policy = get_env_policy(cfg['env'], cfg['seed'])

    V_pi = find_Vpi(env, policy)

    for delay in cfg['delays']:
        agent = FarSightedControlledTD(
            env,
            policy,
            0.99,
            learning_rate_function(1, 0),
            delay
        )

        test_function = build_test_function(cfg['norm'], V_pi)

        for kp, kd, ki in zip(cfg['kp'], cfg['kd'], cfg['ki']):
            history, params = find_optimal_pid_learning_rates(
                agent, kp, kd, ki, test_function, cfg['num_iterations'], True,
                learning_rates=cfg['learning_rates'],
                update_D_rates=cfg['update_D_rates'],
                update_I_rates=cfg['update_I_rates']
            )
            save_array(history, f"{kp=} {kd=} {ki=} {delay=} {params=}", plt)


    plt.title(f"Far Sighted Updates: {cfg['env']}")
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('$||V_k - V^\pi||_\infty$')
    plt.savefig("plot")
    plt.show()


if __name__ == "__main__":
    far_sighted_update_experiment()