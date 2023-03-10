import matplotlib.pyplot as plt
import numpy as np
import hydra

from Environments import ChainWalk
from Agents import FarSightedHardControlledTD
from Experiments.ExperimentHelpers import *

@hydra.main(version_base=None, config_path="../../config/TDExperiments", config_name="FarSightedHardUpdate")
def far_sighted_hard_update_experiment(cfg):
    """Experiments with policy evaluation and TD"""
    env, policy = chain_walk_left(50, 2, cfg['seed'])

    V_pi = find_Vpi(env, policy)

    for delay in cfg['delays']:
        agent = FarSightedHardControlledTD(
            env,
            policy,
            0.99,
            learning_rate_function(1, 0),
            delay
        )

        test_function=lambda V, Vp, BR: np.max(np.abs(V - V_pi))

        for kp, kd, ki in zip(cfg['kp'], cfg['kd'], cfg['ki']):
            history, params = find_optimal_pid_learning_rates(
                agent, kp, kd, ki, test_function, cfg['num_iterations'], False, learning_rates=cfg['learning_rates']
            )
            save_array(history, f"{kp=} {kd=} {ki=} {delay=} {params=}", plt)


    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('$||V_k - V^\pi||_\infty$')
    plt.show()
    plt.savefig("plot")


if __name__ == "__main__":
    far_sighted_hard_update_experiment()