import matplotlib.pyplot as plt
import numpy as np
import hydra

from Environments import ChainWalk
from Agents import SoftControlledTDLearning
from Experiments.ExperimentHelpers import *

@hydra.main(version_base=None, config_path="../../config/TDExperiments", config_name="SoftTDPolicyEvaluation")
def soft_policy_evaluation_experiment(cfg):
    """Experiments with policy evaluation and TD"""
    env, policy = get_env_policy(cfg['env'], cfg['seed'])
    agent = SoftControlledTDLearning(
        env,
        policy,
        0.99,
        (learning_rate_function(cfg['alpha'], cfg['N']),
        learning_rate_function(cfg['update_I_alpha'], cfg['update_I_N']),
        learning_rate_function(cfg['update_D_alpha'], cfg['update_D_N']))
    )

    V_pi = find_Vpi(env, policy)
    test_function = build_test_function(cfg['norm'], V_pi)

    for kp, kd, ki in zip(cfg['kp'], cfg['kd'], cfg['ki']):
        history = run_PID_TD_experiment(agent, kp, kd, ki, test_function, cfg['num_iterations'])
        save_array(history, f"kp={kp} kd={kd} ki={ki}", plt)

    plt.title(f"Soft Updates: {cfg['env']}")
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel(f"$||V_k - V^\pi||_{cfg['norm']}$")
    plt.savefig("plot")
    plt.show()

if __name__ == "__main__":
    soft_policy_evaluation_experiment()