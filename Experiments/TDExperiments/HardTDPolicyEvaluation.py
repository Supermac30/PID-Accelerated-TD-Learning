import matplotlib.pyplot as plt
import hydra

from Experiments.ExperimentHelpers import *
from TabularPID.AgentBuilders.AgentBuilder import build_agent_and_env

@hydra.main(version_base=None, config_path="../../config/TDExperiments", config_name="HardTDPolicyEvaluation")
def policy_evaluation_experiment(cfg):
    """Experiments with policy evaluation and TD"""
    seed = pick_seed(cfg['seed'])
    kp, kd, ki, alpha, beta = cfg['kp'], cfg['kd'], cfg['ki'], cfg['alpha'], cfg['beta']

    agent, env, policy = build_agent_and_env(("hard TD", kp, ki, kd, alpha, beta), cfg['env'], cfg['get_optimal'], seed, cfg['gamma'])
    V_pi = find_Vpi(env, policy, cfg['gamma'])
    test_function = build_test_function(cfg['norm'], V_pi)
    all_histories = []

    for _ in range(cfg['repeat']):
        history, _ = agent.estimate_value_function(num_iterations=cfg['num_iterations'], test_function=test_function, follow_trajectory=cfg['follow_trajectory'])
        all_histories.append(history)

    mean_history = np.mean(np.array(all_histories), axis=0)
    std_dev_history = np.std(np.array(all_histories), axis=0)
    name = cfg['name']
    save_array(mean_history, f"{name}", directory=cfg['save_dir'], subdir="mean")
    save_array(std_dev_history, f"{name}", directory=cfg['save_dir'], subdir="std_dev")




if __name__ == "__main__":
    policy_evaluation_experiment()