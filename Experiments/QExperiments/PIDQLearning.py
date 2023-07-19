import matplotlib.pyplot as plt
import hydra

from Experiments.ExperimentHelpers import *
from TabularPID.AgentBuilders.AgentBuilder import build_agent_and_env
from TabularPID.OptimalRates.HyperparameterTests import get_optimal_pid_q_rates

@hydra.main(version_base=None, config_path="../../config/QExperiments", config_name="PIDQLearning")
def soft_policy_evaluation_experiment(cfg):
    """Experiments with policy evaluation and TD"""
    seed = pick_seed(cfg['seed'])
    agent_name, kp, ki, kd, alpha, beta, decay = cfg['agent_name'], cfg['kp'], cfg['ki'], cfg['kd'], cfg['alpha'], cfg['beta'], cfg['decay']
    if cfg['compute_optimal']:
        get_optimal_pid_q_rates(agent_name, cfg['env'], kp, ki, kd, alpha, beta, cfg['gamma'], cfg['recompute_optimal'], decay=decay)
    agent, env, _ = build_agent_and_env((agent_name, kp, ki, kd, alpha, beta, decay), cfg['env'], cfg['get_optimal'], seed, cfg['gamma'])
    Q_star = find_Qstar(env, cfg['gamma'])
    test_function = build_test_function(cfg['norm'], Q_star)
    
    all_histories = []
    for _ in range(cfg['repeat']):
        history, _ = agent.estimate_value_function(
            num_iterations=cfg['num_iterations'],
            test_function=test_function,
            follow_trajectory=cfg['follow_trajectory'],
            reset_environment=False
        )
        all_histories.append(history)
    mean_history = np.mean(np.array(all_histories), axis=0)
    std_dev_history = np.std(np.array(all_histories), axis=0)
    save_array(mean_history, f"{agent_name} kp={kp} ki={ki} kd={kd} alpha={alpha} beta={beta}", directory=cfg['save_dir'], subdir="mean")
    save_array(std_dev_history, f"{agent_name} kp={kp} ki={ki} kd={kd} alpha={alpha} beta={beta}", directory=cfg['save_dir'], subdir="std_dev")

if __name__ == "__main__":
    soft_policy_evaluation_experiment()