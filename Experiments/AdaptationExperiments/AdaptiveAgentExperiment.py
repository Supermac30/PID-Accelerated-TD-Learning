"""Test the formulation of the adaptive agent without learning rates"""

import hydra

from TabularPID.AgentBuilders.AdaptiveAgentBuilder import build_adaptive_agent_and_env
from TabularPID.OptimalRates.HyperparameterTests import get_optimal_adaptive_rates
from Experiments.ExperimentHelpers import *

@hydra.main(version_base=None, config_path="../../config/AdaptationExperiments", config_name="AdaptiveAgentExperiment")
def adaptive_agent_experiment(cfg):
    """Visualize the behavior of adaptation without learning rates."""
    seed = pick_seed(cfg['seed'])
    
    agent_name, meta_lr, lambd, delay, alpha, beta, epsilon = cfg['agent_name'], cfg['meta_lr'], cfg['lambda'], cfg['delay'], cfg['alpha'], cfg['beta'], cfg['epsilon']
    agent_description = f"Adaptive Agent {agent_name} {meta_lr} {delay} {lambd} {epsilon}"

    if cfg['compute_optimal']:
        get_optimal_adaptive_rates(agent_name, cfg['env'], meta_lr, cfg['gamma'], lambd, delay, alpha, beta, recompute=cfg['recompute_optimal'], epsilon=epsilon, search_steps=cfg['search_steps'])
    agent, env, policy = build_adaptive_agent_and_env(
        agent_name,
        cfg['env'],
        meta_lr,
        lambd,
        delay,
        get_optimal=cfg['get_optimal'],
        seed=seed,
        gamma=cfg['gamma'],
        kp=cfg['kp'],
        ki=cfg['ki'],
        kd=cfg['kd'],
        alpha=alpha,
        beta=beta,
        epsilon=epsilon
    )
    V_pi = find_Vpi(env, policy, cfg['gamma'])
    test_function = build_test_function(cfg['norm'], V_pi)
    # Run the following agent.estimate_value_function 20 times and take an average of the histories
    average_history = np.zeros((cfg['num_iterations'],))
    all_histories = []
    all_gain_histories = []
    for _ in range(cfg['repeat']):
        _, gain_history, history = agent.estimate_value_function(
            cfg['num_iterations'],
            test_function,
            follow_trajectory=cfg['follow_trajectory'],
            stop_if_diverging=cfg['stop_if_diverging']
        )
        all_histories.append(history)
        all_gain_histories.append(gain_history)

    average_history = np.mean(np.array(all_histories), axis=0)
    std_deviation_history = np.std(np.array(all_histories), axis=0)
    average_gain_history = np.mean(np.array(all_gain_histories), axis=0)
    std_deviation_gain_history = np.std(np.array(all_gain_histories), axis=0)
    save_array(average_history, f"{agent_description}", directory=cfg['save_dir'], subdir="mean")
    save_array(std_deviation_history, f"{agent_description}", directory=cfg['save_dir'], subdir="std_dev")
    save_array(average_gain_history, f"gain_history {agent_description}", directory=cfg['save_dir'], subdir="mean")
    save_array(std_deviation_gain_history, f"gain_history {agent_description}", directory=cfg['save_dir'], subdir="std_dev")

    agent.plot(directory=cfg['save_dir'])


if __name__ == '__main__':
    adaptive_agent_experiment()