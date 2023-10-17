"""Test the formulation of the adaptive agent without learning rates"""
import hydra
import logging

import multiprocessing as mp

from TabularPID.AgentBuilders.AdaptiveAgentBuilder import build_adaptive_agent_and_env
from TabularPID.OptimalRates.HyperparameterTests import get_optimal_q_adaptive_rates
from Experiments.ExperimentHelpers import *

@hydra.main(version_base=None, config_path="../../config/AdaptationExperiments", config_name="AdaptiveQAgentExperiment")
def adaptive_agent_experiment(cfg):
    """Visualize the behavior of adaptation without learning rates."""
    seed = pick_seed(cfg['seed'])
    meta_lr, delay, lambd, alpha, beta, epsilon = cfg['meta_lr'], cfg['delay'], cfg['lambda'], cfg['alpha'], cfg['beta'], cfg['epsilon']
    meta_lr_p, meta_lr_I, meta_lr_d = cfg['meta_lr_p'], cfg['meta_lr_I'], cfg['meta_lr_d']

    if cfg['compute_optimal']:
        get_optimal_q_adaptive_rates(cfg['agent_name'], cfg['env'], meta_lr, cfg['gamma'], lambd, delay, alpha, beta, recompute=cfg['recompute_optimal'], epsilon=epsilon, norm=cfg['norm'], is_q=True, search_steps=cfg['search_steps'], meta_lr_p=meta_lr_p, meta_lr_I=meta_lr_I, meta_lr_d=meta_lr_d)
    agent, env, _ = build_adaptive_agent_and_env(
        cfg['agent_name'],
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
        epsilon=epsilon,
        meta_lr_p=meta_lr_p,
        meta_lr_I=meta_lr_I,
        meta_lr_d=meta_lr_d,
    )
    agent.verbose = True
    Q_star = find_Qstar(env, cfg['gamma'])
    test_function = build_test_function(cfg['norm'], Q_star)
    
    def run_test(seed):
        agent.set_seed(seed)
        Q, gain_history, history = agent.estimate_value_function(
            cfg['num_iterations'],
            test_function,
            follow_trajectory=cfg['follow_trajectory'],
            stop_if_diverging=cfg['stop_if_diverging']
        )
        return Q, gain_history, history

    prg = np.random.RandomState(seed)
    if cfg['debug']:
        _, gain_history, history = run_test(prg.randint(0, 1000000))
        all_histories = [history]
        all_gain_histories = [gain_history]
    else:
        num_chunks = mp.cpu_count()
        logging.info(f"Running experiments {num_chunks} times")
        # Run the following agent.estimate_value_function 80 times and take an average of the histories
        pool = mp.Pool()
        results = pool.map(run_test, [prg.randint(0, 1000000) for _ in range(num_chunks)])
        pool.close()
        pool.join()

        all_gain_histories = list(map(lambda n: results[n][1], range(len(results[0]))))
        all_histories = list(map(lambda n: results[n][2], range(len(results[0]))))
        
    average_history = np.mean(np.array(all_histories), axis=0)
    average_gain_history = np.mean(np.array(all_gain_histories), axis=0)
    std_dev_history = np.std(np.array(all_histories), axis=0)
    std_dev_gain_history = np.std(np.array(all_gain_histories), axis=0)


    agent_description = f"Adaptive Agent Q {meta_lr_p} {meta_lr_I} {meta_lr_d} {delay} {lambd} {epsilon}"
    save_array(average_history, f"{agent_description}", directory=cfg['save_dir'], subdir="mean")
    save_array(average_gain_history, f"gain_history {agent_description}", directory=cfg['save_dir'], subdir="mean")
    save_array(std_dev_history, f"{agent_description}", directory=cfg['save_dir'], subdir="std_dev")
    save_array(std_dev_gain_history, f"gain_history {agent_description}", directory=cfg['save_dir'], subdir="std_dev")

    agent.plot(cfg['save_dir'])


if __name__ == '__main__':
    adaptive_agent_experiment()