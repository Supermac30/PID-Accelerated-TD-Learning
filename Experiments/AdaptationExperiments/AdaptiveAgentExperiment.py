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
    
    def run_test(seed):
        agent.set_seed(seed)
        V, gain_history, history = agent.estimate_value_function(
            cfg['num_iterations'],
            test_function,
            follow_trajectory=cfg['follow_trajectory'],
            stop_if_diverging=cfg['stop_if_diverging']
        )
        return V, gain_history, history

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