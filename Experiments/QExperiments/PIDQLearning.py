import hydra

from Experiments.ExperimentHelpers import *
from TabularPID.AgentBuilders.AgentBuilder import build_agent_and_env
from TabularPID.OptimalRates.HyperparameterTests import get_optimal_pid_q_rates
from time import time

@hydra.main(version_base=None, config_path="../../config/QExperiments", config_name="PIDQLearning")
def soft_policy_evaluation_experiment(cfg):
    """Experiments with policy evaluation and TD"""
    seed = pick_seed(cfg['seed'])
    agent_name, kp, ki, kd, alpha, beta = cfg['agent_name'], cfg['kp'], cfg['ki'], cfg['kd'], cfg['alpha'], cfg['beta']
    if cfg['compute_optimal']:
        get_optimal_pid_q_rates(agent_name, cfg['env'], kp, ki, kd, alpha, beta, cfg['gamma'], cfg['recompute_optimal'], search_steps=cfg['search_steps'])
    agent, env, _ = build_agent_and_env((agent_name, kp, ki, kd, alpha, beta), cfg['env'], cfg['get_optimal'], seed, cfg['gamma'])
    Q_star = find_Qstar(env, cfg['gamma'])
    test_function = build_test_function(cfg['norm'], Q_star)

    def run_test(seed):
        agent.set_seed(seed)
        start_time = time()
        history, Q = agent.estimate_value_function(
            num_iterations=cfg['num_iterations'],
            test_function=test_function,
            follow_trajectory=cfg['follow_trajectory'],
            stop_if_diverging=cfg['stop_if_diverging'],
            measure_time=cfg['measure_time']
        )
        end_time = time()
        return history, Q, end_time - start_time
    
    prg = np.random.RandomState(seed)
    if cfg['debug']:
        history, _ , time_taken = run_test(prg.randint(0, 1000000))
        all_histories = [history]
        all_time_taken = [time_taken]
    else:
        # Create a psuedo random number generator with seed seed
        num_chunks = cfg['repeat']
        logging.info(f"Running experiments {num_chunks} times")
        # Run the following agent.estimate_value_function 80 times and take an average of the histories
        pool = mp.Pool()
        results = pool.map(run_test, [prg.randint(0, 1000000) for _ in range(num_chunks)])
        pool.close()
        pool.join()

        all_histories = list(map(lambda n: results[n][0], range(len(results[0]))))
        all_time_taken = list(map(lambda n: results[n][2], range(len(results[0]))))

    mean_history = np.mean(np.array(all_histories), axis=0)
    std_dev_history = np.std(np.array(all_histories), axis=0)
    mean_time_taken = np.mean(np.array(all_time_taken), axis=0)
    std_dev_time_taken = np.std(np.array(all_time_taken), axis=0)
    save_array(mean_history, f"{cfg['name']}", directory=cfg['save_dir'], subdir="mean")
    save_array(std_dev_history, f"{cfg['name']}", directory=cfg['save_dir'], subdir="std_dev")
    save_time(env.num_states, mean_time_taken, std_dev_time_taken, cfg['save_dir'], cfg['name'])


if __name__ == "__main__":
    soft_policy_evaluation_experiment()