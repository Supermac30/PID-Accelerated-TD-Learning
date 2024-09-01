import hydra

from Experiments.ExperimentHelpers import *
from TabularPID.AgentBuilders.AgentBuilder import build_agent_and_env
from TabularPID.OptimalRates.HyperparameterTests import get_optimal_pid_rates

@hydra.main(version_base=None, config_path="../../config/TDExperiments", config_name="SoftTDPolicyEvaluation")
def soft_policy_evaluation_experiment(cfg):
    """Experiments with policy evaluation and TD"""
    seed = pick_seed(cfg['seed'])
    agent_name, kp, ki, kd, alpha, beta = cfg['agent_name'], cfg['kp'], cfg['ki'], cfg['kd'], cfg['alpha'], cfg['beta']

    if cfg['compute_optimal']:
        get_optimal_pid_rates(agent_name, cfg['env'], kp, ki, kd, alpha, beta, cfg['gamma'], recompute=cfg['recompute_optimal'], search_steps=cfg['search_steps'])
    agent, env, policy = build_agent_and_env((agent_name, kp, ki, kd, alpha, beta), cfg['env'], cfg['get_optimal'], seed, cfg['gamma'])
    V_pi = find_Vpi(env, policy, cfg['gamma'])
    test_function = build_test_function(cfg['norm'], V_pi)
    
    def run_test(seed):
        agent.set_seed(seed)
        history, V = agent.estimate_value_function(
            cfg['num_iterations'],
            test_function,
            follow_trajectory=cfg['follow_trajectory'],
            stop_if_diverging=cfg['stop_if_diverging']
        )
        return history, V

    prg = np.random.RandomState(seed)
    if cfg['debug']:
        history, _ = run_test(prg.randint(0, 1000000))
        all_histories = [history]
    else:
        num_chunks = cfg['repeat']
        logging.info(f"Running experiments {num_chunks} times")
        # Run the following agent.estimate_value_function 80 times and take an average of the histories
        pool = mp.Pool()
        results = pool.map(run_test, [prg.randint(0, 1000000) for _ in range(num_chunks)])
        pool.close()
        pool.join()

        logging.info(results)

        all_histories = list(map(lambda n: results[n][0], range(len(results[0]))))

    mean_history = np.mean(np.array(all_histories), axis=0)
    std_dev_history = np.std(np.array(all_histories), axis=0)
    save_array(mean_history, f"{cfg['name']}", directory=cfg['save_dir'], subdir="mean")
    save_array(std_dev_history, f"{cfg['name']}", directory=cfg['save_dir'], subdir="std_dev")

if __name__ == "__main__":
    soft_policy_evaluation_experiment()