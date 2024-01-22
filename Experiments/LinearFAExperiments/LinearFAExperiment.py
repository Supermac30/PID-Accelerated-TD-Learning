import hydra

from Experiments.ExperimentHelpers import *
from TabularPID.AgentBuilders.AgentBuilder import build_agent_and_env
from TabularPID.OptimalRates.HyperparameterTests import get_optimal_linear_FA_rates

@hydra.main(version_base=None, config_path="../../config/LinearFAExperiments", config_name="LinearFAExperiment")
def soft_policy_evaluation_experiment(cfg):
    """Experiments with policy evaluation and TD"""
    seed = pick_seed(cfg['seed'])
    agent_name = "linear TD " + cfg['type']
    if cfg['is_q']:
        agent_name += " Q"
    
    kp, ki, kd, alpha, beta = cfg['kp'], cfg['ki'], cfg['kd'], cfg['alpha'], cfg['beta']
    order = cfg['order']

    if cfg['compute_optimal']:
        get_optimal_linear_FA_rates(agent_name, cfg['env'], kp, ki, kd, alpha, beta, cfg['gamma'], cfg['order'], cfg['recompute_optimal'], search_steps=cfg['search_steps'])
    agent, _, _ = build_agent_and_env((agent_name, kp, ki, kd, alpha, beta, order), cfg['env'], cfg['get_optimal'], seed, cfg['gamma'])
    
    def run_test(seed):
        agent.set_seed(seed)
        history, w_V = agent.estimate_value_function(
            num_iterations=cfg['num_iterations'],
            stop_if_diverging=cfg['stop_if_diverging']
        )
        return w_V, history

    prg = np.random.RandomState(seed)
    if cfg['debug']:
        _, history = run_test(prg.randint(0, 1000000))
        all_histories = [history]
    else:
        num_chunks = mp.cpu_count()
        logging.info(f"Running experiments {num_chunks} times")
        # Run the following agent.estimate_value_function 80 times and take an average of the histories
        pool = mp.Pool()
        results = pool.map(run_test, [prg.randint(0, 1000000) for _ in range(num_chunks)])
        pool.close()
        pool.join()

        all_histories = list(map(lambda n: results[n][1], range(len(results[0]))))
        
    name = cfg['name']
    save_array(np.mean(all_histories, axis=0), f"{name}", directory=cfg['save_dir'], subdir="mean")
    save_array(np.std(all_histories, axis=0), f"{name}", directory=cfg['save_dir'], subdir="std_dev")


if __name__ == "__main__":
    soft_policy_evaluation_experiment()