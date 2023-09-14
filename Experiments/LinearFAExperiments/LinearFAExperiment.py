import hydra

from Experiments.ExperimentHelpers import *
from TabularPID.AgentBuilders.AgentBuilder import build_agent_and_env
from TabularPID.OptimalRates.HyperparameterTests import get_optimal_linear_FA_rates

@hydra.main(version_base=None, config_path="../../config/LinearFAExperiments", config_name="LinearFAExperiment")
def soft_policy_evaluation_experiment(cfg):
    """Experiments with policy evaluation and TD"""
    seed = pick_seed(cfg['seed'])
    name = f"linear TD {cfg['type']}"
    if cfg['is_q']:
        name += " Q"
    kp, ki, kd, alpha, beta = cfg['kp'], cfg['ki'], cfg['kd'], cfg['alpha'], cfg['beta']
    order = cfg['order']

    if cfg['compute_optimal']:
        get_optimal_linear_FA_rates(name, cfg['env'], kp, ki, kd, alpha, beta, cfg['gamma'], cfg['order'], cfg['recompute_optimal'], search_steps=cfg['search_steps'])
    agent, _, _ = build_agent_and_env((name, kp, ki, kd, alpha, beta, order), cfg['env'], cfg['get_optimal'], seed, cfg['gamma'])
    
    def run_test(_):
        history, w_V = agent.estimate_value_function(
            num_iterations=cfg['num_iterations'],
            stop_if_diverging=cfg['stop_if_diverging']
        )
        return w_V, history

    if cfg['debug']:
        _, history = run_test()
        all_histories = [history]
    else:
        num_chunks = mp.cpu_count()
        logging.info(f"Running experiments {num_chunks} times")
        # Run the following agent.estimate_value_function 80 times and take an average of the histories
        pool = mp.Pool()
        results = pool.map(run_test, [None] * num_chunks)
        pool.close()
        pool.join()

        all_histories = list(map(lambda n: results[n][1], range(len(results[0]))))
        
    save_array(np.mean(all_histories, axis=0), f"{name} kp={kp} ki={ki} kd={kd} alpha={alpha} beta={beta}", directory=cfg['save_dir'], subdir="mean")
    save_array(np.std(all_histories, axis=0), f"{name} kp={kp} ki={ki} kd={kd} alpha={alpha} beta={beta}", directory=cfg['save_dir'], subdir="std_dev")


if __name__ == "__main__":
    soft_policy_evaluation_experiment()