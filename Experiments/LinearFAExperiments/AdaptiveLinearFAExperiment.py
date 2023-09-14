import hydra

from Experiments.ExperimentHelpers import *
from TabularPID.AgentBuilders.AdaptiveAgentBuilder import build_adaptive_agent_and_env
from TabularPID.OptimalRates.HyperparameterTests import get_optimal_adaptive_linear_FA_rates

@hydra.main(version_base=None, config_path="../../config/LinearFAExperiments", config_name="LinearFAExperiment")
def soft_policy_evaluation_experiment(cfg):
    """Experiments with policy evaluation and TD"""
    seed = pick_seed(cfg['seed'])
    name = f"linear TD {cfg['type']}"
    if cfg['is_q']:
        name += " Q"

    meta_lr = cfg['meta_lr']
    epsilon = cfg['epsilon']

    if cfg['compute_optimal']:
        get_optimal_adaptive_linear_FA_rates(name, cfg['env'], cfg['order'], meta_lr, cfg['gamma'], cfg['lambd'], cfg['delay'], cfg['alpha'], cfg['beta'], recompute=cfg['recompute_optimal'], epsilon=epsilon, search_steps=cfg['search_steps'])
    agent, _, _ = build_adaptive_agent_and_env(
        name,
        cfg['env'],
        meta_lr,
        cfg['lambd'],
        cfg['delay'],
        get_optimal=cfg['get_optimal'],
        seed=seed,
        gamma=cfg['gamma'],
        kp=cfg['kp'],
        ki=cfg['ki'],
        kd=cfg['kd'],
        alpha=cfg['alpha'],
        beta=cfg['beta'],
        epsilon=epsilon,
        order=cfg['order']
    )

    def run_test(_):
        history, gain_history, w_V = agent.estimate_value_function(
            num_iterations=cfg['num_iterations'],
            stop_if_diverging=cfg['stop_if_diverging']
        )
        return history, gain_history, w_V

    if cfg['debug']:
        history, gain_history, _ = run_test()
        all_histories = [history]
        all_gain_histories = [gain_history]
    else:
        num_chunks = mp.cpu_count()
        logging.info(f"Running experiments {num_chunks} times")
        # Run the following agent.estimate_value_function 80 times and take an average of the histories
        pool = mp.Pool()
        results = pool.map(run_test, [None] * num_chunks)
        pool.close()
        pool.join()

        all_gain_histories = list(map(lambda n: results[n][1], range(len(results[0]))))
        all_histories = list(map(lambda n: results[n][0], range(len(results[0]))))

    description = f"{name} {meta_lr} {epsilon}"
    save_array(np.mean(all_histories, axis=0), description, directory=cfg['save_dir'], subdir="mean")
    save_array(np.std(all_histories, axis=0), description, directory=cfg['save_dir'], subdir="std_dev")
    save_array(np.mean(all_gain_histories, axis=0), f"gain_history {description}", directory=cfg['save_dir'], subdir="mean")
    save_array(np.std(all_gain_histories, axis=0), f"gain_history {description}", directory=cfg['save_dir'], subdir="std_dev")


if __name__ == "__main__":
    soft_policy_evaluation_experiment() 