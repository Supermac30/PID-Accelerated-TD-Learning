import hydra

from Experiments.ExperimentHelpers import *
from TabularPID.AgentBuilders.AgentBuilder import build_agent_and_env
from TabularPID.OptimalRates.HyperparameterTests import get_optimal_past_work_rates

@hydra.main(version_base=None, config_path="../../config/TDExperiments", config_name="PastWorkEvaluation")
def past_work(cfg):
    """Experiments with policy evaluation and TD"""
    seed = pick_seed(cfg['seed'])
    agent_name = cfg['agent_name']

    if cfg['compute_optimal']:
        get_optimal_past_work_rates((agent_name, 1, 0, 0, 0, 0), cfg['env'], cfg['gamma'], cfg['recompute_optimal'])
    agent, env, policy = build_agent_and_env((agent_name, 1, 0, 0, 0, 0), cfg['env'], cfg['get_optimal'], seed, cfg['gamma'])
    V_pi = find_Vpi(env, policy, cfg['gamma'])
    test_function = build_test_function(cfg['norm'], V_pi)
    all_histories = []
    for _ in range(cfg['repeat']):
        history, _ = agent.estimate_value_function(
            num_iterations=cfg['num_iterations'],
            test_function=test_function,
            follow_trajectory=cfg['follow_trajectory'],
            stop_if_diverging=cfg['stop_if_diverging'],
            reset_environment=False
        )
        all_histories.append(history)
    mean_history = np.mean(np.array(all_histories), axis=0)
    std_dev_history = np.std(np.array(all_histories), axis=0)
    save_array(mean_history, f"{agent_name}", directory=cfg['save_dir'], subdir="mean")
    save_array(std_dev_history, f"{agent_name}", directory=cfg['save_dir'], subdir="std_dev")


if __name__ == "__main__":
    past_work()