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
        get_optimal_past_work_rates(agent_name, cfg['env'], cfg['gamma'], cfg['recompute_optimal'])
    agent, env, policy = build_agent_and_env((agent_name), cfg['env'], cfg['get_optimal'], seed, cfg['gamma'])
    V_pi = find_Vpi(env, policy, cfg['gamma'])
    test_function = build_test_function(cfg['norm'], V_pi)
    total_history = 0
    for _ in range(cfg['num_repeats']):
        history, _ = agent.estimate_value_function(
            num_iterations=cfg['num_iterations'],
            test_function=test_function,
            follow_trajectory=cfg['follow_trajectory'],
            stop_if_diverging=cfg['stop_if_diverging']
        )
        total_history += history
    total_history /= cfg['num_repeats']
    save_array(total_history, f"{agent_name}")

if __name__ == "__main__":
    past_work()