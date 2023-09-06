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

    if cfg['compute_optimal']:
        get_optimal_adaptive_linear_FA_rates(name, cfg['env'], cfg['order'], cfg['meta_lr'], cfg['gamma'], cfg['lambd'], cfg['delay'], cfg['alpha'], cfg['beta'], recompute=cfg['recompute_optimal'], epsilon=cfg['epsilon'], search_steps=cfg['search_steps'])
    agent, _, _ = build_adaptive_agent_and_env(
        name,
        cfg['env'],
        cfg['meta_lr'],
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
        epsilon=cfg['epsilon'],
        order=cfg['order']
    )
    histories = []
    gain_histories = []
    for _ in range(cfg['repeat']):
        history, gain_history, _ = agent.estimate_value_function(
            num_iterations=cfg['num_iterations'],
            stop_if_diverging=cfg['stop_if_diverging']
        )
        histories.append(history)
        gain_histories.append(gain_history)
    save_array(np.mean(histories, axis=0), f"{name}", directory=cfg['save_dir'], subdir="mean")
    save_array(np.std(histories, axis=0), f"{name}", directory=cfg['save_dir'], subdir="std_dev")
    save_array(np.mean(gain_histories, axis=0), f"gain_history {name}", directory=cfg['save_dir'], subdir="mean")
    save_array(np.std(gain_histories, axis=0), f"gain_history {name}", directory=cfg['save_dir'], subdir="std_dev")


if __name__ == "__main__":
    soft_policy_evaluation_experiment() 