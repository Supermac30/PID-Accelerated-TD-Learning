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
    histories = []
    for _ in range(cfg['repeat']):
        history, _ = agent.estimate_value_function(
            num_iterations=cfg['num_iterations'],
            stop_if_diverging=cfg['stop_if_diverging'],
            
        )
        histories.append(history)
    save_array(np.mean(histories, axis=0), f"{name} kp={kp} ki={ki} kd={kd} alpha={alpha} beta={beta}", directory=cfg['save_dir'], subdir="mean")
    save_array(np.std(histories, axis=0), f"{name} kp={kp} ki={ki} kd={kd} alpha={alpha} beta={beta}", directory=cfg['save_dir'], subdir="std_dev")


if __name__ == "__main__":
    soft_policy_evaluation_experiment()