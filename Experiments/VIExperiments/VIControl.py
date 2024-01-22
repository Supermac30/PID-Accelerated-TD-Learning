import hydra

from Experiments.ExperimentHelpers import *
from TabularPID.AgentBuilders.AgentBuilder import build_agent_and_env

@hydra.main(version_base=None, config_path="../../config/VIExperiments", config_name="VIControl")
def control_experiment(cfg):
    """Attempt to replicate results in figure 2 of PID Accelerated VI"""
    seed = pick_seed(cfg['seed'])
    kp, kd, ki, alpha, beta = cfg['kp'], cfg['kd'], cfg['ki'], cfg['alpha'], cfg['beta']

    agent, env, _ = build_agent_and_env(("VI control", kp, ki, kd, alpha, beta), cfg['env'], False, seed, cfg['gamma'])
    V_star = find_Vstar(env, cfg['gamma'])
    history, _ = agent.value_iteration(num_iterations=cfg['num_iterations'], test_function=build_test_function(cfg['norm'], V_star))
    name = cfg['name']
    save_array(history, f"{name}", directory=cfg['save_dir'])


if __name__ == "__main__":
    control_experiment()