import matplotlib.pyplot as plt
import hydra

from TabularPID.AgentBuilders.AgentBuilder import build_agent_and_env
from Experiments.ExperimentHelpers import *

@hydra.main(version_base=None, config_path="../../config/VIExperiments", config_name="VIPolicyEvaluation")
def policy_evaluation_experiment(cfg):
    """Attempt to replicate results in figure 1 of PID Accelerated VI"""
    seed = pick_seed(cfg['seed'])
    kp, kd, ki, alpha, beta = cfg['kp'], cfg['kd'], cfg['ki'], cfg['alpha'], cfg['beta']
    agent, env, policy = build_agent_and_env(("VI", kp, ki, kd, alpha, beta), cfg['env'], False, seed, cfg['gamma'])
    history, _ = agent.value_iteration(
        num_iterations=cfg['num_iterations'],
        test_function=build_test_function(cfg['norm'], find_Vpi(env, policy, cfg['gamma']))
    )
    save_array(history, f"kp={kp} kd={kd} ki={ki} alpha={alpha} beta={beta}", directory=cfg['save_dir'])

if __name__ == "__main__":
    policy_evaluation_experiment()