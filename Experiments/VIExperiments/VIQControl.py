import hydra

from Experiments.ExperimentHelpers import *
from TabularPID.AgentBuilders.AgentBuilder import build_agent_and_env

@hydra.main(version_base=None, config_path="../../config/VIExperiments", config_name="VIQControl")
def control_experiment(cfg):
    seed = pick_seed(cfg['seed'])
    kp, kd, ki, alpha, beta = cfg['kp'], cfg['kd'], cfg['ki'], cfg['alpha'], cfg['beta']
    agent, env, _ = build_agent_and_env(("VI Q control", kp, ki, kd, alpha, beta), cfg['env'], False, seed, cfg['gamma'])
    Q_star = find_Qstar(env, cfg['gamma'])
    history, _ = agent.value_iteration(num_iterations=cfg['num_iterations'], test_function=build_test_function(cfg['norm'], Q_star))
    save_array(history, f"kp={kp} kd={kd} ki={ki} alpha={alpha} beta={beta}", directory=cfg['save_dir'])

if __name__ == "__main__":
    control_experiment()