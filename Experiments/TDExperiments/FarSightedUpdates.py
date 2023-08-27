import hydra

from Experiments.ExperimentHelpers import *
from TabularPID.AgentBuilders.AgentBuilder import build_agent_and_env

@hydra.main(version_base=None, config_path="../../config/TDExperiments", config_name="FarSightedUpdate")
def far_sighted_update_experiment(cfg):
    """Experiments with policy evaluation and TD"""
    seed = pick_seed(cfg['seed'])
    delay = cfg['delays']
    kp, kd, ki, alpha, beta = cfg['kp'], cfg['kd'], cfg['ki'], cfg['alpha'], cfg['beta']
    agent, env, policy = build_agent_and_env(("far sighted TD", kp, ki, kd, alpha, beta, delay), cfg['env'], cfg['get_optimal'], seed, cfg['gamma'])
    V_pi = find_Vpi(env, policy, cfg['gamma'])
    test_function = build_test_function(cfg['norm'], V_pi)
    all_histories = []
    for _ in range(cfg['repeat']):
        history, _ = agent.estimate_value_function(num_iterations=cfg['num_iterations'], test_function=test_function, follow_trajectory=cfg['follow_trajectory'])
        all_histories.append(history)
    mean_history = np.mean(np.array(all_histories), axis=0)
    std_dev_history = np.std(np.array(all_histories), axis=0)
    save_array(mean_history, f"kp={kp} ki={ki} kd={kd} alpha={alpha} beta={beta}", directory=cfg['save_dir'], subdir="mean")
    save_array(std_dev_history, f"kp={kp} ki={ki} kd={kd} alpha={alpha} beta={beta}", directory=cfg['save_dir'], subdir="std_dev")



if __name__ == "__main__":
    far_sighted_update_experiment()