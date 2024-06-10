import hydra

from TabularPID.AgentBuilders.AdaptiveAgentBuilder import build_adaptive_agent_and_env
from Experiments.ExperimentHelpers import *

@hydra.main(version_base=None, config_path="../../config/AdaptationExperiments", config_name="AdaptiveAgentPAVIAExperiment")
def adaptive_agent_experiment(cfg):
    """Visualize the behavior of adaptation without learning rates."""
    seed = pick_seed(cfg['seed'])
    agent, env, _ = build_adaptive_agent_and_env(cfg['agent_name'], cfg['env'], cfg['meta_lr'], 0, 1, seed=seed, gamma=cfg['gamma'])

    V_star = find_Vstar(env, cfg['gamma'])
    test_function = build_test_function(cfg['norm'], V_star)

    all_histories = []
    all_gain_histories = []
    for _ in range(cfg['repeat']):
        _, gain_history, history = agent.estimate_value_function(
            cfg['num_iterations'],
            test_function,
            follow_trajectory=cfg['follow_trajectory']
        )
        all_histories.append(history)
        all_gain_histories.append(gain_history)

    agent.gain_updater.plot()

    mean_history = np.mean(np.array(all_histories), axis=0)
    std_deviation_history = np.std(np.array(all_histories), axis=0)
    mean_gain_history = np.mean(np.array(all_gain_histories), axis=0)
    std_deviation_gain_history = np.std(np.array(all_gain_histories), axis=0)

    name = cfg['name']
    save_array(mean_history, f"{name}", directory=cfg['save_dir'], subdir="mean")
    save_array(std_deviation_history, f"{name}", directory=cfg['save_dir'], subdir="std_dev")
    save_array(mean_gain_history, f"gain_history {name}", directory=cfg['save_dir'], subdir="mean")
    save_array(std_deviation_gain_history, f"gain_history {name}", directory=cfg['save_dir'], subdir="std_dev")


if __name__ == '__main__':
    adaptive_agent_experiment()