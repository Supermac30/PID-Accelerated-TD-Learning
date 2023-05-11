import matplotlib.pyplot as plt
import hydra

from Experiments.ExperimentHelpers import *
from Experiments.AgentBuilder import build_agent_and_env

@hydra.main(version_base=None, config_path="../../config/ComparisonExperiments", config_name="HardSoftComparison")
def hard_soft_convergence_experiment(cfg):
    """Experiment with the convergence rate of hard and soft derivative updates."""
    fig, (ax1, ax2) = plt.subplots(2)

    for kp, kd, ki, alpha, beta in zip(cfg['kp'], cfg['kd'], cfg['ki'], cfg['alpha'], cfg['beta']):
        soft, env, policy = build_agent_and_env(("TD", kp, ki, kd, alpha, beta), cfg['env'], cfg['get_optimal'], cfg['seed'], cfg['gamma'])
        V_pi = find_Vpi(env, policy, cfg['gamma'])
        test_function = build_test_function(cfg['norm'], V_pi)

        hard, env, policy = build_agent_and_env(("hard TD", kp, ki, kd, alpha, beta), cfg['env'], cfg['get_optimal'], cfg['seed'], cfg['gamma'])

        soft_history, _ = soft.estimate_value_function(num_iterations=cfg['num_iterations'], test_function=test_function, follow_trajectory=cfg['follow_trajectory'])
        hard_history, _ = hard.estimate_value_function(num_iterations=cfg['num_iterations'], test_function=test_function, follow_trajectory=cfg['follow_trajectory'])

        save_array(soft_history, f"soft kp={kp} kd={kd} ki={ki} alpha={alpha} beta={beta}", ax1)
        save_array(hard_history, f"hard kp={kp} kd={kd} ki={ki} alpha={alpha} beta={beta}", ax2)

    plot_comparison(fig, ax1, ax2, 'Soft Updates', 'Hard Updates', f"$||V_k - V^\pi||_{{{cfg['norm']}}}$")


if __name__ == '__main__':
    hard_soft_convergence_experiment()