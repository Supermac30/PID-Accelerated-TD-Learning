import matplotlib.pyplot as plt
import numpy as np
import hydra

from Experiments.ExperimentHelpers import *
from Experiments.AgentBuilder import build_agent_and_env

from Controllers import Adam_Controller, Adagrad_Controller

@hydra.main(version_base=None, config_path="../../config/NovelControllerExperiments", config_name="NovelControllerExperiment")
def adam_controller_experiment(cfg):
    """Experiments with policy evaluation and TD"""
    agent, env, policy = build_agent_and_env(("hard TD", 1, 0, 0, 0, 0), cfg['env'], cfg['get_optimal'], cfg['seed'], cfg['gamma'])
    num_states = env.num_states

    V_pi = find_Vpi(env, policy, cfg['gamma'])
    test_function = build_test_function(cfg['norm'], V_pi)

    if cfg['type'] == 'Adam':
        novel_controller = Adam_Controller(cfg['ka'] * np.identity(num_states), cfg['beta1'], cfg['beta2'], cfg['epsilon'])
    elif cfg['type'] == "Adagrad":
        novel_controller = Adagrad_Controller(cfg['ka'] * np.identity(num_states))

    def estimate_value():
        total_history = 0
        for _ in range(10):
            history, _ = agent.estimate_value_function(
                novel_controller,
                num_iterations=cfg['num_iterations'],
                test_function=test_function,
                follow_trajectory=cfg['follow_trajectory']
            )
            total_history += history
        return total_history / 10

    history, params = find_optimal_learning_rates(
        agent,
        lambda: estimate_value(),
        True,
        learning_rates=cfg['learning_rates'],
        update_D_rates=cfg['update_D_rates'],
        update_I_rates=cfg['update_I_rates'],
        verbose=False
    )
    save_array(history, f"{cfg['type']} {params}", plt)


    agent, env, policy = build_agent_and_env(("TD", 1, 0, 0, 0, 0), cfg['env'], cfg['get_optimal'], cfg['seed'], cfg['gamma'])
    history, _ = agent.estimate_value_function(num_iterations=cfg['num_iterations'], test_function=test_function, follow_trajectory=cfg['follow_trajectory'])
    save_array(history, f"Regular TD", plt)

    plt.title(f"{cfg['type']}: {cfg['env']}")
    plt.legend()
    plt.title(f"{cfg['type']}")
    plt.xlabel('Iteration')
    plt.ylabel(f"$||V_k - V^\pi||_{{{cfg['norm']}}}$")
    plt.savefig("plot")
    plt.show()


if __name__ == "__main__":
    adam_controller_experiment()