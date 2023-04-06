import matplotlib.pyplot as plt
import numpy as np
import hydra

from Environments import ChainWalk
from Agents import ControlledTDLearning
from Experiments.ExperimentHelpers import *

from Controllers import Adam_Controller, P_Controller, Adagrad_Controller

@hydra.main(version_base=None, config_path="../../config/NovelControllerExperiments", config_name="NovelControllerExperiment")
def adam_controller_experiment(cfg):
    """Experiments with policy evaluation and TD"""
    env, policy = get_env_policy(cfg['env'], env['seed'])
    num_states = env.num_states
    agent = ControlledTDLearning(
        env,
        policy,
        0.99,
        learning_rate_function(1, 0)
    )

    V_pi = find_Vpi(env, policy)
    test_function = build_test_function(cfg['norm'], V_pi)

    if cfg['type'] == 'Adam':
        novel_controller = Adam_Controller(cfg['ka'] * np.identity(num_states), cfg['beta1'], cfg['beta2'], cfg['epsilon'])
    elif cfg['type'] == "Adagrad":
        novel_controller = Adagrad_Controller(cfg['ka'] * np.identity(num_states))
    p_controller = P_Controller(np.identity(num_states))

    def estimate_value():
        total_history = 0
        for _ in range(10):
            history, _ = agent.estimate_value_function(
                novel_controller,
                p_controller,
                num_iterations=cfg['num_iterations'],
                test_function=test_function
            )
            total_history += history
        return total_history / 10

    history, params = find_optimal_learning_rates(
        agent,
        lambda: estimate_value(),
        True,
        learning_rates=cfg['learning_rates'],
        update_D_rates=cfg['update_D_rates'],
        update_I_rates=cfg['update_I_rates']
    )
    save_array(history, f"{cfg['type']} {params}", plt)

    regular_history, regular_params = find_optimal_pid_learning_rates(agent, 1, 0, 0, test_function, cfg['num_iterations'], False)
    save_array(regular_history, f"Regular TD {regular_params}", plt)

    plt.title(f"{cfg['type']}: {cfg['env']}")
    plt.legend()
    plt.title(f"{cfg['type']}")
    plt.xlabel('Iteration')
    plt.ylabel('$||V_k - V^\pi||_1$')
    plt.savefig("plot")
    plt.show()


if __name__ == "__main__":
    adam_controller_experiment()