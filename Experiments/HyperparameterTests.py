"""
Notes about the hyperparameter tuning procedure:

- The seed is randomly chosen, i.e. not fixed. Hopefully, the results are not dependent on this. It may be worth running this again to corroborate the results.
- The parameters are turned with respect to minimizing the L1 norm, i.e. we take
               argmin_theta ||V_theta - V^pi||_1
- We tune after 10000 steps of training.
- The learning rate functions used are min(c, N/(k + 1)), with a different function on each component
- We do not follow a trajectory, choosing instead to receive arbitrary samples from the environment.
"""

from Experiments.ExperimentHelpers import find_optimal_learning_rates, find_Vpi, build_test_function, learning_rate_function
from Experiments.AdaptiveAgentBuilder import build_adaptive_agent_and_env
from Experiments.AgentBuilder import build_agent_and_env
import logging
from Experiments.OptimalRateDatabase import get_stored_optimal_rate, store_optimal_rate

exhaustive_learning_rates = [
    {
            1: {1, 10, 100, 250, 500, 750, 1000},
            0.5: {10, 100, 1000},
            0.25: {100, 1000},
            #0.1: {1, 10, 100, 1000},
            #0.05: {1, 10, 100, 1000},
            #0.01: {1, 10, 100, 1000}
        },
        {
            1: {float("inf"), 1, 10, 100, 1000},
            #0.5: {1, 10, 100, 1000},
            #0.1: {1, 10, 100, 1000},
            #0.01: {1, 10, 100, 1000}
        },
        {
            1: {float("inf"), 1, 10, 100, 1000},
            #0.5: {1, 10, 100, 1000},
            #0.1: {1, 10, 100, 1000},
            #0.01: {1, 10, 100, 1000}
        }
]

def get_optimal_pid_rates(agent_description, env_name, kp, ki, kd, alpha, beta, gamma, recompute=False, seed=-1, norm=1):
    """Find the optimal rates for the choice of controller gains and environment.
    If this has been done before, get the optimal rates from the file of stored rates.

    If recompute is True, recompute the learning rates even if it is in the file of stored rates.
    """
    # To remove duplicates, if ki is zero the values of alpha and beta don't matter
    if ki == 0:
        alpha = 0.05
        beta = 0.95
    optimal_rates = get_stored_optimal_rate((agent_description, kp, ki, kd, alpha, beta), env_name, gamma)
    if optimal_rates is None or recompute:
        optimal_rates = run_pid_search(agent_description, env_name, kp, ki, kd, alpha, beta, seed, norm, gamma)
        store_optimal_rate((agent_description, kp, ki, kd, alpha, beta), env_name, optimal_rates, gamma)

    logging.info(f"The optimal rates for {(env_name, kp, ki, kd)} are: {optimal_rates}")

    return optimal_rates

def get_optimal_pid_q_rates(agent_name, env_name, kp, ki, kd, alpha, beta, gamma, recompute=False, seed=-1, norm=1, decay=1):
    """Find the optimal rates for the choice of controller gains and environment.
    If this has been done before, get the optimal rates from the file of stored rates.

    If recompute is True, recompute the learning rates even if it is in the file of stored rates.
    """
    # To remove duplicates, if ki is zero the values of alpha and beta don't matter
    if ki == 0:
        alpha = 0.05
        beta = 0.95
    optimal_rates = get_stored_optimal_rate((agent_name, kp, ki, kd, alpha, beta, decay), env_name, gamma)
    if optimal_rates is None or recompute:
        optimal_rates = run_pid_q_search(agent_name, env_name, kp, ki, kd, alpha, beta, seed, norm, gamma, decay)
        store_optimal_rate((agent_name, kp, ki, kd, alpha, beta, decay), env_name, optimal_rates, gamma)

    logging.info(f"The optimal rates for {(agent_name, decay, env_name, kp, ki, kd)} are: {optimal_rates}")

    return optimal_rates

def get_optimal_adaptive_rates(agent_name, env_name, meta_lr, gamma, lambd, delay, recompute=False, seed=-1, norm=1):
    """Find the optimal rates for the choice of adaptive agent and environment.
    If this has been done before, get the optimal rates from the file of stored rates.

    If recompute is True, recompute the learning rates even if it is in the file of stored rates.
    """
    optimal_rates = get_stored_optimal_rate((agent_name, meta_lr, lambd, delay), env_name, gamma)

    if optimal_rates is None or recompute:
        optimal_rates = run_adaptive_search(agent_name, env_name, seed, norm, gamma, lambd, delay, meta_lr)
        store_optimal_rate((agent_name, meta_lr, lambd, delay), env_name, optimal_rates, gamma)

    logging.info(f"The optimal rates for {(env_name, agent_name, lambd, delay)} are: {optimal_rates}")

    return optimal_rates

def run_pid_search(agent_description, env_name, kp, ki, kd, alpha, beta, seed, norm, gamma):
    """Run a grid search on the exhaustive learning rates for the choice of controller gains"""
    agent, env, policy = build_agent_and_env((agent_description, kp, ki, kd, alpha, beta), env_name, seed=seed, gamma=gamma)
    V_pi = find_Vpi(env, policy, gamma)

    # Don't search over the learning rates for the components that are 0
    if kp == 0:
        learning_rates = {1: {float("inf")}}
    else:
        learning_rates = exhaustive_learning_rates[0]
    if ki == 0:
        update_I_rates = {1: {float("inf")}}
    else:
        update_I_rates = exhaustive_learning_rates[1]
    if kd == 0:
        update_D_rates = {1: {float("inf")}}
    else:
        update_D_rates = exhaustive_learning_rates[2]

    _, rates = find_optimal_learning_rates(
        agent,
        lambda: agent.estimate_value_function(
            num_iterations=30000,
            test_function=build_test_function(norm, V_pi),
            follow_trajectory=False
        )[0],
        learning_rates,
        update_I_rates,
        update_D_rates,
        True
    )
    return rates

def run_pid_q_search(agent_description, env_name, kp, ki, kd, alpha, beta, seed, norm, gamma, decay):
    """Run a grid search on the exhaustive learning rates for the choice of controller gains"""
    agent, env, policy = build_agent_and_env((agent_description, kp, ki, kd, alpha, beta, decay), env_name, seed=seed, gamma=gamma)
    V_pi = find_Vpi(env, policy, gamma)

    # Don't search over the learning rates for the components that are 0
    if kp == 0:
        learning_rates = {1: {float("inf")}}
    else:
        learning_rates = exhaustive_learning_rates[0]
    if ki == 0:
        update_I_rates = {1: {float("inf")}}
    else:
        update_I_rates = exhaustive_learning_rates[1]
    if kd == 0:
        update_D_rates = {1: {float("inf")}}
    else:
        update_D_rates = exhaustive_learning_rates[2]

    _, rates = find_optimal_learning_rates(
        agent,
        lambda: agent.estimate_value_function(
            num_iterations=30000,
            test_function=build_test_function(norm, V_pi),
            follow_trajectory=False
        )[0],
        learning_rates,
        update_I_rates,
        update_D_rates,
        True
    )
    return rates

def run_adaptive_search(agent_name, env_name, seed, norm, gamma, lambd, delay, meta_lr):
    """Run a grid search on the exhaustive learning rates for the choice of adaptive agent"""
    agent, env, policy = build_adaptive_agent_and_env(agent_name, env_name, meta_lr, lambd, delay, seed=seed, gamma=gamma)
    V_pi = find_Vpi(env, policy, gamma)

    # For now, only optimize the learning rate of the controller
    learning_rates = exhaustive_learning_rates[0]
    update_I_rates = exhaustive_learning_rates[1]
    update_D_rates = exhaustive_learning_rates[2]

    _, rates = find_optimal_learning_rates(
        agent,
        lambda: agent.estimate_value_function(
            num_iterations=30000,
            test_function=build_test_function(norm, V_pi),
            follow_trajectory=False
        )[2],
        learning_rates,
        update_I_rates,
        update_D_rates,
        True
    )

    return rates

if __name__ == '__main__':
    logging.basicConfig()
    logging.root.setLevel(logging.NOTSET)

    for name in {"chain walk", "cliff walk"}:
        for gamma in {0.99, 0.999, 0.9999}:
            get_optimal_pid_rates("TD", name, 1, 0, 0, 0, 0, gamma, recompute=True)
