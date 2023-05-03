"""
Notes about the hyperparameter tuning procedure:

- The seed is randomly chosen, i.e. not fixed. Hopefully, the results are not dependent on this. It may be worth running this again to corroborate the results.
- The parameters are turned with respect to minimizing the L1 norm, i.e. we take
               argmin_theta ||V_theta - V^pi||_1
- We tune after 10000 steps of training.
- The learning rate functions used are min(c, N/(k + 1)), with a different function on each component
"""

from Experiments.ExperimentHelpers import find_optimal_learning_rates, find_Vpi, build_test_function, learning_rate_function
from Experiments.AdaptiveAgentBuilder import build_adaptive_agent_and_env
from Experiments.AgentBuilder import build_agent_and_env
from Agents import PID_TD
import logging
from Experiments.OptimalRateDatabase import get_stored_optimal_rate, store_optimal_rate

exhaustive_learning_rates = [
    {
            1: {1, 10, 100, 1000},
            0.5: {1, 10, 100, 1000},
            0.25: {1, 10, 100, 1000},
            0.1: {1, 10, 100, 1000},
            0.05: {1, 10, 100, 1000},
            0.01: {1, 10, 100, 1000}
        },

        {
            1: {float("inf"), 1, 10, 100, 1000},
            0.5: {1, 10, 100, 1000},
            0.1: {1, 10, 100, 1000},
            0.01: {1, 10, 100, 1000}
        },
        {
            1: {float("inf"), 1, 10, 100, 1000},
            0.5: {1, 10, 100, 1000},
            0.1: {1, 10, 100, 1000},
            0.01: {1, 10, 100, 1000}
        }
]

exhaustive_meta_lr = [
    0.1,
    0.01,
    0.001,
    0.0001
]

def get_optimal_pid_rates(env_name, kp, kd, ki, alpha, beta, gamma, recompute=False):
    """Find the optimal rates for the choice of controller gains and environment.
    If this has been done before, get the optimal rates from the file of stored rates.

    If recompute is True, recompute the learning rates even if it is in the file of stored rates.
    """
    optimal_rates = get_stored_optimal_rate((kp, ki, kd), env_name, gamma)
    if optimal_rates is None or recompute:
        optimal_rates = run_pid_search(env_name, kp, kd, ki, alpha, beta, -1, 1, gamma)
        store_optimal_rate((kp, ki, kd), env_name, optimal_rates)

    logging.info(f"The optimal rates for {(env_name, kp, kd, ki)} are: {optimal_rates}")

    return optimal_rates

def get_optimal_adaptive_rates(agent_name, env_name, gamma, recompute=False):
    """Find the optimal rates for the choice of adaptive agent and environment.
    If this has been done before, get the optimal rates from the file of stored rates.

    If recompute is True, recompute the learning rates even if it is in the file of stored rates.
    """
    optimal_rates = get_stored_optimal_rate(agent_name, env_name, gamma)
    if optimal_rates is None or recompute:
        optimal_rates = run_adaptive_search(agent_name, env_name, -1, 1, gamma, meta_lr=set_meta_lr(agent_name))
        store_optimal_rate(agent_name, env_name, optimal_rates, gamma)

    logging.info(f"The optimal rates for {(env_name, agent_name)} are: {optimal_rates}")

    return optimal_rates

def run_pid_search(env_name, kp, kd, ki, alpha, beta, seed, norm, gamma):
    """Run a grid search on the exhaustive learning rates for the choice of controller gains"""
    agent, env, policy = build_agent_and_env(("TD", kp, ki, kd, alpha, beta), env_name, seed=seed, gamma=gamma)
    V_pi = find_Vpi(env, policy)
    _, rates = find_optimal_learning_rates(
        agent,
        lambda: agent.estimate_value_function(
            num_iterations=10000,
            test_function=build_test_function(norm, V_pi)
        ),
        True,
        *exhaustive_learning_rates[0],
        True
    )
    return rates

def run_adaptive_search(agent_name, env_name, seed, norm, gamma, meta_lr=False):
    """Run a grid search on the exhaustive learning rates for the choice of adaptive agent

    If meta_lr is True, run a grid search on exhaustive_meta_lr. Otherwise, set it to 1.
    """
    if meta_lr:
        search = [1]
    else:
        search = exhaustive_meta_lr

    optimal_value, optimal_rates = float("inf"), None
    for meta_lr in search:
        agent, env, policy = build_adaptive_agent_and_env(agent_name, env_name, meta_lr_value=meta_lr, seed=seed, gamma=gamma)
        V_pi = find_Vpi(env, policy)

        history, rates = find_optimal_learning_rates(
            agent,
            lambda: agent.estimate_value_function(
                num_iterations=10000,
                test_function=build_test_function(norm, V_pi)
            ),
            True,
            *exhaustive_learning_rates,
            True
        )

        if history[-1] < optimal_value:
            optimal_value, optimal_rates = history[-1], rates

    return optimal_rates

def set_meta_lr(agent_name):
    """Return True if meta_lr should be set for this agent_name, and otherwise return False.
    That is, simply implement a look up table that returns whether the meta learning rate actually
    matters, or can simply be absorbed into the learning rate.
    """
    # TODO:
    return False

if __name__ == '__main__':
    logging.basicConfig()
    logging.root.setLevel(logging.NOTSET)

    pid_tests = [
        (1, 0, 0, 0.05, 0.95),
        (1, 0.1, 0, 0.05, 0.95),
        (1, 0.2, 0, 0.05, 0.95),
        (1, 0.3, 0, 0.05, 0.95),
        (1, 0.4, -0.2, 0.05, 0.95)
    ]

    for (kp, kd, ki, alpha, beta) in pid_tests:
        get_optimal_pid_rates("chain walk", kp, kd, ki, alpha, beta, gamma)