"""
Notes about the hyperparameter tuning procedure:

- The seed is randomly chosen, i.e. not fixed. Hopefully, the results are not dependent on this. It may be worth running this again to corroborate the results.
- The parameters are turned with respect to minimizing the L1 norm, i.e. we take
               argmin_theta ||V_theta - V^pi||_1
- The learning rate functions used are min(c, N/(k + 1)), with a different function on each component
- We do not follow a trajectory, choosing instead to receive arbitrary samples from the environment.
"""

import logging

from Experiments.ExperimentHelpers import find_optimal_learning_rates, find_Vpi, build_test_function, find_Qstar, pick_seed
from TabularPID.AgentBuilders.AdaptiveAgentBuilder import build_adaptive_agent_and_env
from TabularPID.AgentBuilders.AgentBuilder import build_agent_and_env
from TabularPID.OptimalRates.OptimalRateDatabase import get_stored_optimal_rate, store_optimal_rate

default_rates = (0.1, 100, 0.1, 1, 0, float("inf"))

exhaustive_learning_rates = [
    {
        1: {10, 50, 100, 500, 1000, 10000},
    },
    {
        0.25: {float("inf"), 10, 100, 500, 1000, 10000},
        # 0.001: {float("inf")},
        0: {float("inf")},
    },
    {
        0.25: {float("inf"), 10, 100, 500, 1000, 10000},
        #0.001: {float("inf")},
        0: {float("inf")},
    }
]


def get_optimal_past_work_rates(agent_description, env_name, gamma, recompute=False, seed=-1, norm=1, search_steps=10000):
    optimal_rates = get_stored_optimal_rate((agent_description, 1, 0, 0, 0, 0), env_name, gamma)
    if optimal_rates is None or recompute:
        seed = pick_seed(seed)
        optimal_rates = run_past_work_search(agent_description, env_name, seed, norm, gamma, search_steps)
        store_optimal_rate((agent_description, 1, 0, 0, 0, 0), env_name, optimal_rates, gamma)
    
    logging.info(f"The optimal rates for {(env_name, gamma)} are: {optimal_rates}")

    return optimal_rates


def get_optimal_pid_rates(agent_description, env_name, kp, ki, kd, alpha, beta, gamma, recompute=False, seed=-1, norm=1, search_steps=10000):
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
        seed = pick_seed(seed)
        optimal_rates = run_pid_search(agent_description, env_name, kp, ki, kd, alpha, beta, seed, norm, gamma, search_steps)
        store_optimal_rate((agent_description, kp, ki, kd, alpha, beta), env_name, optimal_rates, gamma)

    logging.info(f"The optimal rates for {(env_name, kp, ki, kd)} are: {optimal_rates}")

    return optimal_rates


def get_optimal_pid_q_rates(agent_name, env_name, kp, ki, kd, alpha, beta, gamma, recompute=False, seed=-1, norm="fro", search_steps=10000):
    """Find the optimal rates for the choice of controller gains and environment.
    If this has been done before, get the optimal rates from the file of stored rates.

    If recompute is True, recompute the learning rates even if it is in the file of stored rates.
    """
    # To remove duplicates, if ki is zero the values of alpha and beta don't matter
    if ki == 0:
        alpha = 0.05
        beta = 0.95

    optimal_rates = get_stored_optimal_rate((agent_name, kp, ki, kd, alpha, beta), env_name, gamma)
    if optimal_rates is None or recompute:
        seed = pick_seed(seed)
        optimal_rates = run_pid_q_search(agent_name, env_name, kp, ki, kd, alpha, beta, seed, norm, gamma, search_steps)
        store_optimal_rate((agent_name, kp, ki, kd, alpha, beta), env_name, optimal_rates, gamma)

    logging.info(f"The optimal rates for {(agent_name, env_name, kp, ki, kd)} are: {optimal_rates}")

    return optimal_rates


def get_optimal_adaptive_rates(agent_name, env_name, meta_lr, gamma, lambd, delay, alpha, beta, recompute=False, seed=-1, norm=1, epsilon=0.01, is_q=False, search_steps=1000):
    """Find the optimal rates for the choice of adaptive agent and environment.
    If this has been done before, get the optimal rates from the file of stored rates.

    If recompute is True, recompute the learning rates even if it is in the file of stored rates.
    """
    optimal_rates = get_stored_optimal_rate((agent_name, meta_lr, lambd, delay, alpha, beta, epsilon), env_name, gamma)

    if optimal_rates is None or recompute:
        seed = pick_seed(seed)
        optimal_rates = run_adaptive_search(agent_name, env_name, seed, norm, gamma, lambd, delay, meta_lr, alpha, beta, epsilon, is_q, search_steps)
        store_optimal_rate((agent_name, meta_lr, lambd, delay, alpha, beta, epsilon), env_name, optimal_rates, gamma)

    logging.info(f"The optimal rates for {(env_name, agent_name, meta_lr, epsilon)} are: {optimal_rates}")

    return optimal_rates


def get_optimal_q_adaptive_rates(agent_name, env_name, meta_lr, gamma, lambd, delay, alpha, beta, recompute=False, seed=-1, norm=1, epsilon=0.01, is_q=False, search_steps=1000, meta_lr_p=None, meta_lr_I=None, meta_lr_d=None):
    """Find the optimal rates for the choice of adaptive agent and environment.
    If this has been done before, get the optimal rates from the file of stored rates.

    If recompute is True, recompute the learning rates even if it is in the file of stored rates.
    """
    optimal_rates = get_stored_optimal_rate((agent_name, meta_lr, lambd, delay, alpha, beta, epsilon, meta_lr_p, meta_lr_I, meta_lr_d), env_name, gamma)

    if optimal_rates is None or recompute:
        seed = pick_seed(seed)
        optimal_rates = run_adaptive_search(agent_name, env_name, seed, norm, gamma, lambd, delay, meta_lr, alpha, beta, epsilon, is_q, search_steps, meta_lr_p, meta_lr_I, meta_lr_d)
        store_optimal_rate((agent_name, meta_lr, lambd, delay, alpha, beta, epsilon, meta_lr_p, meta_lr_I, meta_lr_d), env_name, optimal_rates, gamma)

    logging.info(f"The optimal rates for {(env_name, agent_name, meta_lr, epsilon)} are: {optimal_rates}")

    return optimal_rates


def get_optimal_adaptive_linear_FA_rates(agent_name, env_name, order, meta_lr, gamma, lambd, delay, alpha, beta, recompute=False, seed=-1, norm=1, epsilon=0.01, search_steps=1000):
    """Find the optimal rates for the choice of adaptive agent and environment.
    If this has been done before, get the optimal rates from the file of stored rates.

    If recompute is True, recompute the learning rates even if it is in the file of stored rates.
    """
    optimal_rates = get_stored_optimal_rate((agent_name, meta_lr, lambd, delay, alpha, beta, epsilon, order), env_name, gamma)

    if optimal_rates is None or recompute:
        seed = pick_seed(seed)
        optimal_rates = run_adaptive_linear_FA_search(agent_name, env_name, meta_lr, lambd, delay, alpha, beta, epsilon, seed, gamma, order, search_steps)
        store_optimal_rate((agent_name, meta_lr, lambd, delay, alpha, beta, epsilon, order), env_name, optimal_rates, gamma)

    logging.info(f"The optimal rates for {(env_name, agent_name, meta_lr, epsilon)} are: {optimal_rates}")

    return optimal_rates


def get_optimal_linear_FA_rates(agent_name, env_name, kp, ki, kd, alpha, beta, gamma, order, recompute=False, seed=-1, norm=1, search_steps=10000):
    """Find the optimal rates for the choice of adaptive agent and environment.
    If this has been done before, get the optimal rates from the file of stored rates.

    If recompute is True, recompute the learning rates even if it is in the file of stored rates.
    """
    optimal_rates = get_stored_optimal_rate((agent_name, kp, ki, kd, alpha, beta, order), env_name, gamma)

    if optimal_rates is None or recompute:
        seed = pick_seed(seed)
        optimal_rates = run_linear_FA_search(agent_name, env_name, kp, ki, kd, alpha, beta, seed, gamma, order, search_steps)
        store_optimal_rate((agent_name, kp, ki, kd, alpha, beta, order), env_name, optimal_rates, gamma)

    logging.info(f"The optimal rates for {(agent_name, kp, ki, kd, alpha, beta, order)} are: {optimal_rates}")

    return optimal_rates


def run_adaptive_linear_FA_search(agent_name, env_name, meta_lr, lambd, delay, alpha, beta, epsilon, seed, gamma, order, search_steps=10000):
    """Run a grid search on the exhaustive learning rates for the choice of controller gains"""
    agent, _, _ = build_adaptive_agent_and_env(agent_name, env_name, meta_lr, lambd, delay, alpha=alpha, beta=beta, seed=seed, gamma=gamma, epsilon=epsilon, order=order)

    # Don't search over the learning rates for the components that are 0
    learning_rates = exhaustive_learning_rates[0]
    update_I_rates = exhaustive_learning_rates[1]
    update_D_rates = exhaustive_learning_rates[2]

    _, rates = find_optimal_learning_rates(
        agent,
        lambda: agent.estimate_value_function(
            num_iterations=search_steps,
            stop_if_diverging=True
        )[0],
        learning_rates,
        update_I_rates,
        update_D_rates,
        True
    )
    if rates is None:
        return default_rates
    return rates


def run_linear_FA_search(agent_name, env_name, kp, ki, kd, alpha, beta, seed, gamma, order, search_steps=10000):
    """Run a grid search on the exhaustive learning rates for the choice of controller gains"""
    agent, _, _ = build_agent_and_env((agent_name, kp, ki, kd, alpha, beta, order), env_name, seed=seed, gamma=gamma)

    # Don't search over the learning rates for the components that are 0
    if kp == 0:
        learning_rates = {0: {float("inf")}}
    else:
        learning_rates = exhaustive_learning_rates[0]
    if ki == 0:
        update_I_rates = {0: {float("inf")}}
    else:
        update_I_rates = exhaustive_learning_rates[1]
    if kd == 0:
        update_D_rates = {0: {float("inf")}}
    else:
        update_D_rates = exhaustive_learning_rates[2]

    _, rates = find_optimal_learning_rates(
        agent,
        lambda: agent.estimate_value_function(
            num_iterations=search_steps,
            stop_if_diverging=True
        )[0],
        learning_rates,
        update_I_rates,
        update_D_rates,
        True
    )
    if rates is None:
        return default_rates
    return rates


def run_pid_search(agent_description, env_name, kp, ki, kd, alpha, beta, seed, norm, gamma, search_steps=10000):
    """Run a grid search on the exhaustive learning rates for the choice of controller gains"""
    agent, env, policy = build_agent_and_env((agent_description, kp, ki, kd, alpha, beta), env_name, seed=seed, gamma=gamma)
    V_pi = find_Vpi(env, policy, gamma)

    # Don't search over the learning rates for the components that are 0
    if kp == 0:
        learning_rates = {0: {float("inf")}}
    else:
        learning_rates = exhaustive_learning_rates[0]
    if ki == 0:
        update_I_rates = {0: {float("inf")}}
    else:
        update_I_rates = exhaustive_learning_rates[1]
    if kd == 0:
        update_D_rates = {0: {float("inf")}}
    else:
        update_D_rates = exhaustive_learning_rates[2]

    return run_search(agent, norm, V_pi, search_steps, learning_rates, update_I_rates, update_D_rates, 0)


def run_past_work_search(agent_description, env_name, seed, norm, gamma, search_steps=50000):
    """Run a grid search on learning rates for any of the past work algorithms"""
    agent, env, policy = build_agent_and_env((agent_description, 1, 0, 0, 0, 0), env_name, seed=seed, gamma=gamma)
    V_pi = find_Vpi(env, policy, gamma)

    dummy = {1: {float("inf")}}
    if agent_description == "TIDBD":
        # Taken from the TIDBD paper, 200 values between 0 and 0.2
        # We add more thetas to account for more complex environments
        learning_rates0 = {i / 10000: {float("inf")} for i in range(2000)}
        # Set initial learning rates as in the TIDBD paper
        learning_rates1 = {0.0005: {float("inf")}, 0.0025: {float("inf")}, 0.005: {float("inf")}, 0.025: {float("inf")}, 0.05: {float("inf")}, 0.25: {float("inf")}, 0.5: {float("inf")}}
        learning_rates2 = dummy
    elif agent_description == "speedy Q learning":
        learning_rates0 = exhaustive_learning_rates[0]
        learning_rates1 = dummy
        learning_rates2 = dummy
    elif agent_description == "zap Q learning":
        learning_rates0 = {i / 100: {0} for i in range(50, 105, 5)}
        learning_rates1 = dummy
        learning_rates2 = dummy
        return (0.85, 0, 0, 0, 0, 0)

    else:
        raise ValueError(f"Unknown agent description: {agent_description}")

    return run_search(agent, norm, V_pi, search_steps, learning_rates0, learning_rates1, learning_rates2, 0)


def run_pid_q_search(agent_description, env_name, kp, ki, kd, alpha, beta, seed, norm, gamma, search_steps=50000):
    """Run a grid search on the exhaustive learning rates for the choice of controller gains"""
    agent, env, _ = build_agent_and_env((agent_description, kp, ki, kd, alpha, beta), env_name, seed=seed, gamma=gamma)
    Q_star = find_Qstar(env, gamma)

    # Don't search over the learning rates for the components that are 0
    if kp == 0:
        learning_rates = {0: {float("inf")}}
    else:
        learning_rates = exhaustive_learning_rates[0]
    if ki == 0:
        update_I_rates = {0: {float("inf")}}
    else:
        update_I_rates = exhaustive_learning_rates[1]
    if kd == 0:
        update_D_rates = {0: {float("inf")}}
    else:
        update_D_rates = exhaustive_learning_rates[2]

    return run_search(agent, norm, Q_star, search_steps, learning_rates, update_I_rates, update_D_rates, 0)


def run_adaptive_search(agent_name, env_name, seed, norm, gamma, lambd, delay, meta_lr, alpha, beta, epsilon, is_q, search_steps=50000, meta_lr_p=None, meta_lr_I=None, meta_lr_d=None):
    """Run a grid search on the exhaustive learning rates for the choice of adaptive agent"""
    agent, env, policy = build_adaptive_agent_and_env(agent_name, env_name, meta_lr, lambd, delay, alpha=alpha, beta=beta, seed=seed, gamma=gamma, epsilon=epsilon, meta_lr_p=meta_lr_p, meta_lr_I=meta_lr_I, meta_lr_d=meta_lr_d)
    if is_q:
        goal = find_Qstar(env, gamma)
    else:
        goal = find_Vpi(env, policy, gamma)
    
    # For now, only optimize the learning rate of the controller
    learning_rates = exhaustive_learning_rates[0]
    update_I_rates = exhaustive_learning_rates[1]
    update_D_rates = exhaustive_learning_rates[2]

    return run_search(agent, norm, goal, search_steps, learning_rates, update_I_rates, update_D_rates, 2)


def run_search(agent, norm, goal, search_steps, learning_rates, update_I_rates, update_D_rates, index):
    _, rates = find_optimal_learning_rates(
        agent,
        lambda: agent.estimate_value_function(
            num_iterations=search_steps,
            test_function=build_test_function(norm, goal),
            follow_trajectory=False,
            stop_if_diverging=True,
        )[index],
        learning_rates,
        update_I_rates,
        update_D_rates,
        verbose=True,
        repeat=20
    )
    if rates is None:
        return default_rates
    return rates