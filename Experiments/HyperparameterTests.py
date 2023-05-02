"""
Notes about the hyperparameter tuning procedure:

- The seed is randomly chosen, i.e. not fixed. Hopefully, the results are not dependent on this. It may be worth running this again to corroborate the results.
- The parameters are turned with respect to minimizing the L1 norm, i.e. we take
               argmin_theta ||V_theta - V^pi||_1
- We tune after 10000 steps of training.
- The learning rate functions used are min(c, N/(k + 1)), with a different function on each component
"""

from Experiments.ExperimentHelpers import find_optimal_learning_rates, find_optimal_pid_learning_rates, get_env_policy, find_Vpi, build_test_function, learning_rate_function
from Experiments.AdaptiveAgentBuilder import build_adaptive_agent
from Agents import ControlledTDLearning
import pickle

FILE_NAME = "Experiments/optimal_learning_rates.pickle"

exhaustive_learning_rates = [
    {
            1: {1, 10, 100, 1000, 10000},
            0.5: {1, 10, 100, 1000, 10000},
            0.25: {1, 10, 100, 1000, 10000},
            0.1: {1, 10, 100, 1000, 10000},
            0.05: {1, 10, 100, 1000, 10000},
            0.01: {1, 10, 100, 1000, 10000}
        },

        {
            1: {float("inf"), 1, 10, 100, 1000, 10000},
            0.5: {1, 10, 100, 1000, 10000},
            0.25: {1, 10, 100, 1000, 10000},
            0.1: {1, 10, 100, 1000, 10000},
            0.05: {1, 10, 100, 1000, 10000},
            0.01: {1, 10, 100, 1000, 10000}
        },
        {
            1: {float("inf"), 1, 10, 100, 1000, 10000},
            0.5: {1, 10, 100, 1000, 10000},
            0.25: {1, 10, 100, 1000, 10000},
            0.1: {1, 10, 100, 1000, 10000},
            0.05: {1, 10, 100, 1000, 10000},
            0.01: {1, 10, 100, 1000, 10000}
        }
]

def get_optimal_pid_rates(env_name, kp, kd, ki):
    """Find the optimal rates for the choice of controller gains and environment.
    If this has been done before, get the optimal rates from the file of stored rates.
    """
    optimal_rates = get_stored_optimal_rate((kp, ki, kd), env_name)
    if optimal_rates is None:
        optimal_rates = run_pid_search(env_name, kp, kd, ki, -1, 1)

    return optimal_rates

def find_optimal_adaptive_rates(agent_name, env_name):
    """Find the optimal rates for the choice of adaptive agent and environment.
    If this has been done before, get the optimal rates from the file of stored rates.
    """
    optimal_rates = get_stored_optimal_rate(agent_name, env_name)
    if optimal_rates is None:
        optimal_rates = run_adaptive_search(agent_name, env_name, -1, 1)

    return optimal_rates

def run_pid_search(env_name, kp, kd, ki, seed, norm):
    """Run a grid search on the exhaustive learning rates for the choice of controller gains"""
    env, policy = get_env_policy(env_name, seed)
    V_pi = find_Vpi(env, policy)
    agent = ControlledTDLearning(
        env,
        policy,
        0.99,
        learning_rate_function(1, 1)
    )
    return find_optimal_pid_learning_rates(
        agent,
        kp,
        kd,
        ki,
        build_test_function(norm, V_pi),
        10000,
        True,
        *exhaustive_learning_rates,
        True
    )

def run_adaptive_search(agent_name, env_name, seed, norm):
    """Run a grid search on the exhaustive learning rates for the choice of adaptive agent"""
    env, policy = get_env_policy(env_name, seed)
    V_pi = find_Vpi(env, policy)
    agent = build_adaptive_agent(agent_name, env_name)

    return find_optimal_learning_rates(
        agent,
        agent.estimate_value_function(
            10000,
            build_test_function(norm, V_pi)
        ),
        True,
        *exhaustive_learning_rates,
        True
    )

def get_stored_optimal_rate(model, env_name):
    """If the optimal rate is in FILE_NAME then return it,
    otherwise return None

    model: A consistent description of the model name, either a string or tuple
    env_name: A consistent description of the environment, a string

    If the file is not found, raise a FileNotFoundException
    """
    with open(FILE_NAME, 'rb') as f:
        optimal_rates = pickle.load(f)
    if (model, env_name) in optimal_rates:
        return optimal_rates[(model, env_name)]
    return None


def put_optimal_rate(model, env_name, optimal_rate):
    """Store the optimal rate in FILE_NAME.

    model: A consistent description of the model name, either a string or tuple
    env_name: A consistent description of the environment, a string
    optimal_rate: A tuple of learning rates for each component

    If the file is not found, raise a FileNotFoundException
    """
    with open(FILE_NAME, 'rb') as f:
        optimal_rates = pickle.load(f)
        optimal_rates[(model, env_name)] = optimal_rate
        pickle.dump(optimal_rate, FILE_NAME)


if __name__ == '__main__':
    import logging
    logging.basicConfig()
    logging.root.setLevel(logging.NOTSET)
    get_optimal_pid_rates("chain walk", 1, 0, 0)