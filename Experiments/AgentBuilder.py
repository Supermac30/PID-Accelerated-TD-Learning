from Experiments.OptimalRateDatabase import get_stored_optimal_rate
from Experiments.ExperimentHelpers import get_env_policy, learning_rate_function
from Agents import PID_TD
from MDP import PolicyEvaluation

"""
Types of agents:
- VI (kp, kd, ki): The agent that uses PID to learn the optimal policy with VI
- TD (kp, kd, ki): The agent that uses PID to learn the optimal policy with TD
"""

default_learning_rates = (
    learning_rate_function(1, 1),
    learning_rate_function(1, 1),
    learning_rate_function(1, 1)
)

def build_agent_and_env(agent_name, env_name, get_optimal=False, seed=-1, gamma=0.99):
    """Return both the agent and the environment & policy given their names.
    agent_name is a tuple of the form ("TD" or "VI", kp, kd, ki), where kp, kd, ki are floats.

    get_optimal: Try to find the optimal learning rate and set it
    Return None is the agent name is not found.
    """
    env, policy = get_env_policy(env_name, seed)
    return build_agent(agent_name, env_name, env, policy, get_optimal, gamma), env, policy

def build_agent(agent_name, env_name, env, policy, get_optimal, gamma):
    """Return the agent given its name.
    agent_name is a tuple of the form ("TD" or "VI", kp, kd, ki), where kp, kd, ki are floats.

    get_optimal: Try to find the optimal learning rate and set it

    Return None is the agent name is not found.
    """
    agent_name, kp, kd, ki = agent_name

    if agent_name == "VI":
        return build_VI_PID(env, policy, kp, kd, ki, gamma)
    elif agent_name == "TD":
        if get_optimal:
            learning_rates = get_stored_optimal_rate(agent_name, env_name)
        if not get_optimal or learning_rates is None:
            learning_rates = default_learning_rates

        return build_TD_PID(env, policy, kp, kd, ki, learning_rates, gamma)
    return None

def build_VI_PID(env, policy, kp, kd, ki, alpha, beta, gamma):
    """Build the VI agent with PID

    The code needs to be refactored to allow for the gains to be passed in.
    """
    # Build the reward and transition matrices
    reward = env.build_policy_reward_matrix(policy)
    transition = env.build_policy_probability_transition_kernel(policy)

    # Build the agent
    return PolicyEvaluation(
        env.num_states,
        env.num_actions,
        reward,
        transition,
        kp,
        ki,
        kd,
        alpha,
        beta,
        gamma
    )

def build_TD_PID(env, policy, kp, kd, ki, alpha, beta, learning_rates, gamma):
    """Build the TD agent with PID

    The code needs to be refactored to allow for the gains to be passed in.
    """
    # Build the agent
    return PID_TD(
        env,
        policy,
        gamma,
        kp,
        kd,
        ki,
        alpha,
        beta,
        learning_rates
    )