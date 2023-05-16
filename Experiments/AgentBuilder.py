from Experiments.OptimalRateDatabase import get_stored_optimal_rate
from Experiments.ExperimentHelpers import get_env_policy, learning_rate_function
from Agents import PID_TD, FarSighted_PID_TD, Hard_PID_TD, PID_TD_with_momentum, ControlledQLearning
from MDP import PolicyEvaluation, Control, Control_Q
import logging

"""
Types of agents:
- VI (kp, kd, ki): The agent that uses PID to learn the optimal policy with VI
- TD (kp, kd, ki): The agent that uses PID to learn the optimal policy with TD
"""

default_optimal_rates = (1, 20, 1, float("inf"), 1, float("inf"))

def build_agent_and_env(agent_name, env_name, get_optimal=False, seed=-1, gamma=0.99):
    """Return both the agent and the environment & policy given their names.
    agent_name is a tuple of the form ("TD" or "VI", kp, ki, kd, alpha, beta, *kwargs),
    where kp, ki, kd, alpha, beta are floats, and **kwargs are any special keyword arguments for the agent.

    get_optimal: Try to find the optimal learning rate and set it
    Return None is the agent name is not found.
    """
    env, policy = get_env_policy(env_name, seed)
    return build_agent(agent_name, env_name, env, policy, get_optimal, gamma), env, policy

def build_agent(agent_name, env_name, env, policy, get_optimal, gamma):
    """Return the agent given its name.
    agent_name is a tuple of the form (agent_description, kp, ki, kd, alpha, beta, *kwargs),
    where kp, kd, ki, alpha, beta are floats, and **kwargs are any special keyword arguments for the agent.

    where the agent_description is one of the following:
    - "VI": The agent that uses PID to learn the optimal policy with VI
    - "VI control": The agent that uses PID to learn the optimal policy with VI
    - "TD": The agent that uses PID to learn the optimal policy with TD
    - "far sighted TD": The agent that uses Far Sighted PID to learn the optimal policy with TD. kwargs is the delay.
    - "hard TD": The agent that uses hard updates with PID TD
    - "momentum TD": The agent that uses momentum with PID TD
    - "Q learning": The agent that uses PID to learn the optimal policy with Q learning

    get_optimal: Try to find the optimal learning rate and set it

    Return None is the agent name is not found.
    """
    agent_description, kp, ki, kd, alpha, beta, *kwargs = agent_name

    if agent_description == "VI":
        return build_VI_PID(env, policy, kp, ki, kd, alpha, beta, gamma)
    elif agent_description == "VI control":
        return build_VI_Control_PID(env, kp, ki, kd, alpha, beta, gamma)
    elif agent_description == "VI Q control":
        return build_VI_Q_Control_PID(env, kp, ki, kd, alpha, beta, gamma)

    if get_optimal:
        optimal_rates = get_stored_optimal_rate(agent_name, env_name, gamma)
    if not get_optimal or optimal_rates is None:
        optimal_rates = default_optimal_rates

    learning_rate = learning_rate_function(optimal_rates[0], optimal_rates[1])
    update_I_rate = learning_rate_function(optimal_rates[2], optimal_rates[3])
    update_D_rate = learning_rate_function(optimal_rates[4], optimal_rates[5])

    learning_rates = (learning_rate, update_I_rate, update_D_rate)

    logging.info(f"Using rates {optimal_rates} for agent {agent_name} on env {env_name}")

    if agent_description == "TD":
        return build_TD_PID(env, policy, kp, ki, kd, alpha, beta, learning_rates, gamma)
    elif agent_description == "far sighted TD":
        delay = kwargs[0]
        return build_FarSighted_TD_PID(env, policy, kp, ki, kd, alpha, beta, learning_rates, gamma, delay)
    elif agent_description == "hard TD":
        return build_hard_TD_PID(env, policy, kp, ki, kd, alpha, beta, learning_rates, gamma)
    elif agent_description == "momentum TD":
        return build_momentum_TD_PID(env, policy, kp, ki, kd, alpha, beta, learning_rates, gamma)
    elif agent_description == "Q learning":
        decay = kwargs[0]
        return build_Q_PID(env, kp, ki, kd, alpha, beta, learning_rates, gamma, decay)
    return None


def build_Q_PID(env, kp, ki, kd, alpha, beta, learning_rates, gamma, decay):
    """Build the Q agent with PID
    """
    return ControlledQLearning(
        env,
        gamma,
        kp, ki, kd, alpha, beta,
        learning_rates,
        decay
    )

def build_VI_Q_Control_PID(env, kp, ki, kd, alpha, beta, gamma):
    """Build the VI Q Control agent with PID
    """
    # Build the reward and transition matrices
    reward = env.build_reward_matrix()
    transition = env.build_probability_transition_kernel()

    return Control_Q(
        env.num_states,
        env.num_actions,
        reward,
        transition,
        gamma,
        kp, ki, kd, alpha, beta
    )



def build_momentum_TD_PID(env, policy, kp, ki, kd, alpha, beta, learning_rates, gamma):
    """Build the momentum TD agent with PID
    """
    return PID_TD_with_momentum(
        env,
        policy,
        gamma,
        kp, ki, kd, alpha, beta,
        learning_rates
    )


def build_hard_TD_PID(env, policy, kp, ki, kd, alpha, beta, learning_rates, gamma):
    """Build the hard TD agent with PID
    """
    return Hard_PID_TD(
        env,
        policy,
        gamma,
        kp, ki, kd, alpha, beta,
        learning_rates
    )

def build_FarSighted_TD_PID(env, policy, kp, ki, kd, alpha, beta, learning_rates, gamma, delay):
    """Build the Far Sighted TD agent with PID
    """
    return FarSighted_PID_TD(
        env,
        policy,
        gamma,
        kp, ki, kd, alpha, beta,
        learning_rates,
        delay
    )

def build_VI_Control_PID(env, kp, ki, kd, alpha, beta, gamma):
    """Build the VI Control agent with PID
    """
    # Build the reward and transition matrices
    reward = env.build_reward_matrix()
    transition = env.build_probability_transition_kernel()

    # Build the agent
    return Control(
        env.num_states,
        env.num_actions,
        reward,
        transition,
        kp, ki, kd, alpha, beta,
        gamma
    )

def build_VI_PID(env, policy, kp, ki, kd, alpha, beta, gamma):
    """Build the VI agent with PID
    """
    # Build the reward and transition matrices
    reward = env.build_policy_reward_vector(policy)
    transition = env.build_policy_probability_transition_kernel(policy)

    # Build the agent
    return PolicyEvaluation(
        env.num_states,
        env.num_actions,
        reward,
        transition,
        kp, ki, kd, alpha, beta,
        gamma
    )

def build_TD_PID(env, policy, kp, ki, kd, alpha, beta, learning_rates, gamma):
    """Build the TD agent with PID
    """
    return PID_TD(
        env,
        policy,
        gamma,
        kp, ki, kd, alpha, beta,
        learning_rates
    )