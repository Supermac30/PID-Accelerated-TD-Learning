from TabularPID.OptimalRates.OptimalRateDatabase import get_stored_optimal_rate
from TabularPID.EmpericalTester import build_emperical_TD_tester, build_emperical_Q_tester
from TabularPID.AgentBuilders.EnvBuilder import get_env_policy
from TabularPID.Agents.Agents import PID_TD, FarSighted_PID_TD, Hard_PID_TD, PID_TD_with_momentum, ControlledQLearning, learning_rate_function, ControlledDoubleQLearning
from TabularPID.Agents.LinearFA import LinearTD, FourierBasis, PolynomialBasis, TileCodingBasis, TrivialBasis
from TabularPID.Agents.LinearQ import LinearTDQ
from TabularPID.MDPs.MDP import PolicyEvaluation, Control, Control_Q
from TabularPID.Agents.Comparison.TIDBD import TIDBD
from TabularPID.Agents.Comparison.SpeedyQLearning import SpeedyQLearning
from TabularPID.Agents.Comparison.ZapQLearning import ZapQLearning

import logging

"""
Types of agents:
- VI (kp, kd, ki): The agent that uses PID to learn the optimal policy with VI
- TD (kp, kd, ki): The agent that uses PID to learn the optimal policy with TD
"""

default_optimal_rates = (0.0001, float("inf"), 0, float("inf"), 0, float("inf"))

def build_agent_and_env(agent_name, env_name, get_optimal=False, seed=42, gamma=0.99):
    """Return both the agent and the environment & policy given their names.
    agent_name is a tuple of the form ("TD" or "VI", kp, ki, kd, alpha, beta, *kwargs),
    where kp, ki, kd, alpha, beta are floats, and **kwargs are any special keyword arguments for the agent.

    get_optimal: Try to find the optimal learning rate and set it
    Return None is the agent name is not found.
    """
    env, policy = get_env_policy(env_name, seed)
    return build_agent(agent_name, env_name, env, policy, get_optimal, gamma, seed), env, policy

def build_agent(agent_name, env_name, env, policy, get_optimal, gamma, seed=42):
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
        return build_Q_PID(env, kp, ki, kd, alpha, beta, learning_rates, gamma)
    # elif agent_description == "double Q learning":
    #     return build_Q_PID(env, kp, ki, kd, alpha, beta, learning_rates, gamma, double=True)
    elif agent_description == "double Q learning":
        return build_double_Q_PID(env, kp, ki, kd, alpha, beta, learning_rates, gamma)
    
    # If agent_description starts with linear TD, but could have more after
    elif agent_description.startswith("linear TD"):
        order = kwargs[0]
        is_q = (agent_description[-1] == "Q")
        if agent_description.startswith("linear TD polynomial"):
            basis = PolynomialBasis(env, order, is_q)
            return build_linear_TD(env, policy, kp, ki, kd, alpha, beta, learning_rates, gamma, basis, is_q, seed=seed)
        elif agent_description.startswith("linear TD fourier"):
            basis = FourierBasis(env, order, is_q)
            return build_linear_TD(env, policy, kp, ki, kd, alpha, beta, learning_rates, gamma, basis, is_q, seed=seed)
        elif agent_description.startswith("linear TD tile coding"):
            basis = TileCodingBasis(env, order, is_q)
            return build_linear_TD(env, policy, kp, ki, kd, alpha, beta, learning_rates, gamma, basis, is_q, seed=seed)
        elif agent_description.startswith("linear TD trivial"):
            basis = TrivialBasis(env, order, is_q)
            return build_linear_TD(env, policy, kp, ki, kd, alpha, beta, learning_rates, gamma, basis, is_q, seed=seed)
        
    elif agent_description == "TIDBD":
        return build_TIDBD(env, policy, learning_rates, gamma)
    elif agent_description == "zap Q learning":
        return build_Zap_Q_learning(env, policy, learning_rates, gamma)
    elif agent_description == "speedy Q learning":
        return build_Speedy_Q_learning(env, policy, learning_rates, gamma)
    return None


def build_linear_TD(env, policy, kp, ki, kd, alpha, beta, learning_rates, gamma, basis, is_q, adapt_gains=False, meta_lr=0, epsilon=1, seed=42):
    if is_q:
        optimal_agent = build_emperical_Q_tester(env, gamma, seed=seed)
        return LinearTDQ(
            env, policy, gamma, basis,
            kp, ki, kd, alpha, beta,
            *learning_rates, adapt_gains=adapt_gains,
            meta_lr=meta_lr, epsilon=epsilon, solved_agent=optimal_agent
        )
    else:
        optimal_agent = build_emperical_TD_tester(env, policy, gamma)
        return LinearTD(
            env, policy, gamma, basis,
            kp, ki, kd, alpha, beta,
            *learning_rates, adapt_gains=adapt_gains,
            meta_lr=meta_lr, epsilon=epsilon, solved_agent=optimal_agent
        )

def build_double_Q_PID(env, kp, ki, kd, alpha, beta, learning_rates, gamma):
    """Build the double Q learning agent.
    """
    return ControlledDoubleQLearning(
        env, gamma, kp, ki, kd, alpha, beta, learning_rates
    )


def build_TIDBD(env, policy, learning_rates, gamma):
    """Build the TIDBD agent.
    """
    # The learning rate for the TIDBD agent is a float stored in the first component
    return TIDBD(
        env, policy, gamma, theta=learning_rates[0]
    )

def build_Zap_Q_learning(env, policy, learning_rates, gamma):
    # The learning rate for the Zap Q learning agent is stored in the first two components
    return ZapQLearning(
        learning_rates[0], learning_rates[1], env, policy, gamma
    )

def build_Speedy_Q_learning(env, policy, learning_rates, gamma):
    # The learning rate for the speedy Q learning agent is stored in the first component
    return SpeedyQLearning(
        learning_rates[0], env, policy, gamma
    )


def build_Q_PID(env, kp, ki, kd, alpha, beta, learning_rates, gamma, double=False):
    """Build the Q agent with PID
    """
    return ControlledQLearning(
        env,
        gamma,
        kp, ki, kd, alpha, beta,
        learning_rates,
        double=double
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
        kp, ki, kd, alpha, beta,
        gamma
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
