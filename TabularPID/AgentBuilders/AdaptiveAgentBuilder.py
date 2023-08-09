"""
Builds the adaptive agent given a string description of it, and a string description of the environment

Possible agents include:
- Samplers: TD Agents
- Planners: VI Agents (For reproducing PAVIA results)
"""
from Experiments.ExperimentHelpers import get_env_policy
from TabularPID.OptimalRates.OptimalRateDatabase import get_stored_optimal_rate
from TabularPID.Agents.AdaptiveAgents import *
import TabularPID.Agents.AdaptiveQAgents as AdaptiveQAgents
import TabularPID.Agents.AdaptiveDoubleQAgents as AdaptiveDoubleQAgents
from TabularPID.Agents.Agents import learning_rate_function
import logging

default_meta_lr = 1
default_learning_rates =  (1, 100, 1, float("inf"), 0.01, float("inf"))  #  (0.1, 10, 1, float("inf"), 0.01, float("inf"))

def build_adaptive_agent_and_env(agent_name, env_name, meta_lr, lambd, delay, get_optimal=False, seed=42, gamma=0.99, kp=1, kd=0, ki=0, alpha=0.05, beta=0.95, epsilon=0.1):
    """Return the adaptive agent and the environment & policy given its name. The names include:
    - planner: The original PAVIA gain adaptation algorithm
    - log space planner: The original PAVIA gain adaptation algorithm, but in the log space
    - true cost: Gain adaptation sampling the true gradients (here we don't scale by the learning rate)
    - diagonal true cost: Gain adaptation sampling the true gradients, and using diagonal gains.
    - scaled true cost: Gain adaptation using the true cost, but re-deriving the samples (here we scale by the learning rate)
    - sampled true cost: Gain adaptation by sampling the cost (here we scale by the learning rate)
    - empirical cost: Gain adaptation by changing the cost function to be the empirical cost (here we scale by the learning rate)
    - naive soft sampler: The naive gain adaptation algorithm
    - diagonal sampler: The diagonal gain adaptation algorithm
    - diagonal log space updater: The diagonal log space updater
    - diagonal semi gradient updater: The diagonal semi gradient updater
    - partials exact sampler: Uses the exact gradients for the partials, but not for the bellman
    - bellman exact sampler: Uses the exact gradients for the bellman, but not for the partials
    - semi gradient updater: Uses the semi-gradient updater
    - true soft sampler: The true soft sampling gain adaptation algorithm
    - log space updater: Updates the gains in the log space
    - true log space updater: Updates the gains in the log space using the true gradients
    - true diagonal semi gradient updater: Updates the gains using true knowledge of the BR, and using diagonal gains
    - true semi gradient updater: Updates the gains using true knowledge of the BR

    get_optimal: Try to find the optimal learning rate and set it
    meta_lr_value: It is useful to be able to set meta_lr manually. If this is not None, then regardless of any grid search, meta_lr
                is set to meta_lr_value

    Return None if the names are not in the list of possible names.
    """
    env, policy = get_env_policy(env_name, seed)
    agent = build_adaptive_agent(agent_name, env_name, env, policy, meta_lr, lambd, delay, get_optimal, gamma, kp, kd, ki, alpha, beta, epsilon)
    return agent, env, policy

def build_adaptive_agent(agent_name, env_name, env, policy, meta_lr, lambd, delay, get_optimal, gamma, kp, kd, ki, alpha, beta, epsilon):
    """Return the adaptive agent given its name.

    get_optimal: Try to find the optimal learning rate and set it
    meta_lr_value: It is useful to be able to set meta_lr manually. If this is not None, then regardless of any grid search, meta_lr
                is set to meta_lr_value

    Return None if the names are not in the list of possible names.
    """
    if get_optimal:
        optimal_rates = get_stored_optimal_rate((agent_name, meta_lr, lambd, delay, alpha, beta, epsilon), env_name, gamma)
    if not get_optimal or optimal_rates is None:
        optimal_rates = default_learning_rates

    learning_rate = learning_rate_function(optimal_rates[0], optimal_rates[1])
    update_I_rate = learning_rate_function(optimal_rates[2], optimal_rates[3])
    update_D_rate = learning_rate_function(optimal_rates[4], optimal_rates[5])

    rates = (learning_rate, update_I_rate, update_D_rate)

    logging.info(f"Using rates {optimal_rates} for agent {agent_name} on env {env_name}")

    params = env, policy, meta_lr, lambd, rates, gamma, delay, kp, kd, ki, alpha, beta, epsilon

    if agent_name == "planner":
        return build_planner(*params)
    elif agent_name == "no gain adapter":
        return build_no_gain_adapter(*params)
    elif agent_name == "log space planner":
        return build_log_space_planner(*params)
    elif agent_name == "true cost":
        return build_true_cost(*params)
    elif agent_name == "scaled true cost":
        return build_scaled_true_cost(*params)
    elif agent_name =="sampled true cost":
        return build_sampled_true_cost(*params)
    elif agent_name == "empirical cost":
        return build_empirical_cost(*params)
    elif agent_name == "naive soft sampler":
        return build_naive_soft_updates(*params)
    elif agent_name == "diagonal sampler":
        return build_diagonal_sampler(*params)
    elif agent_name == "diagonal log space updater":
        return build_diagonal_log_space_updater(*params)
    elif agent_name == "partials exact sampler":
        return build_partials_exact_sampler(*params)
    elif agent_name == "bellman exact sampler":
        return build_bellman_exact_sampler(*params)
    elif agent_name == "semi gradient updater":
        return build_semi_gradient_updater(*params)
    elif agent_name == "true soft sampler":
        return build_true_soft_updates(*params)
    elif agent_name == "log space updater":
        return build_log_space_updater(*params)
    elif agent_name == "true log space updater":
        return build_true_log_space_updater(*params)
    elif agent_name == "diagonal semi gradient updater":
        return build_diagonal_semi_gradient_updater(*params)
    elif agent_name == "diagonal true cost":
        return build_diagonal_true_cost(*params)
    elif agent_name == "true diagonal semi gradient updater":
        return build_true_diagonal_semi_gradient_updater(*params)
    elif agent_name == "true semi gradient updater":
        return build_true_semi_gradient_updater(*params)


    elif agent_name == "no gain Q adapter":
        return build_no_gain_Q_adapter(*params)
    elif agent_name == "semi gradient Q updater":
        return build_semi_gradient_Q_updater(*params)
    elif agent_name == "diagonal semi gradient Q updater":
        return build_diagonal_semi_gradient_Q_updater(*params)

    elif agent_name == "semi gradient double Q updater":
        return build_semi_gradient_double_Q_updater(*params)
    return None


def build_semi_gradient_double_Q_updater(env, policy, meta_lr, lambd, learning_rates, gamma, delay, kp, kd, ki, alpha, beta, epsilon):
    gain_updater = AdaptiveDoubleQAgents.SemiGradientUpdater(lambd, epsilon)
    return AdaptiveDoubleQAgents.AdaptiveSamplerAgent(
        gain_updater,
        learning_rates,
        meta_lr,
        env,
        gamma,
        delay,
        kp, kd, ki, alpha, beta
    )


def build_diagonal_semi_gradient_Q_updater(env, policy, meta_lr, lambd, learning_rates, gamma, delay, kp, kd, ki, alpha, beta, epsilon):
    gain_updater = AdaptiveQAgents.DiagonalSemiGradient(lambd, epsilon)
    return AdaptiveQAgents.DiagonalAdaptiveSamplerAgent(
        gain_updater,
        learning_rates,
        meta_lr,
        env,
        gamma,
        delay,
        kp, kd, ki, alpha, beta
    )


def build_no_gain_Q_adapter(env, policy, meta_lr, lambd, learning_rates, gamma, delay, kp, kd, ki, alpha, beta, epsilon):
    gain_updater = AdaptiveQAgents.NoGainUpdater()
    return AdaptiveQAgents.AdaptiveSamplerAgent(
        gain_updater,
        learning_rates,
        meta_lr,
        env,
        gamma,
        delay,
        kp, kd, ki, alpha, beta
    )


def build_semi_gradient_Q_updater(env, policy, meta_lr, lambd, learning_rates, gamma, delay, kp, kd, ki, alpha, beta, epsilon):
    gain_updater = AdaptiveQAgents.SemiGradientUpdater(lambd, epsilon)
    return AdaptiveQAgents.AdaptiveSamplerAgent(
        gain_updater,
        learning_rates,
        meta_lr,
        env,
        gamma,
        delay,
        kp, kd, ki, alpha, beta
    )


def build_no_gain_adapter(env, policy, meta_lr, lambd, learning_rates, gamma, delay, kp, kd, ki, alpha, beta, epsilon):
    gain_updater = NoGainUpdater()
    return AdaptiveSamplerAgent(
        gain_updater,
        learning_rates,
        meta_lr,
        env,
        policy,
        gamma,
        delay,
        kp, kd, ki, alpha, beta
    )


def build_true_semi_gradient_updater(env, policy, meta_lr, lambd, learning_rates, gamma, delay, kp, kd, ki, alpha, beta, epsilon):
    reward = env.build_policy_reward_vector(policy)
    transition = env.build_policy_probability_transition_kernel(policy)
    gain_updater = TrueSemiGradient(reward, transition, lambd, epsilon)
    return AdaptiveSamplerAgent(
        gain_updater,
        learning_rates,
        meta_lr,
        env,
        policy,
        gamma,
        delay,
        kp, kd, ki, alpha, beta
    )


def build_true_diagonal_semi_gradient_updater(env, policy, meta_lr, lambd, learning_rates, gamma, delay, kp, kd, ki, alpha, beta, epsilon):
    reward = env.build_policy_reward_vector(policy)
    transition = env.build_policy_probability_transition_kernel(policy)
    gain_updater = TrueDiagonalSemiGradient(reward, transition, lambd, epsilon)
    return DiagonalAdaptiveSamplerAgent(
        gain_updater,
        learning_rates,
        meta_lr,
        env,
        policy,
        gamma,
        delay,
        kp, kd, ki, alpha, beta
    )


def build_diagonal_true_cost(env, policy, meta_lr, lambd, learning_rates, gamma, delay, kp, kd, ki, alpha, beta, epsilon):
    reward = env.build_policy_reward_vector(policy)
    transition = env.build_policy_probability_transition_kernel(policy)
    gain_updater = DiagonalExactUpdater(transition, reward, lambd, epsilon)
    return DiagonalAdaptiveSamplerAgent(
        gain_updater,
        learning_rates,
        meta_lr,
        env,
        policy,
        gamma,
        delay,
        kp, kd, ki, alpha, beta
    )


def build_diagonal_semi_gradient_updater(env, policy, meta_lr, lambd, learning_rates, gamma, delay, kp, kd, ki, alpha, beta, epsilon):
    gain_updater = DiagonalSemiGradient(lambd, epsilon)
    return DiagonalAdaptiveSamplerAgent(
        gain_updater,
        learning_rates,
        meta_lr,
        env,
        policy,
        gamma,
        delay,
        kp, kd, ki, alpha, beta
    )


def build_diagonal_log_space_updater(env, policy, meta_lr, lambd, learning_rates, gamma, delay, kp, kd, ki, alpha, beta, epsilon):
    gain_updater = DiagonalLogSpaceUpdater(env.num_states, lambd, epsilon)
    return DiagonalAdaptiveSamplerAgent(
        gain_updater,
        learning_rates,
        meta_lr,
        env,
        policy,
        gamma,
        delay,
        kp, kd, ki, alpha, beta
    )



def build_true_log_space_updater(env, policy, meta_lr, lambd, learning_rates, gamma, delay, kp, kd, ki, alpha, beta, epsilon):
    reward = env.build_policy_reward_vector(policy)
    transition = env.build_policy_probability_transition_kernel(policy)

    gain_updater = LogisticExactUpdater(
        transition,
        reward,
        N_p = 0.75,
        N_d = (1 - gamma)/4,
        N_I = (10/19) * (1 - gamma)/(1 + gamma),
        lambd=lambd,
        epsilon=epsilon
    )
    return AdaptiveSamplerAgent(
        gain_updater,
        learning_rates,
        meta_lr,
        env,
        policy,
        gamma,
        delay,
        kp, kd, ki, alpha, beta
    )


def build_log_space_updater(env, policy, meta_lr, lambd, learning_rates, gamma, delay, kp, kd, ki, alpha, beta, epsilon):
    gain_updater = LogSpaceUpdater(
        env.num_states,
        N_p = 9,
        N_d = (1 - gamma)/4,
        N_I = (1/19) * (1 - gamma)/(1 + gamma),
        lambd=lambd,
        lax=10,
        epsilon=epsilon
    )
    return AdaptiveSamplerAgent(
        gain_updater,
        learning_rates,
        meta_lr,
        env,
        policy,
        gamma,
        delay,
        kp, kd, ki, alpha, beta
    )


def build_true_soft_updates(env, policy, meta_lr, lambd, learning_rates, gamma, delay, kp, kd, ki, alpha, beta, epsilon):
    gain_updater = TrueSoftGainUpdater(env.num_states, lambd, epsilon)
    return AdaptiveSamplerAgent(
        gain_updater,
        learning_rates,
        meta_lr,
        env,
        policy,
        gamma,
        delay,
        kp, kd, ki, alpha, beta
    )


def build_semi_gradient_updater(env, policy, meta_lr, lambd, learning_rates, gamma, delay, kp, kd, ki, alpha, beta, epsilon):
    gain_updater = SemiGradientUpdater(lambd, epsilon)
    return AdaptiveSamplerAgent(
        gain_updater,
        learning_rates,
        meta_lr,
        env,
        policy,
        gamma,
        delay,
        kp, kd, ki, alpha, beta
    )


def build_bellman_exact_sampler(env, policy, meta_lr, lambd, learning_rates, gamma, delay, kp, kd, ki, alpha, beta, epsilon):
    reward = env.build_policy_reward_vector(policy)
    transition = env.build_policy_probability_transition_kernel(policy)
    gain_updater = BellmanExactUpdater(transition, reward, 1, False, lambd, epsilon)
    return AdaptiveSamplerAgent(
        gain_updater,
        learning_rates,
        meta_lr,
        env,
        policy,
        gamma,
        delay,
        kp, kd, ki, alpha, beta
    )


def build_partials_exact_sampler(env, policy, meta_lr, lambd, learning_rates, gamma, delay, kp, kd, ki, alpha, beta, epsilon):
    reward = env.build_policy_reward_vector(policy)
    transition = env.build_policy_probability_transition_kernel(policy)
    gain_updater = PartialsExactUpdater(transition, reward, 1, False, lambd, epsilon)
    return AdaptiveSamplerAgent(
        gain_updater,
        learning_rates,
        meta_lr,
        env,
        policy,
        gamma,
        delay,
        kp, kd, ki, alpha, beta
    )


def build_diagonal_sampler(env, policy, meta_lr, lambd, learning_rates, gamma, delay, kp, kd, ki, alpha, beta, epsilon):
    gain_updater = DiagonalSoftGainUpdater(env.num_states, lambd, epsilon)
    return DiagonalAdaptiveSamplerAgent(
        gain_updater,
        learning_rates,
        meta_lr,
        env,
        policy,
        gamma,
        delay,
        kp, kd, ki, alpha, beta
    )


def build_planner(env, policy, meta_lr, lambd, learning_rates, gamma, delay, kp, kd, ki, alpha, beta, epsilon):
    reward = env.build_policy_reward_vector(policy)
    transition = env.build_policy_probability_transition_kernel(policy)

    gain_updater = ExactUpdater(transition, reward, False, lambd, epsilon)

    return AdaptivePlannerAgent(
        reward,
        transition,
        gain_updater,
        learning_rates,
        meta_lr,
        env,
        policy,
        gamma,
        delay,
        kp, kd, ki, alpha, beta
    )

def build_log_space_planner(env, policy, meta_lr, lambd, learning_rates, gamma, delay, kp, kd, ki, alpha, beta, epsilon):
    reward = env.build_policy_reward_vector(policy)
    transition = env.build_policy_probability_transition_kernel(policy)

    gain_updater = LogisticExactUpdater(transition, reward, lambd, epsilon)

    return AdaptivePlannerAgent(
        reward,
        transition,
        gain_updater,
        learning_rates,
        meta_lr,
        env,
        policy,
        gamma,
        delay,
        kp, kd, ki, alpha, beta
    )

def build_true_cost(env, policy, meta_lr, lambd, learning_rates, gamma, delay, kp, kd, ki, alpha, beta, epsilon):
    reward = env.build_policy_reward_vector(policy)
    transition = env.build_policy_probability_transition_kernel(policy)
    gain_updater = ExactUpdater(transition, reward, True, lambd, epsilon)
    return AdaptiveSamplerAgent(
        gain_updater,
        learning_rates,
        meta_lr,
        env,
        policy,
        gamma,
        delay,
        kp, kd, ki, alpha, beta
    )

def build_scaled_true_cost(env, policy, meta_lr, lambd, learning_rates, gamma, delay, kp, kd, ki, alpha, beta, epsilon):
    reward = env.build_policy_reward_vector(policy)
    transition = env.build_policy_probability_transition_kernel(policy)

    gain_updater = ExactUpdater(transition, reward, True, lambd, epsilon)
    return AdaptiveSamplerAgent(
        gain_updater,
        learning_rates,
        meta_lr,
        env,
        policy,
        gamma,
        delay,
        kp, kd, ki, alpha, beta
    )

def build_sampled_true_cost(env, policy, meta_lr, lambd, learning_rates, gamma, delay, kp, kd, ki, alpha, beta, epsilon):
    gain_updater = SamplerUpdater(4, True, lambd, epsilon)
    return AdaptiveSamplerAgent(
        gain_updater,
        learning_rates,
        meta_lr,
        env,
        policy,
        gamma,
        delay,
        kp, kd, ki, alpha, beta
    )

def build_naive_soft_updates(env, policy, meta_lr, lambd, learning_rates, gamma, delay, kp, kd, ki, alpha, beta, epsilon):
    gain_updater = NaiveSoftGainUpdater(env.num_states, lambd, epsilon)
    return AdaptiveSamplerAgent(
        gain_updater,
        learning_rates,
        meta_lr,
        env,
        policy,
        gamma,
        delay,
        kp, kd, ki, alpha, beta
    )

def build_empirical_cost(env, policy, meta_lr, lambd, learning_rates, gamma, delay, kp, kd, ki, alpha, beta, epsilon):
    assert delay >= 2  # delays have to be at least two for this to make sense
    gain_updater = EmpiricalCostUpdater(lambd, epsilon)
    return AdaptiveSamplerAgent(
        gain_updater,
        learning_rates,
        meta_lr,
        env,
        policy,
        gamma,
        delay,
        kp, kd, ki, alpha, beta
    )
