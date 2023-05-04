"""
Builds the adaptive agent given a string description of it, and a string description of the environment

Possible agents include:
- Samplers: TD Agents
- Planners: VI Agents (For reproducing PAVIA results)

- Sample True Cost Gradients: Finds the true gradients then samples
- Sample Empirical Cost Gradient: Finds the cost after sampling
"""
from Experiments.ExperimentHelpers import learning_rate_function, get_env_policy
from Experiments.OptimalRateDatabase import get_stored_optimal_rate
from AdaptiveAgents import AdaptivePlannerAgent, AdaptiveSamplerAgent, ExactUpdater, SamplerUpdater, SoftGainUpdater

default_meta_lr = 1
default_learning_rates = (
    learning_rate_function(1, 1),
    learning_rate_function(1, 1),
    learning_rate_function(1, 1)
)

def build_adaptive_agent_and_env(agent_name, env_name, get_optimal=False, meta_lr_value=None, seed=-1, gamma=0.99):
    """Return the adaptive agent and the environment & policy given its name. The names include:
    - planner: The original PAVIA gain adaptation algorithm
    - true cost: Gain adaptation sampling the true gradients (here we don't scale by the learning rate)
    - sampled true cost: Gain adaptation using the true cost, but re-deriving the samples (here we scale by the learning rate)
    - sampled empirical cost: Gain adaptation by sampling the cost (here we scale by the learning rate)
    - naive soft sampler: The naive gain adaptation algorithm

    get_optimal: Try to find the optimal learning rate and set it
    meta_lr_value: It is useful to be able to set meta_lr manually. If this is not None, then regardless of any grid search, meta_lr
                is set to meta_lr_value

    Return None if the names are not in the list of possible names.
    """
    env, policy = get_env_policy(env_name, seed)
    agent = build_adaptive_agent(agent_name, env_name, env, policy, get_optimal, meta_lr_value, gamma)
    return agent, env, policy

def build_adaptive_agent(agent_name, env_name, env, policy, get_optimal=False, meta_lr_value=None, gamma=0.99):
    """Return the adaptive agent given its name.

    get_optimal: Try to find the optimal learning rate and set it
    meta_lr_value: It is useful to be able to set meta_lr manually. If this is not None, then regardless of any grid search, meta_lr
                is set to meta_lr_value

    Return None if the names are not in the list of possible names.
    """
    if get_optimal:
        meta_lr, learning_rates = get_stored_optimal_rate(agent_name, env_name)
    if not get_optimal or meta_lr is None:
        meta_lr, learning_rates = default_meta_lr, default_learning_rates

    if meta_lr_value is not None:
        meta_lr = meta_lr_value

    if agent_name == "planner":
        return build_planner(env, policy, meta_lr, learning_rates, gamma)
    elif agent_name == "true cost":
        return build_true_cost(env, policy, meta_lr, learning_rates, gamma)
    elif agent_name == "sampled true cost":
        return build_sampled_true_cost(env, policy, meta_lr, learning_rates, gamma)
    elif agent_name =="sampled empirical cost":
        return build_sampled_empirical_cost(env, policy, meta_lr, learning_rates, gamma)
    elif agent_name == "naive soft sampler":
        return build_naive_soft_updates(env, policy, meta_lr, learning_rates, gamma)
    return None


def build_planner(env, policy, meta_lr, learning_rates, gamma):
    reward = env.build_policy_reward_vector(policy)
    transition = env.build_policy_probability_transition_kernel(policy)

    gain_updater = ExactUpdater(transition, reward, False)

    return AdaptivePlannerAgent(
        reward,
        transition,
        gain_updater,
        learning_rates,
        meta_lr,
        env,
        policy,
        gamma
    )

def build_true_cost(env, policy, meta_lr, learning_rates, gamma):
    reward = env.build_policy_reward_vector(policy)
    transition = env.build_policy_probability_transition_kernel(policy)
    gain_updater = ExactUpdater(transition, reward, True)
    return AdaptiveSamplerAgent(
        gain_updater,
        learning_rates,
        meta_lr,
        env,
        policy,
        gamma
    )

def build_sampled_true_cost(env, policy, meta_lr, learning_rates, gamma):
    reward = env.build_policy_reward_vector(policy)
    transition = env.build_policy_probability_transition_kernel(policy)

    gain_updater = ExactUpdater(transition, reward, False)
    return AdaptiveSamplerAgent(
        gain_updater,
        learning_rates,
        meta_lr,
        env,
        policy,
        gamma
    )

def build_sampled_empirical_cost(env, policy, meta_lr, learning_rates, gamma):
    gain_updater = SamplerUpdater(10, True)
    return AdaptiveSamplerAgent(
        gain_updater,
        learning_rates,
        meta_lr,
        env,
        policy,
        gamma
    )

def build_naive_soft_updates(env, policy, meta_lr, learning_rates, gamma):
    gain_updater = SoftGainUpdater(env.num_states)
    return AdaptiveSamplerAgent(
        gain_updater,
        learning_rates,
        meta_lr,
        env,
        policy,
        gamma
    )