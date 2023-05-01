"""
Builds the adaptive agent given a string description of it, and a string description of the environment

Possible agents include:
- Samplers: TD Agents
- Planners: VI Agents (For reproducing PAVIA results)

- Sample True Cost Gradients: Finds the true gradients then samples
- Sample Empirical Cost Gradient: Finds the cost after sampling
"""
from Experiments.ExperimentHelpers import get_env_policy, learning_rate_function
from AdaptiveAgents import AdaptivePlannerAgent, AdaptiveSamplerAgent, ExactUpdater, SamplerUpdater

default_meta_lr = 0.1
default_learning_rates = (
    learning_rate_function(1, 1),
    learning_rate_function(1, 1),
    learning_rate_function(1, 1)
)

def build_adaptive_agent(agent_name, env_name):
    """Return the adaptive agent given its name. The names include:
    - planner: The original PAVIA gain adaptation algorithm
    - true cost: Gain adaptation sampling the true gradients (here we don't scale by the learning rate)
    - sampled true cost: Gain adaptation using the true cost, but re-deriving the samples (here we scale by the learning rate)
    - sampled empirical cost: Gain adaptation by sampling the cost

    Return None if the names are not in the list of possible names.
    """
    env, policy = get_env_policy(env_name)
    if agent_name == "planner":
        return build_planner(env, policy)
    elif agent_name == "sampled true cost":
        return build_sampled_true_cost(env, policy)
    elif agent_name =="sampled empirical cost":
        return build_sampled_empirical_cost(env, policy)
    return None


def build_planner(env, policy):
    reward = env.build_policy_reward_matrix(policy)
    transition = env.build_policy_probability_transition_kernel(policy)

    gain_updater = ExactUpdater(transition, reward, False)

    return AdaptivePlannerAgent(
        reward,
        transition,
        gain_updater,
        default_learning_rates,
        default_meta_lr,
        env,
        policy,
        0.99
    )

def build_sampled_true_cost(env, policy):
    reward = env.build_policy_reward_matrix(policy)
    transition = env.build_policy_probability_transition_kernel(policy)

    gain_updater = ExactUpdater(transition, reward, False)
    return AdaptiveSamplerAgent(
        gain_updater,
        default_learning_rates,
        default_meta_lr,
        env,
        policy
    )

def build_sampled_empirical_cost(env, policy):
    gain_updater = SamplerUpdater(10, True)
    return AdaptiveSamplerAgent(
        gain_updater,
        default_learning_rates,
        default_meta_lr,
        env,
        policy
    )