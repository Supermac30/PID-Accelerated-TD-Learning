import gymnasium as gym
from TabularPID.MDPs.Policy import Policy
from TabularPID.AgentBuilders.DQNBuilder import get_model
from TabularPID.MDPs.Environments import *

def get_env_policy(name, seed):
    """Return an environment, policy tuple given its string name as input. A seed is inputted as well
    for reproducibility.
    The environments are as follows:
        - "garnet": The garnet problem with the default settings in PAVIA
        - "chain walk": The chain walk problem with 50 states, and a policy that always moves left
        - "chain walk random": The chain walk problem with 50 states, and a policy that takes random choices
        - "cliff walk": The Cliff walk problem as implemented in OS-Dyna
    """
    if name[:6] == "garnet":
        # The name is of the form garnet <seed> <num_states>
        seed, num_states = map(int, name[7:].split(" "))
        return garnet_problem(num_states, 3, 5, 10, seed)
    elif name == "chain walk":
        return chain_walk_left(50, 2, seed)
    elif name == "chain walk random":
        return chain_walk_random(50, 2, seed)
    elif name == "cliff walk":
        return cliff_walk(seed)
    elif name == "cliff walk optimal":
        return cliff_walk_optimal(seed)
    elif name == "identity":
        return identity(seed)
    elif name[:6] == "normal":
        variance = float(name[7:])
        return normal_env(variance, seed)
    elif name == "bernoulli":
        return bernoulli_env(seed)
    elif name in {"CartPole-v1", "Acrobot-v1", "MountainCar-v0", "LunarLander-v2"}:
        return gym.make(name), GymPolicy(name)
    else:
        raise Exception("Environment not indexed")


class GymPolicy(Policy):
    def __init__(self, name):
        self.model = get_model(name)

    def get_action(self, state):
        return self.model.predict(state.reshape(1, -1))[0]


def bernoulli_env(seed):
    env = BernoulliApproximation(seed)
    return env, Policy(env.num_actions, env.num_states, env.prg, None)

def normal_env(variance, seed):
    env = NormalApproximation(variance, seed)
    return env, Policy(env.num_actions, env.num_states, env.prg, None)

def cliff_walk(seed):
    """Return the CliffWalk Environment with the policy that moves randomly"""
    env = CliffWalk(0.9, seed)
    return env, Policy(env.num_actions, env.num_states, env.prg, None)

def cliff_walk_optimal(seed):
    """Return the CliffWalk Environment with the optimal policy"""
    env = CliffWalk(0.9, seed)
    policy = np.zeros((env.num_states, env.num_actions))

    for i in range(env.num_states):
        if i in {0, 12, 24}:
            policy[i, 2] = 1
        elif i in range(6, 12) or i in range(18, 23) or i in range(30, 35):
            policy[i, 1] = 1
        else:
            policy[i, 0] = 1

    return env, Policy(env.num_actions, env.num_states, env.prg, policy)

def garnet_problem(num_states, num_actions, bP, bR, seed):
    """Return the Garnet Environment with the policy that chooses the first move at each iteration"""
    env = Garnet(
        num_states, num_actions, bP, bR, seed
    )
    return env, Policy(num_actions, num_states, env.prg, None)

def PAVIA_garnet_settings(seed):
    """Return Garnet used in the experiments of PAVIA."""
    return garnet_problem(50, 1, 3, 5, seed)

def chain_walk_random(num_states, num_actions, seed):
    """Return the chain walk environment with the policy that takes random moves."""
    env = ChainWalk(num_states, seed)
    return env, Policy(num_actions, num_states, env.prg, None)

def chain_walk_left(num_states, num_actions, seed):
    """Return the chain walk environment with the policy that always moves left."""
    env = ChainWalk(num_states, seed)
    policy = np.zeros((num_states, num_actions))
    for i in range(num_states):
        policy[i,0] = 1
        policy[i,1] = 0
    return env, Policy(num_actions, num_states, env.prg, policy)

def identity(seed):
    """Return the identity environment the policy that picks the only available action."""
    env = IdentityEnv(1, seed)
    return env, Policy(1, 1, env.prg, None)
