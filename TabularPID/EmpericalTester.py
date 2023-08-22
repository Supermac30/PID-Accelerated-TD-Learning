"""Classes that test the performance of our agents empirically on a
policy evaluation task where either the exact Q values are hard to estimate,
or the agent's optimal setting is hard to find. These are the cases where the exact
distance from the optimum is hard to find, and we need to use an unbiased estimate of it."""

from copy import deepcopy
import numpy as np

import globals
from TabularPID.MDPs.Environments import Environment
from TabularPID.MDPs.MDP import PolicyEvaluation, Control
from TabularPID.AgentBuilders.DQNBuilder import get_model

# TODO: Refactor code to follow the policy instead of doing it ad hoc.

def build_emperical_TD_tester(env, policy, gamma):
    env = deepcopy(env)

    # check if env inherits from Environment
    if isinstance(env, Environment):
        return get_optimal_TD(env, policy, gamma)
    else:
        return GymTester(env, gamma)

def build_emperical_Q_tester(env, gamma):
    if env in {'CartPole-v1', 'MountainCar-v0', 'Acrobot-v1', 'LunarLander-v2'}:
        return GymTesterDatabase(env)
    
    else:
        return get_optimal_Q(env, gamma)

class GymTesterDatabase():
    def __init__(self, env_name):
        self.Q_values = np.load(f'{globals.base_directory}/models/{env_name}/bufferQValues.npy')

    def randomly_query_agent(self):
        # Pick an entry at random from the buffer
        state, action, q_value = self.Q_values[np.random.randint(0, len(self.Q_values))]
        return state, action, q_value

class GymTester():
    def __init__(self, env, gamma):
        self.env = env
        self.model = get_model(self.env.unwrapped.spec.id)
        self.gamma = gamma

    def randomly_query_agent(self):
        # Choose a state at random
        state = self.env.observation_space.sample()

        self.env.reset()
        self.env.state = state
        total_reward = 0
        discount = 1

        done = False
        truncated = False
        while not (done or truncated):
            action = self.model.predict(self.env.state.reshape(1, -1))[0].item()
            state, reward, done, truncated, _ = self.env.step(action)
            total_reward += discount * reward
            discount *= self.gamma

        return state, total_reward
    
def get_optimal_TD(env, policy, gamma):
    oracle = PolicyEvaluation(
        env.num_states,
        env.num_actions,
        env.build_policy_reward_vector(policy),
        env.build_policy_probability_transition_kernel(policy),
        1,0,0,0,0,
        gamma
    )
    oracle.value_iteration(num_iterations=10000)
    return oracle

def get_optimal_Q(env, gamma):
    oracle = Control(
        env.num_states,
        env.num_actions,
        env.build_reward_matrix(),
        env.build_probability_transition_kernel(),
        1,0,0,0,0,
        gamma
    )
    oracle.value_iteration(num_iterations=10000)
    return oracle
