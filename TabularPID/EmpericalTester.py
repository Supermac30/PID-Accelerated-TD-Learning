"""Classes that test the performance of our agents empirically on a
policy evaluation task where either the exact Q values are hard to estimate,
or the agent's optimal setting is hard to find. These are the cases where the exact
distance from the optimum is hard to find, and we need to use an unbiased estimate of it."""

from copy import deepcopy
import numpy as np

import globals
from TabularPID.MDPs.Environments import Environment
from TabularPID.MDPs.MDP import PolicyEvaluation, Control_Q
from TabularPID.AgentBuilders.DQNBuilder import get_model

def build_emperical_TD_tester(env, policy, gamma):
    # check if env inherits from Environment
    if isinstance(env, Environment):
        return TrueEnvTester(env, policy, gamma)
    else:
        env = deepcopy(env)
        return GymTesterDatabase(env, gamma)

def build_emperical_Q_tester(env, gamma):
    if isinstance(env, Environment):
        return TrueQEnvTester(env, gamma)
    else:
        name = env.unwrapped.spec.id
        return GymTesterDatabase(name)

class GymTesterDatabase():
    def __init__(self, env_name):
        self.Q_values = np.load(f'{globals.base_directory}/models/{env_name}/bufferQValues.npy')

    def measure_performance(self, query):
        """
        The input is a function that returns the agent's
        Q-value given the state and action. This function returns the average
        distance between the agent's Q-values and the oracle's Q-values.
        """
        total_distance = 0
        for entry in self.Q_values:
            state = entry[:-2]
            action = entry[-2]
            q_value = entry[-1]
            total_distance += abs(q_value - query(state, action))
        return total_distance / len(self.Q_values)

    def randomly_query_agent(self):
        # Pick an entry at random from the buffer
        entry = self.Q_values[np.random.randint(0, len(self.Q_values))]
        state = entry[:-2]
        action = entry[-2]
        q_value = entry[-1]
        return state, action, q_value

class GymTester():
    def __init__(self, env, gamma):
        self.env = env
        self.model = get_model(self.env.unwrapped.spec.id)
        self.gamma = gamma
    
    def measure_performance(self, query):
        """Measure the performance of the agent using the optimal agent.
        """
        distance = 0

        for _ in range(10):
            state, q_value = self.solved_agent.randomly_query_agent()
            distance += abs(q_value - self.query_agent(state))

        # Return the mean of these values
        return np.mean(distance)

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
    
class TrueEnvTester():
    def __init__(self, env, policy, gamma):
        self.oracle = get_optimal_TD(env, policy, gamma)

    def measure_performance(self, query):
        return sum(abs(self.oracle.V[state] - query(state)) for state in range(self.oracle.num_states))

    
class TrueQEnvTester():
    def __init__(self, env, gamma):
        self.oracle = get_optimal_Q(env, gamma)

    def measure_performance(self, query):
        return sum(
            abs(self.oracle.Q[state][action] - query(state, action))
            for state in range(self.oracle.num_states)
            for action in range(self.oracle.num_actions)
        )


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
    oracle = Control_Q(
        env.num_states,
        env.num_actions,
        env.build_reward_matrix(),
        env.build_probability_transition_kernel(),
        1,0,0,0,0,
        gamma
    )
    oracle.value_iteration(num_iterations=10000)
    return oracle