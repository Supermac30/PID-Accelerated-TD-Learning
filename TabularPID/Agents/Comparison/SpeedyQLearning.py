from TabularPID.Agents.Agents import Agent, learning_rate_function
import numpy as np

class SpeedyQLearning(Agent):
    def __init__(self, learning_rate, environment, policy, gamma):
        super().__init__(environment, policy, gamma)
        self.learning_rate = learning_rate
        self.previous_Q = np.zeros((self.num_states, self.num_actions))
        self.current_Q = np.zeros((self.num_states, self.num_actions))

    def set_learning_rates(self, a, b, c, d, e, f):
        self.learning_rate = learning_rate_function(a, b)

    def estimate_value_function(self, follow_trajectory=True, num_iterations=1000, test_function=None, reset=True, reset_environment=True, stop_if_diverging=True):
        """Estimate the value function of the current policy using the TIDBD algorithm
        theta is the meta learning rate
        """
        if reset:
            self.reset(reset_environment)

        # The history of test_function
        history = np.zeros(num_iterations)

        # A vector storing the number of times we have seen a state.
        frequency = np.zeros((self.num_states, 1))

        for k in range(num_iterations):
            current_state, action, next_state, reward = self.take_action(follow_trajectory, is_q=True)
            frequency[current_state] += 1

            previous_bellman = reward + self.gamma * max(self.previous_Q[next_state])
            current_bellman = reward + self.gamma * max(self.current_Q[next_state])

            learning_rate = self.learning_rate(frequency[current_state])

            self.previous_Q = self.current_Q.copy()
            self.current_Q[current_state, action] = (1 - learning_rate) * self.current_Q[current_state, action] + learning_rate * previous_bellman
            self.current_Q[current_state, action] += (1 - learning_rate) * (current_bellman - previous_bellman)

            if test_function is not None:
                history[k] = test_function(self.current_Q, None, current_bellman - self.current_Q[current_state, action])
                if stop_if_diverging and history[k] > 10 * history[0]:
                    # If we are too large, stop learning
                    history[k:] = float('inf')
                    break

        if test_function is None:
            return self.current_Q

        return history, self.current_Q