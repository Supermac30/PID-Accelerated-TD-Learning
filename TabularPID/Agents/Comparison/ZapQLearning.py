from TabularPID.Agents.Agents import Agent, learning_rate_function
import numpy as np


class ZapQLearning(Agent):
    def __init__(self, gamma_lr, alpha_lr, environment, policy, gamma):
        super().__init__(environment, policy, gamma)
        self.gamma_lr = gamma_lr
        self.alpha_lr = alpha_lr
        self.Q = np.zeros((self.num_states, self.num_actions))

    def unit_mat(self, a, b):
        """Create a matrix with a one at (a, b) and zeros everywhere else
        """
        matrix = np.zeros((self.num_states, self.num_actions))
        matrix[a, b] = 1
        return matrix

    def set_learning_rates(self, a, b, c, d, e, f):
        self.gamma_lr = learning_rate_function(a, b)
        self.alpha_lr = learning_rate_function(c, d)

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

        # The matrix A, initialized randomly to avoid singularities
        rolling_A = np.random.rand(self.num_states, self.num_states)

        for k in range(num_iterations):
            current_state, action, next_state, reward = self.take_action(follow_trajectory, is_q=True)
            frequency[current_state] += 1

            best_action = np.argmax(self.Q[next_state])
            BR = reward + self.gamma * self.Q[next_state, best_action] - self.Q[current_state, action]

            gamma_lr = self.gamma_lr(frequency[current_state])
            alpha_lr = self.alpha_lr(frequency[current_state])

            A = self.unit_mat(current_state, action) @ (self.gamma * self.unit_mat(next_state, best_action) - self.unit_mat(current_state, action)).T
            rolling_A = rolling_A + gamma_lr * (A - rolling_A)
            
            self.Q -= alpha_lr * np.linalg.inv(rolling_A) @ BR

            if test_function is not None:
                history[k] = test_function(self.current_Q, None, BR)
                if stop_if_diverging and history[k] > 10 * history[0]:
                    # If we are too large, stop learning
                    history[k:] = float('inf')
                    break

        if test_function is None:
            return self.current_Q

        return history, self.current_Q