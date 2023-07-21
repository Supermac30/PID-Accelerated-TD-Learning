from TabularPID.Agents.Agents import Agent, learning_rate_function
import numpy as np

class TIDBD(Agent):
    def __init__(self, environment, policy, gamma, theta):
        super().__init__(environment, policy, gamma)
        self.V = np.zeros((self.num_states, 1))
        self.theta = theta

    def set_learning_rates(self, a, b, c, d, e, f):
        self.theta = learning_rate_function(a, b)

    def estimate_value_function(self, follow_trajectory=True, num_iterations=1000, test_function=None, reset=True, reset_environment=True, stop_if_diverging=True):
        """Estimate the value function of the current policy using the TIDBD algorithm
        theta is the meta learning rate
        """
        if reset:
            self.reset(reset_environment)

        # The history of test_function
        history = np.zeros(num_iterations)
        
        betas = np.zeros((self.num_states, 1))
        H = np.zeros((self.num_states, 1))

        for k in range(num_iterations):
            theta = self.theta(k)
            current_state, _, next_state, reward = self.take_action(follow_trajectory)

            BR = reward + self.gamma * self.V[next_state] - self.V[current_state]

            betas[current_state] += theta * BR * H[current_state]
            learning_rate = np.exp(betas[current_state])
            self.V[current_state] = self.V[current_state][0] + learning_rate * BR
            H[current_state] = H[current_state][0] * max(0, 1 - learning_rate) + learning_rate * BR

            if test_function is not None:
                history[k] = test_function(self.V, None, BR)
                if stop_if_diverging and history[k] > 10 * history[0]:
                    # If we are too large, stop learning
                    history[k:] = float('inf')
                    break

        if test_function is None:
            return self.V

        return history, self.V