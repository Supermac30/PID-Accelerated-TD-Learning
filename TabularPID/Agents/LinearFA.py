"""
Use linear function approximation to learn the optimal policy.
"""
import numpy as np
import itertools
from functools import lru_cache

from TabularPID.Agents.Agents import learning_rate_function

class LinearFuncSpace():
    """A set of basis functions to approximate the value function.
    """
    def value(self, state):
        """Return the value of the basis functions at the given state.
        """
        raise NotImplementedError

class FourierBasis(LinearFuncSpace):
    """A set of Fourier basis functions.
    """
    def __init__(self, env, order):
        self.env = env
        self.order = order
        self.dim = self.env.observation_space.shape[0]
        self.num_features = (self.order + 1) ** self.dim

    def value(self, state):
        """Return the value of the basis functions at the given state.
        """
        all_arrays = list(itertools.product(range(self.order + 1), repeat=self.dim))
        
        if hasattr(state, '__iter__'):
            return np.cos(np.pi * np.array(list(np.array(arr).T @ state for arr in all_arrays))).reshape(-1, 1)
            
        return np.cos(np.pi * np.array(list(np.array(arr).T @ np.array((state,)) for arr in all_arrays))).reshape(-1, 1)
        

class RBF(LinearFuncSpace):
    """A set of radial basis functions.
    """
    def __init__(self, env, num_features, sigma):
        self.env = env
        self.num_features = num_features
        self.sigma = sigma
        self.centers = np.random.uniform(low=self.env.observation_space.low, high=self.env.observation_space.high, size=(self.num_features, self.env.observation_space.shape[0]))

    def value(self, state):
        """Return the value of the basis functions at the given state.
        """
        return np.exp(-np.linalg.norm(state - self.centers, axis=1) / (2 * self.sigma ** 2))

class PolynomialBasis(LinearFuncSpace):
    """A set of polynomial basis functions.
    """
    def __init__(self, env, order):
        self.env = env
        self.order = order
        self.num_features = (self.order + 1) ** self.env.observation_space.shape[0]

    def value(self, state):
        """Return the value of the basis functions at the given state.
        """
        # Check if state is an iterable
        if hasattr(state, '__iter__'):
            return np.array(list(PolynomialBasis.find_all_monomials(tuple(state), self.order))).reshape(-1, 1)

        return np.array(list(PolynomialBasis.find_all_monomials((state,), self.order))).reshape(-1, 1)

    @staticmethod
    @lru_cache(maxsize=128)
    def find_all_monomials(state, order):
        if len(state) == 1:
            return tuple(state[0]**i for i in range(order + 1))
        else:
            return tuple(
                (state[0]**j) * monomial
                for j in range(order + 1)
                for monomial in PolynomialBasis.find_all_monomials(state[1:], order - j)
            )


class LinearTD():
    def __init__(self, env, policy, gamma, basis, kp, ki, kd, alpha, beta, lr_V, lr_z, lr_Vp, adapt_gains=False, meta_lr=0.1, epsilon=0.001, solved_agent=None):
        """
        Initialize the agent.

        Parameters

        env: gym.Env
            The environment to learn from.
        policy: Policy object
            The policy to use.
        gamma: float
            The discount factor.
        basis: LinearFuncSpace
            The basis functions to use.
        lr_V: float
            The learning rate for the P component.
        lr_z: float
            The learning rate for the I component.
        lr_Vp: float
            The learning rate for the D component.
        solved_agent: <has the query_agent method>
            An agent that can be queried with a state to get the v-value for that state, given the policy we are evaluating.
        """
        self.env = env

        # An easy way of checking if the environment is a gym environment, the final piece of hacky code I promise
        self.is_gym_env = hasattr(env, "step")

        self.policy = policy
        self.gamma = gamma
        self.basis = basis
        self.lr_V = lr_V
        self.lr_Vp = lr_Vp
        self.lr_z = lr_z

        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.alpha = alpha
        self.beta = beta

        self.epsilon = 1e-2

        self.current_state = None
        self.solved_agent = solved_agent
    
    def take_action(self):
        action = self.policy.get_action(self.current_state)
        if self.is_gym_env:
            next_state, reward, done, _, _ = self.env.step(action.item())
            if done:
                next_state = self.env.reset()[0]
        else:
            next_state, reward = self.env.take_action(action)
        
        return self.current_state, next_state, reward

    def reset(self, reset_environment=True):
        num_features = self.basis.num_features
        if reset_environment:
            if self.is_gym_env:
                self.current_state = self.env.reset()[0]
            else:
                self.current_state = self.env.reset()

        self.w_V = np.zeros((num_features, 1))
        self.w_Vp = np.zeros((num_features, 1))
        self.w_z = np.zeros((num_features, 1))

        self.running_BR = 0

    def estimate_value_function(self, num_iterations, test_function=None, reset_environment=True, stop_if_diverging=True, adapt_gains=False):
        self.reset(reset_environment)

        # The history of the gains
        self.gain_history = [[] for _ in range(5)]
        self.history = np.zeros((num_iterations))

        for k in range(num_iterations):
            current_state, next_state, reward = self.take_action()
            self.current_state = current_state

            # Update the value function using the floats kp, ki, kd
            current_state_value = self.basis.value(current_state).T @ self.w_V
            next_state_value = self.basis.value(next_state).T @ self.w_V
            current_state_Vp_value = self.basis.value(current_state).T @ self.w_Vp
            current_state_z_value = self.basis.value(current_state).T @ self.w_z

            self.BR = reward + self.gamma * next_state_value - current_state_value
            V_update = current_state_value + self.kp * self.BR \
                + self.kd * (current_state_Vp_value - current_state_value) \
                + self.ki * (self.beta * current_state_z_value + self.alpha * self.BR)
            Vp_update = current_state_value
            z_update = self.beta * current_state_z_value + self.alpha * self.BR

            lr_V, lr_Vp, lr_z = self.lr_V(k), self.lr_Vp(k), self.lr_z(k)

            self.w_V = (1 - lr_V) * self.w_V + lr_V * V_update.item() * self.basis.value(current_state)
            self.w_Vp = (1 - lr_Vp) * self.w_Vp + lr_Vp * Vp_update.item() * self.basis.value(current_state)
            self.w_z = (1 - lr_z) * self.w_z + lr_z * z_update.item() * self.basis.value(current_state)

            if self.solved_agent is not None:
                self.history[k] = self.measure_performance()
                if stop_if_diverging and self.history[k] > 2 * self.history[0]:
                    # If we are too large, stop learning
                    self.history[k:] = float('inf')
                    break

            if adapt_gains:
                self.update_gains()
                self.update_gain_history()

        if self.solved_agent is None:
            return self.w_V

        if adapt_gains:
            self.history = self.history, self.gain_history
        return self.history, self.w_V

    def measure_performance(self):
        """Measure the performance of the agent using the optimal agent.
        """
        distance = 0

        for _ in range(10):
            state, q_value = self.solved_agent.randomly_query_agent()
            distance += abs(q_value - self.query_agent(state))

        # Return the mean of these values
        return np.mean(distance)

    def update_gains(self):
        """Update the gains kp, ki, and kd.
        """
        self.running_BR = 0.5 * self.running_BR + 0.5 * self.BR * self.BR
        normalizer = self.epsilon + self.running_BR

        V = np.dot(self.basis.value(self.current_state).T, self.w_V)
        z = np.dot(self.basis.value(self.current_state).T, self.w_z)
        Vp = np.dot(self.basis.value(self.current_state).T, self.w_Vp)

        self.kp += self.meta_lr * self.BR * self.BR / normalizer
        self.ki += self.meta_lr * self.BR * (self.beta * z + self.alpha * self.BR) / normalizer
        self.kd += self.meta_lr * self.BR * (Vp - V) / normalizer

    def update_gain_history(self):
        """Update the gain history.
        """
        self.gain_history[0].append(self.kp)
        self.gain_history[1].append(self.ki)
        self.gain_history[2].append(self.kd)
        self.gain_history[3].append(self.alpha)
        self.gain_history[4].append(self.beta)

    def query_agent(self, state):
        """Query the agent for the value at a state"""
        return self.basis.value(state).T @ self.w_V
    
    def set_learning_rates(self, a, b, c, d, e, f):
        self.lr_V = learning_rate_function(a, b)
        self.lr_z = learning_rate_function(c, d)
        self.lr_Vp = learning_rate_function(e, f)