"""
Use linear function approximation to learn the optimal policy.
"""
import numpy as np

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
        self.num_features = (order + 1) ** self.env.observation_space.shape[0]

    def value(self, state):
        """Return the value of the basis functions at the given state.
        """
        return np.cos(np.pi * np.dot(self.order * state, self.env.observation_space.high))

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
        self.num_features = (order + 1) ** self.env.observation_space.shape[0]

    def value(self, state):
        """Return the value of the basis functions at the given state.
        """
        return np.prod(np.power(state, np.arange(self.order + 1)), axis=1)

class LinearFA():
    def __init__(self, env, gamma, basis, kp, ki, kd, alpha, beta, lr_V, lr_z, lr_Vp):
        """
        Initialize the agent.

        Parameters

        env: gym.Env
            The environment to learn from.
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
        """
        self.env = env
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

    def reset(self, reset_environment=True):
        if reset_environment:
            self.current_state = self.env.reset()[0]
        self.w_V = np.zeros((self.env.observation_space.shape[0], 1))
        self.w_Vp = np.zeros((self.env.observation_space.shape[0], 1))
        self.w_z = np.zeros((self.env.observation_space.shape[0], 1))

        self.running_BR = 0

    def estimate_value_function(self, num_iterations, test_function, follow_trajectory=False, reset_environment=True, stop_if_diverging=True, adapt_gains=False):
        self.reset(reset_environment)

        # The history of test_function
        history = np.zeros(num_iterations)
        self.gain_history = [[] for _ in range(5)]

        for k in range(num_iterations):
            self.epsilon *= self.decay
            self.policy.set_policy_from_Q(self.Q, self.epsilon)
            current_state, action, next_state, reward = self.take_action(follow_trajectory, on_policy=False)
            self.current_state = current_state

            # Update the value function using the floats kp, ki, kd
            current_state_value = np.dot(self.basis.value(current_state), self.w_V)
            next_state_value = np.dot(self.basis.value(next_state), self.w_V)
            current_state_Vp_value = np.dot(self.basis.value(current_state), self.w_Vp)
            current_state_z_value = np.dot(self.basis.value(current_state), self.w_z)

            self.BR = reward + self.gamma * next_state_value - current_state_value
            V_update = self.w_V + self.kp * self.BR \
                + self.kd * (current_state_Vp_value - current_state_value) \
                + self.ki * (self.beta * current_state_z_value + self.alpha * self.BR)
            Vp_update = current_state_value
            z_update = self.beta * current_state_z_value + self.alpha * self.BR

            w_V += self.lr_V * V_update * self.basis.value(current_state)
            w_Vp += self.lr_Vp * Vp_update * self.basis.value(current_state)
            w_z += self.lr_z * z_update * self.basis.value(current_state)

            if test_function is not None:
                history[k] = test_function(self.w_V, self.w_Vp, self.BR)
                if stop_if_diverging and history[k] > 2 * history[0]:
                    # If we are too large, stop learning
                    history[k:] = float('inf')
                    break

            if adapt_gains:
                self.update_gains()
                self.update_gain_history()

        if test_function is None:
            return self.Q

        if adapt_gains:
            history = history, self.gain_history
        return history, self.Q

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