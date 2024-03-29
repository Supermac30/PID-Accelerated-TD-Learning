"""
Use linear function approximation to learn the optimal policy.
"""
import numpy as np
import itertools
from functools import lru_cache

from TabularPID.Agents.Agents import learning_rate_function
from TabularPID.Agents.Tiles.tiles import tiles

class LinearFuncSpace():
    """A set of basis functions to approximate the value function.
    """
    def __init__(self, env, order, is_q=False):
        self.env = env

        self.clip_value = 1e5

        # Set observation_space_high to be self.env.observation_space where any elements larger than self.clip_value are replaced with self.clip_value
        observation_space_high = np.array([np.sign(x) * self.clip_value if abs(x) > self.clip_value else x for x in self.env.observation_space.high])
        observation_space_low = np.array([np.sign(x) * self.clip_value if abs(x) > self.clip_value else x for x in self.env.observation_space.low])

        self.range = observation_space_high - observation_space_low
        self.mean = (observation_space_high + observation_space_low) / 2
    
        self.order = order
        self.dim = self.env.observation_space.shape[0]
        
        if is_q:
            num_actions = self.env.action_space.n
            self.range = np.append(self.range, num_actions)
            self.mean = np.append(self.mean, num_actions / 2)
            self.dim += 1

    def value(self, state):
        """Return the value of the basis functions at the given state, normalizing the input
        """
        if not hasattr(state, '__iter__'):
            state = np.array((state,))
        # Clip inputs by the clip_value
        state = np.array([np.sign(x) * self.clip_value if abs(x) > self.clip_value else x for x in state])

        return self.base_value(2 * ((state - self.mean) / self.range))
    
    def base_value(self, state):
        """Return the value of the basis functions at the given state.
        """
        raise NotImplementedError

class FourierBasis(LinearFuncSpace):
    """A set of Fourier basis functions.
    """
    def __init__(self, env, order, is_q=False):
        super().__init__(env, order, is_q)
        self.num_features = (self.order + 1) ** self.dim

        # For the Fourier basis with only cosines, we only project the input to [0, 1] instead of [-1, 1].
        self.mean = 0
        self.range *= 2

    def base_value(self, state):
        """Return the value of the basis functions at the given state.
        """
        all_arrays = list(itertools.product(range(self.order + 1), repeat=self.dim))

        return np.cos(
            np.pi * np.array(list(
                np.array(arr).T @ state
                for arr in all_arrays
            ))
        ).reshape(-1, 1)


class PolynomialBasis(LinearFuncSpace):
    """A set of polynomial basis functions.
    """
    def __init__(self, env, order, is_q=False):
        super().__init__(env, order, is_q)
        self.num_features = (self.order + 1) ** self.env.observation_space.shape[0]

    def base_value(self, state):
        """Return the value of the basis functions at the given state.
        """
        # Check if state is an iterable
        return np.array(list(PolynomialBasis.find_all_monomials(tuple(state), self.order))).reshape(-1, 1)

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


class TileCodingBasis(LinearFuncSpace):
    """ From https://github.com/amarack/python-rl/blob/master/pyrl/basis/tilecode.py, which
        in turn is from Rich Sutton's implementation,
        http://incompleteideas.net/rlai.cs.ualberta.ca/RLAI/RLtoolkit/tiles.html
    """

    def __init__(self, env, order, num_tiles=100, is_q=False):
        super().__init__(env, order, is_q)
        self.num_tiles = num_tiles
        self.num_features = self.order

    def value(self, state):
        indices = tiles(self.num_tiles, self.order, state)
        result = np.zeros((self.order,))
        result[indices] = 1.0
        return result.reshape(-1, 1)


class TrivialBasis(LinearFuncSpace):
    """A basis that simply returns the state in a one-hot-encoding for the purpose of debugging."""

    def __init__(self, env, order, is_q=False):
        super().__init__(env, order, is_q)
        self.num_features = self.env.observation_space.n

    def value(self, state):
        # if state is not a numpy array containing only one element, raise an exception
        if state.shape != (1,):
            raise Exception("TrivialBasis only works for environments with a single state variable")
        result = np.zeros((self.env.observation_space.n,))
        result[state] = 1.0
        return result.reshape(-1, 1)


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

        self.starting_kp = kp
        self.starting_ki = ki
        self.starting_kd = kd
        self.starting_alpha = alpha
        self.starting_beta = beta

        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.alpha = alpha
        self.beta = beta

        self.current_state = self.env.reset()
        self.solved_agent = solved_agent

        # Gain adaptation params
        self.adapt_gains = adapt_gains
        self.meta_lr = meta_lr
        self.epsilon = epsilon
    
    def take_action(self):
        action = self.policy.get_action(self.current_state)
        current_state = self.current_state
        if self.is_gym_env:
            next_state, reward, done, _, _ = self.env.step(action.item())
            if done:
                next_state = self.env.reset()[0]
        else:
            next_state, reward = self.env.take_action(action)

        self.current_state = next_state
        return current_state, next_state, reward
    
    def set_seed(self, seed):
        """Set the seed for the environment and the agent.
        """
        if self.is_gym_env:
            self.env.seed(seed)
        else:
            self.env.set_seed(seed)
        self.policy.set_seed(seed)

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

        self.kp = self.starting_kp
        self.ki = self.starting_ki
        self.kd = self.starting_kd
        self.alpha = self.starting_alpha
        self.beta = self.starting_beta

        self.running_BR = 0
        self.num_steps = 0

    def estimate_value_function(self, num_iterations, test_function=None, reset_environment=True, stop_if_diverging=True):
        self.reset(reset_environment)

        # The history of the gains
        self.gain_history = np.zeros((num_iterations // (num_iterations // 100), 5))
        self.history = np.zeros(num_iterations // (num_iterations // 100))
        index = 0

        for k in range(num_iterations):
            self.num_steps += 1
            current_state, next_state, reward = self.take_action()

            # Update the value function using the floats kp, ki, kd
            current_state_value = self.basis.value(current_state).T @ self.w_V
            next_state_value = self.basis.value(next_state).T @ self.w_V
            current_state_Vp_value = self.basis.value(current_state).T @ self.w_Vp
            current_state_z_value = self.basis.value(current_state).T @ self.w_z

            self.BR = reward + self.gamma * next_state_value - current_state_value
            V_update = current_state_value + self.kp * self.BR \
                + self.kd * (current_state_value - current_state_Vp_value) \
                + self.ki * (self.beta * current_state_z_value + self.alpha * self.BR)
            Vp_update = current_state_value
            z_update = self.beta * current_state_z_value + self.alpha * self.BR

            lr_V, lr_Vp, lr_z = self.lr_V(k), self.lr_Vp(k), self.lr_z(k)
            self.learning_rate = lr_V

            self.w_V += lr_V * (V_update.item() - current_state_value.item()) * self.basis.value(current_state)
            self.w_Vp += lr_Vp * (Vp_update.item() - current_state_Vp_value.item()) * self.basis.value(current_state)
            self.w_z += lr_z * (z_update.item() - current_state_z_value.item()) * self.basis.value(current_state)

            new_next_state_value = self.basis.value(next_state).T @ self.w_V
            new_current_state_value = self.basis.value(current_state).T @ self.w_V
            self.next_BR = reward + self.gamma * new_next_state_value - new_current_state_value

            if self.solved_agent is not None and k % (num_iterations // 100) == 0:
                self.history[index] = self.solved_agent.measure_performance(self.query_agent)
                if self.adapt_gains:
                    self.update_gain_history(index)
                if stop_if_diverging and self.history[index] > 2 * self.history[0]:
                    # If we are too large, stop learning
                    self.history[index:] = float('inf')
                    break

                index += 1

            if self.adapt_gains:
                self.update_gains()

        if self.solved_agent is None:
            return self.w_V

        self.history = np.array(self.history)

        if self.adapt_gains:
            return self.history, self.gain_history, self.w_V

        return self.history, self.w_V

    def update_gains(self):
        """Update the gains kp, ki, and kd.
        """
        scale = 1 / self.num_steps
        self.running_BR = (1 - scale) * self.running_BR + scale * self.BR * self.BR
        normalizer = self.epsilon + self.running_BR

        current_state_value = self.basis.value(self.current_state)
        l2_basis = np.dot(current_state_value.T, current_state_value)

        V = np.dot(current_state_value.T, self.w_V)
        z = np.dot(current_state_value.T, self.w_z)
        Vp = np.dot(current_state_value.T, self.w_Vp)

        self.kp += self.learning_rate * self.meta_lr * self.next_BR * self.BR * l2_basis / normalizer
        self.ki += self.learning_rate * self.meta_lr * self.next_BR * (self.beta * z + self.alpha * self.BR) * l2_basis / normalizer
        self.kd += self.learning_rate * self.meta_lr * self.next_BR * (V - Vp) * l2_basis / normalizer

    def update_gain_history(self, index):
        """Update the gain history.
        """
        self.gain_history[index][0] = self.kp
        self.gain_history[index][1] = self.ki
        self.gain_history[index][2] = self.kd
        self.gain_history[index][3] = self.alpha
        self.gain_history[index][4] = self.beta

    def query_agent(self, state):
        """Query the agent for the value at a state"""
        return self.basis.value(state).T @ self.w_V
    
    def set_learning_rates(self, a, b, c, d, e, f):
        self.lr_V = learning_rate_function(a, b)
        self.lr_z = learning_rate_function(c, d)
        self.lr_Vp = learning_rate_function(e, f)
