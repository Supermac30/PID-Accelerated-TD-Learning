"""
A collection of agents that learn in an RL setting
"""
import numpy as np

from TabularPID.MDPs.Policy import Policy

def learning_rate_function(alpha, N):
    """Return the learning rate function alpha(k) parameterized by alpha and N.
    If N is infinity, return a constant function that outputs alpha.
    """
    if N == 'inf':
        return lambda k: alpha
    return lambda k: min(alpha, N/(k + 1))

class Agent():
    """An abstract class that represents the agent interacting
    in the environment

    self.environment: An Environment object that the agent is interacting with
    self.policy: A matrix of size self.environment.num_states x self.environment.num_actions,
    where the sum of each row is one. An off-policy agent can safely ignore this.
    """
    def __init__(self, environment, policy, gamma):
        self.gamma = gamma
        self.environment = environment
        self.num_actions = environment.num_actions
        self.num_states = environment.num_states
        self.policy = policy

    def reset(self, reset_environment=True):
        """Reset the agent to its initial state."""
        if reset_environment:
            self.environment.reset()

    def estimate_value_function(self, follow_trajectory=True, num_iterations=1000, test_function=None):
        """Estimate the value function. This could be V^pi or V*"""
        raise NotImplementedError

    def take_action(self, follow_trajectory, is_q=False):
        """Use the current policy to play an action in the environment.
        Return a 4-tuple of (current_state, action, next_state, reward).
        """
        if follow_trajectory:
            state = self.environment.current_state
            action = self.policy.get_action(state)
        else:
            if is_q:
                state, action = self.policy.get_random_sample()
            else:
                state, action = self.policy.get_on_policy_sample()
            self.environment.current_state = state

        next_state, reward = self.environment.take_action(action)

        return state, action, next_state, reward

    def set_learning_rates(self, a, b, c, d, e, f):
        """
        The learning rates are parameterized by six values for convenience in the grid search.

        This in all cases (except in the case of TIDBD, speedy, and zap) represent
        the learning rates of P, I, and D components respectively.
        """
        raise NotImplementedError


    def generate_episode(self, num_steps=1000, reset_environment=True):
        """Return a full episode following the policy matrix policy

        The returned object is a trajectory represented as a list of 4-tuples
        (state, action, reward, first_time_seen),
        where first_time_seen is True if and only if this is the first
        time we have visited the state.
        """
        if reset_environment:
            self.environment.reset()
        trajectory = []
        seen = set()
        for _ in range(num_steps):
            state = self.environment.current_state
            first_time_seen = False
            if state not in seen:
                seen.add(state)
                first_time_seen = True

            # Choose and perform an action
            action, reward = self.environment.take_action(action)[1]

            trajectory.append((state, action, reward, first_time_seen))

        return trajectory


class MonteCarloPE(Agent):
    def __init__(self, environment, policy, gamma):
        super().__init__(environment, policy, gamma)

    def estimate_value_function(self, num_iterations=1000, test_function=None):
        """For the purpose of debugging, return a naive monte carlo estimate of V_pi
        The algorithm can be found in Sutton and Barto page 99, Monte Carlo Exploring Starts.

        num_steps is the number of steps we run each episode for.
        """
        V = np.zeros((self.num_states, 1))
        G = 0

        trajectory = self.generate_episode(num_iterations=num_iterations)
        for state, _, reward, first_time_seen in trajectory[::-1]:
            G = self.gamma * G + reward
            if first_time_seen:
                V[state] = G

        return [], V

class PID_TD(Agent):
    """The bread and butter of our work, this is the agent that
    can be augmented with controllers, namely the PID controller.

    The updates to the past value of Vp are made in a soft fashion,
    unlike the Hard_PID_TD agent below.
    """
    def __init__(self, environment, policy, gamma, kp, ki, kd, alpha, beta, learning_rate):
        """learning_rate is either:
            - a triple of three functions, the first updates the P
                component, the second the I component, and the third the D component.
            - a learning rate function for the P component, and the other rates have hard updates.

            Note: From a design perspective, this was a bad choice. In the summer,
            I should come back and clean up this part of the code
        """
        super().__init__(environment, policy, gamma)
        if type(learning_rate) == type(()):
            self.learning_rate = learning_rate[0]
            self.update_I_rate = learning_rate[1]
            self.update_D_rate = learning_rate[2]
        else:
            self.learning_rate = learning_rate
            self.update_D_rate = self.update_I_rate = lambda _: 1  # Hard updates
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.alpha = alpha
        self.beta = beta
        self.reset()

    def reset(self, reset_environment=True):
        """Reset parameters to be able to run a new test."""
        self.V, self.Vp, self.z = (np.zeros((self.num_states, 1)) for _ in range(3))
        if reset_environment:
            self.environment.reset()

    def estimate_value_function(self, num_iterations=1000, test_function=None, stop_if_diverging=True, follow_trajectory=True, reset=True, reset_environment=True):
        """Computes V^pi of the inputted policy using TD learning augmented with controllers.
        Takes in Controller objects that the agent will use to control the dynamics of learning.

        If test_function is not None, we record the value of test_function on V, Vp.

        If threshold and test_function are not None, we stop after test_function outputs a value smaller than threshold.

        If stop_if_diverging is True, then when the test_function is 10 times as large as its initial value,
        we stop learning and return a history with the last element being very large
        """
        if reset:
            self.reset(reset_environment)

        # A vector storing the number of times we have seen a state.
        frequency = np.zeros((self.num_states, 1))

        # The history of test_function
        history = np.zeros(num_iterations)

        for k in range(num_iterations):
            current_state, _, next_state, reward = self.take_action(follow_trajectory)
            frequency[current_state] += 1

            BR = reward + self.gamma * self.V[next_state] - self.V[current_state]

            learning_rate = self.learning_rate(frequency[current_state])
            update_D_rate = self.update_D_rate(frequency[current_state])
            update_I_rate = self.update_I_rate(frequency[current_state])

            # Update the value function using the floats kp, ki, kd
            z_update = (self.beta * self.z[current_state][0] + self.alpha * BR)
            self.z[current_state] = (1 - update_I_rate) * self.z[current_state][0] + update_I_rate * z_update
            update = self.kp * BR + self.ki * z_update + self.kd * (self.V[current_state][0] - self.Vp[current_state][0])
            self.Vp[current_state] = (1 - update_D_rate) * self.Vp[current_state][0] + update_D_rate * self.V[current_state][0]
            self.V[current_state] = self.V[current_state][0] + learning_rate * update

            if test_function is not None:
                history[k] = test_function(self.V, self.Vp, BR)
                if stop_if_diverging and history[k] > 10 * history[0]:
                    # If we are too large, stop learning
                    history[k:] = float('inf')
                    break

        if test_function is None:
            return self.V

        return history, self.V

    def set_learning_rates(self, a, b, c, d, e, f):
        self.learning_rate = learning_rate_function(a, b)
        self.update_I_rate = learning_rate_function(c, d)
        self.update_D_rate = learning_rate_function(e, f)


class Hard_PID_TD(PID_TD):
    """The bread and butter of our work, this is the agent that
    can be augmented with controllers, namely the PID controller.

    self.learning_rate: A function that takes in the current iteration number
    and returns a learning rate

    This is SoftControlledTDLearning with a constant update rate of 1.
    """
    def __init__(self, environment, policy, gamma, kp, ki, kd, alpha, beta, learning_rate):
        super().__init__(
            environment,
            policy,
            gamma,
            kp,
            ki,
            kd,
            alpha,
            beta,
            learning_rate
        )

class PID_TD_with_momentum(PID_TD):
    """The old flawed updates that seemed to work better
    """
    def estimate_value_function(self, num_iterations=1000, test_function=None, stop_if_diverging=True, follow_trajectory=True):
        """Computes V^pi of the inputted policy using TD learning augmented with controllers.
        Takes in Controller objects that the agent will use to control the dynamics of learning.

        If test_function is not None, we record the value of test_function on V, Vp.

        If threshold and test_function are not None, we stop after test_function outputs a value smaller than threshold.

        If stop_if_diverging is True, then when the test_function is 10 times as large as its initial value,
        we stop learning and return a history with the last element being very large
        """
        self.reset()
        # A vector storing the number of times we have seen a state.
        frequency = np.zeros((self.num_states, 1))

        # The history of test_function
        history = np.zeros(num_iterations)

        for k in range(num_iterations):
            current_state, _, next_state, reward = self.take_action(follow_trajectory)

            frequency[current_state] += 1

            BR = reward + self.gamma * self.V[next_state] - self.V[current_state]

            learning_rate = self.learning_rate(frequency[current_state])
            update_D_rate = self.update_D_rate(frequency[current_state])
            update_I_rate = self.update_I_rate(frequency[current_state])

            # Update the value function using the floats kp, ki, kd
            self.z[current_state] = (1 - update_I_rate) * self.z[current_state][0] + update_I_rate * (self.beta * self.z[current_state][0] + self.alpha * BR)
            update = self.ki * self.z + self.kd * (self.V - self.Vp)
            self.Vp = (1 - update_D_rate) * self.Vp + update_D_rate * self.V
            self.V[current_state] += learning_rate * self.kp * BR
            self.V += learning_rate * update

            if test_function is not None:
                history[k] = test_function(self.V, self.Vp, BR)
                if stop_if_diverging and history[k] > 2 * history[0]:
                    # If we are too large, stop learning
                    history[k:] = float('inf')
                    break

        if test_function is None:
            return self.V

        return history, self.V


class FarSighted_PID_TD(PID_TD):
    """Soft updates, with V_k - V_{k - N} for some large N
    """
    def __init__(self, environment, policy, gamma, kp, ki, kd, alpha, beta, learning_rate, delay):
        self.delay = delay
        super().__init__(
            environment,
            policy,
            gamma,
            kp,
            ki,
            kd,
            alpha,
            beta,
            learning_rate
        )

    def estimate_value_function(self, controllers=[], num_iterations=1000, test_function=None, stop_if_diverging=True, follow_trajectory=False):
        """Computes V^pi of the inputted policy using TD learning augmented with controllers.
        Takes in Controller objects that the agent will use to control the dynamics of learning.

        If test_function is not None, we record the value of test_function on V, Vp.

        If threshold and test_function are not None, we stop after test_function outputs a value smaller than threshold.

        If stop_if_diverging is True, then when the test_function is 10 times as large as its initial value,
        we stop learning and return a history with the last element being very large
        """
        self.environment.reset()
        # A vector storing the number of times we have seen a state.
        frequency = np.zeros((self.num_states, 1))

        # The history of test_function
        history = np.zeros(num_iterations)

        # To look back N steps, we store the full learning history
        history_of_Vs = [[] for _ in range(self.num_states)]

        for k in range(num_iterations):
            current_state, _, next_state, reward = self.take_action(follow_trajectory)

            frequency[current_state] += 1

            # An estimate of the bellman update
            BR = np.zeros((self.num_states, 1))
            BR[current_state] = reward + self.gamma * self.V[next_state] - self.V[current_state]

            learning_rate = self.learning_rate(frequency[current_state])
            update_D_rate = self.update_D_rate(frequency[current_state])
            update_I_rate = self.update_I_rate(frequency[current_state])
            update = sum(map(lambda n: n.evaluate_controller(BR, self.V, self.Vp, update_I_rate), controllers))

            if frequency[current_state] > self.delay:
                self.Vp[current_state] = self.Vp[current_state] * (1 - update_D_rate) + update_D_rate * history_of_Vs[current_state][-self.delay]
            self.V = self.V + learning_rate * update
            history_of_Vs[current_state].append(self.V[current_state])

            if test_function is not None:
                history[k] = test_function(self.V, self.Vp, BR)
                if stop_if_diverging and history[k] > 2 * history[0]:
                    # If we are too large, stop learning
                    history[k:] = float('inf')
                    break

        if test_function is None:
            return self.V

        return history, self.V

class ControlledQLearning(Agent):
    def __init__(self, environment, gamma, kp, ki, kd, alpha, beta, learning_rate):
        """
        kp, ki, kd are floats that are the coefficients of the PID controller
        alpha, beta are floats that are the coefficients of the PID controller
        learning_rate is a float or a tuple of floats (learning_rate, update_I_rate, update_D_rate)
        decay is the float that we multiply the exploration rate by at each iteration.
        """
        super().__init__(environment, None, gamma)
        if type(learning_rate) == type(()):
            self.learning_rate = learning_rate[0]
            self.update_I_rate = learning_rate[1]
            self.update_D_rate = learning_rate[2]
        else:
            self.learning_rate = learning_rate
            self.update_D_rate = self.update_I_rate = lambda _: 1  # Hard updates
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.alpha = alpha
        self.beta = beta
        self.reset()

    def reset(self, reset_environment=True):
        """Reset parameters to be able to run a new test."""
        if reset_environment:
            self.environment.reset()

        self.Q = np.zeros((self.num_states, self.num_actions))
        self.Qp = np.zeros((self.num_states, self.num_actions))
        self.z = np.zeros((self.num_states, self.num_actions))
        self.policy = Policy(self.num_actions, self.num_states, self.environment.prg, None)

    def estimate_value_function(self, num_iterations=1000, test_function=None, stop_if_diverging=True, follow_trajectory=False, reset_environment=True):
        """Use the Q-learning algorithm to estimate the value function.
        That is, create a matrix of size num_states by num_actions, Q, and update it according to the Q-learning update rule.
        """
        self.reset(reset_environment)
        # A vector storing the number of times we have seen a state.
        frequency = np.zeros((self.num_states, 1))

        # The history of test_function
        history = np.zeros(num_iterations)

        for k in range(num_iterations):
            current_state, action, next_state, reward = self.take_action(follow_trajectory, is_q=True)

            frequency[current_state] += 1

            learning_rate = self.learning_rate(frequency[current_state])
            update_D_rate = self.update_D_rate(frequency[current_state])
            update_I_rate = self.update_I_rate(frequency[current_state])

            # Update the value function using the floats kp, ki, kd
            BR = reward + self.gamma * np.max(self.Q[next_state]) - self.Q[current_state][action]
            z_update = (self.beta * self.z[current_state][action] + self.alpha * BR)
            self.z[current_state][action] = (1 - update_I_rate) * self.z[current_state][action] + update_I_rate * z_update
            update = self.kp * BR + self.ki * z_update + self.kd * (self.Q[current_state][action] - self.Qp[current_state][action])
            self.Qp[current_state][action] = (1 - update_D_rate) * self.Qp[current_state][action] + update_D_rate * self.Q[current_state][action]
            self.Q[current_state][action] += learning_rate * update

            if test_function is not None:
                history[k] = test_function(self.Q, self.Qp, BR)
                if stop_if_diverging and history[k] > 2 * history[0]:
                    # If we are too large, stop learning
                    history[k:] = float('inf')
                    break

        if test_function is None:
            return self.Q

        return history, self.Q
    
    def set_learning_rates(self, a, b, c, d, e, f):
        self.learning_rate = learning_rate_function(a, b)
        self.update_I_rate = learning_rate_function(c, d)
        self.update_D_rate = learning_rate_function(e, f)



class ControlledSARSA(Agent):
    def __init__(self, environment, policy, gamma, kp, ki, kd, alpha, beta, learning_rate):
        super().__init__(
            environment,
            policy,
            gamma,
            kp,
            ki,
            kd,
            alpha,
            beta,
            learning_rate
        )

        self.V = np.zeros((self.num_states, 1))
        self.Vp = np.zeros((self.num_states, 1))
        self.z = np.zeros((self.num_states, 1))

    def estimate_value_function(self, num_iterations=1000, test_function=None, stop_if_diverging=True, N=1000):
        """Use the SARSA algorithm to estimate the value function.
        That is, create a vector V of size num_states, then repeatedly do the following num_iterations times:
        - perform policy evaluation using the PID-TD algorithm above for N steps
        - update the policy to greedily follow V, then repeat the above
        """
        pass
        # TODO: Implement