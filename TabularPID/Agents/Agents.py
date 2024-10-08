"""
A collection of agents that learn in an RL setting
"""
from Experiments.ExperimentHelpers import find_Vpi
import numpy as np

from TabularPID.MDPs.Policy import Policy
from TabularPID.MDPs.MDP import Control_Q

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
    
    def set_seed(self, seed):
        """Set the seed of the environment and the policy"""
        self.environment.set_seed(seed)
        self.policy.set_seed(seed)

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
        self.V, self.Vp = (np.zeros((self.num_states, 1)) for _ in range(2))
        self.z = np.zeros((self.num_states, 1))
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
        frequency = np.zeros((self.num_states))

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
    
    def randomly_query_agent(self):
        """Returns a random state and the value of the state"""
        state = np.random.randint(self.num_states)
        return state, self.V[state][0]


class PID_Q_TD(Agent):
    """The PID_TD agent with Q functions instead
    """
    def __init__(self, environment, policy, gamma, kp, ki, kd, alpha, beta, learning_rate):
        """learning_rate is either:
            - a triple of three functions, the first updates the P
                component, the second the I component, and the third the D component.
            - a learning rate function for the P component, and the other rates have hard updates.
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

        self.oracle = Control_Q(
            environment.num_states,
            environment.num_actions,
            environment.build_reward_matrix(),
            environment.build_probability_transition_kernel(),
            1,0,0,0,0,
            gamma
        )
    
    def true_BR(self):
        return self.oracle.bellman_operator(self.Q)

    def reset(self, reset_environment=True):
        """Reset parameters to be able to run a new test."""
        self.Q, self.Qp = (np.zeros((self.num_states, self.num_actions)) for _ in range(2))
        self.z = np.zeros((self.num_states, self.num_actions))
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
        frequency = np.zeros((self.num_states))

        # The history of test_function
        history = np.zeros(num_iterations)

        for k in range(num_iterations):
            current_state, action, next_state, reward = self.take_action(follow_trajectory)
            next_action = self.policy.get_action(next_state)
            frequency[current_state] += 1

            BR = reward + self.gamma * self.Q[next_state][next_action] - self.Q[current_state][action]

            learning_rate = self.learning_rate(frequency[current_state])
            update_D_rate = self.update_D_rate(frequency[current_state])
            update_I_rate = self.update_I_rate(frequency[current_state])

            # Update the value function using the floats kp, ki, kd
            z_update = (self.beta * self.z[current_state][0] + self.alpha * BR)
            self.z[current_state][action] = (1 - update_I_rate) * self.z[current_state][action] + update_I_rate * z_update
            update = self.kp * BR + self.ki * z_update + self.kd * (self.Q[current_state][action] - self.Qp[current_state][action])
            self.Qp[current_state][action] = (1 - update_D_rate) * self.Qp[current_state][action] + update_D_rate * self.Q[current_state][action]
            self.Q[current_state][action] = self.V[current_state][action] + learning_rate * update

            if test_function is not None:
                history[k] = test_function(self.V, self.Vp, self.true_BR())
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
    
    def randomly_query_agent(self):
        """Returns a random state and the value of the state"""
        state = np.random.randint(self.num_states)
        return state, self.V[state][0]


class nth_order_TD(Agent):
    """A variation studied for curiosity's sake
    """
    def __init__(self, environment, policy, gamma, eta, alpha, beta, learning_rate, n=3):
        """learning_rate is either:
            - a triple of three functions, the first updates the P
                component, the second the I component, and the third the D component.
            - a learning rate function for the P component, and the other rates have hard updates.
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
        frequency = np.zeros((self.num_states))

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
    
    def randomly_query_agent(self):
        """Returns a random state and the value of the state"""
        state = np.random.randint(self.num_states)
        return state, self.V[state][0]


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
    def estimate_value_function(self, num_iterations=1000, test_function=None, stop_if_diverging=True, follow_trajectory=True, reset_environment=False):
        """Computes V^pi of the inputted policy using TD learning augmented with controllers.
        Takes in Controller objects that the agent will use to control the dynamics of learning.

        If test_function is not None, we record the value of test_function on V, Vp.

        If threshold and test_function are not None, we stop after test_function outputs a value smaller than threshold.

        If stop_if_diverging is True, then when the test_function is 10 times as large as its initial value,
        we stop learning and return a history with the last element being very large
        """
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

    def estimate_value_function(self, controllers=[], num_iterations=1000, test_function=None, stop_if_diverging=True, follow_trajectory=False, reset_environment=False):
        """Computes V^pi of the inputted policy using TD learning augmented with controllers.
        Takes in Controller objects that the agent will use to control the dynamics of learning.

        If test_function is not None, we record the value of test_function on V, Vp.

        If threshold and test_function are not None, we stop after test_function outputs a value smaller than threshold.

        If stop_if_diverging is True, then when the test_function is 10 times as large as its initial value,
        we stop learning and return a history with the last element being very large
        """
        if reset_environment:
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
    def __init__(self, environment, gamma, kp, ki, kd, alpha, beta, learning_rate, double=False):
        """
        kp, ki, kd are floats that are the coefficients of the PID controller
        alpha, beta are floats that are the coefficients of the PID controller
        learning_rate is a float or a tuple of floats (learning_rate, update_I_rate, update_D_rate)
        decay is the float that we multiply the exploration rate by at each iteration.
        double is true when we use Qp to evaluate the value of the current state in the BR
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
        self.double = double
        self.reset()

    def reset(self, reset_environment=True):
        """Reset parameters to be able to run a new test."""
        if reset_environment:
            self.environment.reset()

        self.Q = np.zeros((self.num_states, self.num_actions))
        self.Qp = np.zeros((self.num_states, self.num_actions))
        self.z = np.zeros((self.num_states, self.num_actions))
        self.policy = Policy(self.num_actions, self.num_states, self.environment.prg, None)

    def estimate_value_function(self, num_iterations=1000, test_function=None, stop_if_diverging=True, follow_trajectory=False, reset_environment=True, measure_time=False):
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

            if self.double:
                best_action = np.argmax(self.Qp[next_state])
            else:
                best_action = np.argmax(self.Q[next_state])

            # Update the value function using the floats kp, ki, kd
            BR = reward + self.gamma * self.Q[next_state][best_action] - self.Q[current_state][action]
            z_update = self.beta * self.z[current_state][action] + self.alpha * BR
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

            if measure_time and history[k] / history[0] < 0.2:
                break

        if test_function is None:
            return self.Q

        return history, self.Q
    
    def set_learning_rates(self, a, b, c, d, e, f):
        self.learning_rate = learning_rate_function(a, b)
        self.update_I_rate = learning_rate_function(c, d)
        self.update_D_rate = learning_rate_function(e, f)


class ControlledDoubleQLearning(Agent):
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

        self.Q_A = np.zeros((self.num_states, self.num_actions))
        self.Qp_A = np.zeros((self.num_states, self.num_actions))
        self.z_A = np.zeros((self.num_states, self.num_actions))
        self.Q_B = np.zeros((self.num_states, self.num_actions))
        self.Qp_B = np.zeros((self.num_states, self.num_actions))
        self.z_B = np.zeros((self.num_states, self.num_actions))

        self.policy = Policy(self.num_actions, self.num_states, self.environment.prg, None)

    def estimate_value_function(self, num_iterations=1000, test_function=None, stop_if_diverging=True, follow_trajectory=False, reset_environment=True):
        """Use the Q-learning algorithm to estimate the value function.
        That is, create a matrix of size num_states by num_actions, Q, and update it according to the Q-learning update rule.
        """
        self.reset(reset_environment)
        # A vector storing the number of times we have seen a state.
        frequency_A = np.zeros((self.num_states, 1))
        frequency_B = np.zeros((self.num_states, 1))

        # The history of test_function
        history = np.zeros(num_iterations)

        for k in range(num_iterations):
            current_state, action, next_state, reward = self.take_action(follow_trajectory, is_q=True)

            if np.random.rand() < 0.5:
                frequency_A[current_state] += 1
                learning_rate = self.learning_rate(frequency_A[current_state])
                update_D_rate = self.update_D_rate(frequency_A[current_state])
                update_I_rate = self.update_I_rate(frequency_A[current_state])
                # Update the value function using the floats kp, ki, kd
                best_action = np.argmax(self.Q_A[next_state])
                BR = reward + self.gamma * self.Q_B[next_state][best_action] - self.Q_A[current_state][action]
                z_update = self.beta * self.z_A[current_state][action] + self.alpha * BR
                self.z_A[current_state][action] = (1 - update_I_rate) * self.z_A[current_state][action] + update_I_rate * z_update
                update = self.kp * BR + self.ki * z_update + self.kd * (self.Q_A[current_state][action] - self.Qp_A[current_state][action])
                self.Qp_A[current_state][action] = (1 - update_D_rate) * self.Qp_A[current_state][action] + update_D_rate * self.Q_A[current_state][action]
                self.Q_A[current_state][action] += learning_rate * update
            else:
                frequency_B[current_state] += 1
                learning_rate = self.learning_rate(frequency_B[current_state])
                update_D_rate = self.update_D_rate(frequency_B[current_state])
                update_I_rate = self.update_I_rate(frequency_B[current_state])
                # Update the value function using the floats kp, ki, kd
                best_action = np.argmax(self.Q_B[next_state])
                BR = reward + self.gamma * self.Q_A[next_state][best_action] - self.Q_B[current_state][action]
                z_update = self.beta * self.z_B[current_state][action] + self.alpha * BR
                self.z_B[current_state][action] = (1 - update_I_rate) * self.z_B[current_state][action] + update_I_rate * z_update
                update = self.kp * BR + self.ki * z_update + self.kd * (self.Q_B[current_state][action] - self.Qp_B[current_state][action])
                self.Qp_B[current_state][action] = (1 - update_D_rate) * self.Qp_B[current_state][action] + update_D_rate * self.Q_B[current_state][action]
                self.Q_B[current_state][action] += learning_rate * update

            if test_function is not None:
                history[k] = test_function(self.Q_A, self.Qp_A, BR)
                if stop_if_diverging and history[k] > 2 * history[0]:
                    # If we are too large, stop learning
                    history[k:] = float('inf')
                    break

        if test_function is None:
            return self.Q_A
        return history, self.Q_A

    def set_learning_rates(self, a, b, c, d, e, f):
        self.learning_rate = learning_rate_function(a, b)
        self.update_I_rate = learning_rate_function(c, d)
        self.update_D_rate = learning_rate_function(e, f)
