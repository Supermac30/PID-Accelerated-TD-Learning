"""
A collection of agents that learn in an RL setting
"""
import numpy as np

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

    def reset(self):
        """Reset the agent to its initial state."""
        self.environment.reset()

    def estimate_value_function(self):
        """Estimate the value function. This could be V^pi or V*"""
        raise NotImplementedError

    def pick_action(self):
        """Use the current policy to pick an action"""
        # Current state
        state = self.environment.current_state

        random_number = np.random.uniform()
        action = 0
        total = self.policy[state][0]
        while total < random_number:
            action += 1
            total += self.policy[state][action]

        return action

    def take_action(self):
        """Use the current policy to play an action in the environment.
        Return the action, next_state, and the reward.
        """
        action = self.pick_action()
        return action, *self.environment.take_action(action)

    def generate_episode(self, num_steps=1000):
        """Return a full episode following the policy matrix policy

        The returned object is a trajectory represented as a list of 4-tuples
        (state, action, reward, first_time_seen),
        where first_time_seen is True if and only if this is the first
        time we have visited the state.
        """
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
        for state, action, reward, first_time_seen in trajectory[::-1]:
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

    def reset(self):
        """Reset parameters to be able to run a new test."""
        self.V, self.Vp, self.z = (np.zeros((self.num_states, 1)) for _ in range(3))
        self.environment.reset()

    def estimate_value_function(self, controllers=[], num_iterations=1000, test_function=None, stop_if_diverging=True):
        """Computes V^pi of the inputted policy using TD learning augmented with controllers.
        Takes in Controller objects that the agent will use to control the dynamics of learning.

        If test_function is not None, we record the value of test_function on V, Vp.

        If threshold and test_function are not None, we stop after test_function outputs a value smaller than threshold.

        If stop_if_diverging is True, then when the test_function is 10 times as large as its initial value,
        we stop learning and return a history with the last element being very large
        """
        # A vector storing the number of times we have seen a state.
        frequency = np.zeros((self.num_states, 1))

        # The history of test_function
        history = np.zeros(num_iterations)

        for k in range(num_iterations):
            current_state = self.environment.current_state
            _, next_state, reward = self.take_action()

            frequency[current_state] += 1

            # An estimate of the bellman update
            BR = np.zeros((self.num_states, 1))
            BR[current_state] = reward + self.gamma * self.V[next_state] - self.V[current_state]

            learning_rate = self.learning_rate(frequency[current_state])
            update_D_rate = self.update_D_rate(frequency[current_state])
            update_I_rate = self.update_I_rate(frequency[current_state])

            # Deprecated, but I'm keeping it here for now to allow the novel controller experiments to work
            # TODO: Remove this later.
            if len(controllers) == 0:
                update = sum(map(lambda n: n.evaluate_controller(BR, self.V, self.Vp, update_I_rate), controllers))

            # Update the value function using the floats kp, ki, kd
            self.z = (1 - update_I_rate) * self.z + update_I_rate * (self.beta * self.z + self.alpha * BR)
            update = self.kp * BR + self.ki * self.z + self.kd * (self.V - self.Vp)
            self.Vp[current_state] = (1 - update_D_rate) * self.Vp[current_state] + update_D_rate * self.V[current_state]
            self.V = self.V + learning_rate * update

            if test_function is not None:
                history[k] = test_function(self.V, self.Vp, BR)
                if stop_if_diverging and history[k] > 2 * history[0]:
                    # If we are too large, stop learning
                    history[k:] = float('inf')
                    break

        if test_function is None:
            return self.V

        return history, self.V


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

    def estimate_value_function(self, controllers=[], num_iterations=1000, test_function=None, stop_if_diverging=True):
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
            current_state = self.environment.current_state
            _, next_state, reward = self.take_action()

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
                if stop_if_diverging and history[k] > 5 * history[0]:
                    # If we are too large, stop learning
                    history[k:] = float('inf')
                    break

        if test_function is None:
            return self.V

        return history, self.V

class ControlledQLearning(Agent):
    # TODO: Implement
    pass

class ControlledSARSA(Agent):
    # TODO: Implement
    pass