"""
A collection of agents that learn in an RL setting
"""

import numpy as np
import matplotlib.pyplot as plt

class Agent():
    """An abstract class that represents the agent interacting
    in the environment

    self.environment: An Environment object that the agent is interacting with
    self.policy: A matrix of size self.environment.num_states x self.environment.num_actions,
    where the sum of each row is one. An off-policy agent can safely ignore this.
    (TODO: Consider building off-policy, on-policy classes?)
    """
    def __init__(self, environment, policy, gamma):
        self.gamma = gamma
        self.environment = environment
        self.num_actions = environment.num_actions
        self.num_states = environment.num_states
        self.policy = policy

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
        Return the action played, and the reward.
        """
        action = self.pick_action()
        return action, self.environment.take_action(action)

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

    def estimate_value_function(self, num_steps=1000):
        """For the purpose of debugging, return a naive monte carlo estimate of V_pi
        The algorithm can be found in Sutton and Barto page 99, Monte Carlo Exploring Starts.

        num_steps is the number of steps we run each episode for.
        """
        V = np.zeros((self.num_states, 1))
        G = 0
        trajectory = self.generate_episode(num_steps=num_steps)
        for state, action, reward, first_time_seen in trajectory[::-1]:
            G = self.gamma * G + reward
            if first_time_seen:
                V[state] = G

        return V


class ControlledTDLearning(Agent):
    """The bread and butter of our work, this is the agent that
    can be augmented with controllers, namely the PID controller.

    self.learning_rate: A function that takes in the current iteration number
    and returns a learning rate
    """
    def __init__(self, environment, policy, gamma, learning_rate):
        self.learning_rate = learning_rate
        super().__init__(environment, policy, gamma)

    def estimate_value_function(self, *controllers, num_iterations=50000, V=None, label=""):
        """Computes V^pi of the inputted policy using TD learning augmented with controllers.
        Takes in Controller objects that the agent will use to control the dynamics of learning.
        If V is not None, we record the difference between the current value estimate and V during learning.
        """
        self.environment.reset()
        # V1 is the current value function, V0 is the previous value function
        # V0(x) stores the previous value of the x state when it was last changed
        V0 = np.zeros((self.num_states, 1))
        V1 = np.zeros((self.num_states, 1))

        # A vector storing the number of times we have seen a state.
        frequency = np.zeros((self.num_states, 1))

        # The history of the norms
        history = np.zeros(num_iterations)

        for k in range(num_iterations):
            current_state = self.environment.current_state
            action, reward = self.take_action()
            next_state = self.environment.current_state

            frequency[current_state] += 1

            # An estimate of the bellman update
            BR = np.zeros((self.num_states, 1))
            BR[current_state] = reward + self.gamma * V1[next_state] - V1[current_state]

            update = sum(map(lambda n: n.evaluate_controller(BR, V1, V0), controllers))
            learning_rate = self.learning_rate(frequency[current_state])

            V0[current_state] = V1[current_state]
            V1 = V1 + learning_rate * update

            if V is not None:
                history[k] = np.max(np.abs(V - V1))

        return history, V1

class ControlledQLearning(Agent):
    # TODO: Implement
    pass

class ControlledSARSA(Agent):
    # TODO: Implement
    pass