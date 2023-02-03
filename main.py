"""
As a dynamical system that we can apply control theory to,
value iteration is of the form:
V_{k + 1} = V_k + u_k
where u_k is the output of the controller.

To do implement this, we build various controllers using a Controller class,
and allow for multiple controllers to be attached to the MDP.
By building this, we can compose controllers together, taking the sum of their
outputted u_k. This allows us to build a single P, I, and D controller class, and build
all possible controllers from there, as well as easily test different controllers, or mix and match.
"""

import numpy as np
import Controllers

class MDP:
    """
    An abstract class that represents a discounted MDP with states
    {x_1, ..., x_{num_of_states}} and actions {a_1, ..., a_{num_of_actions}}.

    self.R is the reward distribution
    self.P is the transition probability kernel of some policy
    self.gamma is the discount factor
    """
    def __init__(self, num_of_states, num_of_actions, R, P, gamma):
        self.num_of_states = num_of_states
        self.num_of_actions = num_of_actions
        self.R = R
        self.P = P
        self.gamma = gamma

        self.controllers = []

    def value_iteration(self, num_iterations=1000):
        """Compute the value function via VI using the added controllers.
        """
        # V1 is the current value function, V0 is the previous value function
        V0 = np.zeroes((1, self.num_of_states))
        V1 = np.zeroes((1, self.num_of_states))

        for _ in range(num_iterations):
            TV = self.bellman_operator(V1)
            V0, V1 = V1, V1 + sum(map(lambda n: n.evaluate_controller(TV, V1, V0), self.controllers))

        return V1

    def bellman_operator(self, V):
        """Computes the Bellman Operator.
        The implementation depending on whether we are using policy evaluation
        or control.
        """
        raise NotImplementedError

    def attach_controllers(self, *controllers):
        """Attaches the inputted controllers to the MDP, so that their valuation
        is added to the dynamical system during VI.
        """
        self.controllers.extend(controllers)


class PolicyEvaluation(MDP):
    """
    An MDP where:
    - self.P is the transition probability kernel of a policy pi of dimension num_states * num_states,
    - self.R is the expected reward under policy pi, r^pi, of dimension num_states * 1,
    and value iteration performs policy evaluation.
    """
    def bellman_operator(self, V):
        return self.R + self.gamma * self.P @ V


class Control(MDP):
    """
    An MDP where:
    - self.P is the transition probability kernel of dimension num_states * num_states * num_actions,
    - self.R is the expected value of the reward distribution of dimension num_states * num_actions,
    and value iteration performs control.
    """
    def bellman_operator(self, V):
        return np.max(self.R + self.gamma * (self.P @ V).reshape(self.R.shape), axis=1)
