"""
As a dynamical system that we can apply control theory to,
value iteration is of the form:
V_{k + 1} = V_k + u_k
where u_k is the output of the controller.

To do implement this, we build various controllers using a Controller class,
and allow for multiple controllers to be passed to the MDP during the VI step.
By building this, we can compose controllers together, taking the sum of their
outputted u_k. This allows us to build a single P, I, and D controller class, and build
all possible controllers from there, as well as easily test different controllers, or mix and match.
"""

import numpy as np
import matplotlib.pyplot as plt

class MDP:
    """
    An abstract class that represents a discounted MDP with states
    {x_1, ..., x_{num_of_states}} and actions {a_1, ..., a_{num_of_actions}}.

    self.R is the reward distribution
    self.P is the transition probability kernel of some policy
    self.gamma is the discount factor
    """
    def __init__(self, num_states, num_actions, R, P, gamma):
        self.num_states = num_states
        self.num_actions = num_actions
        self.R = R
        self.P = P
        self.gamma = gamma

    def value_iteration(self, *controllers, num_iterations=500, V=None, label=""):
        """Compute the value function via VI using the added controllers.
        If V is not None, the norm of V1-V per iteration is plotted
        """
        # V1 is the current value function, V0 is the previous value function
        V0 = np.zeros((self.num_states, 1))
        V1 = np.zeros((self.num_states, 1))

        # The history of the norms
        history = []

        for _ in range(num_iterations):
            TV = self.bellman_operator(V1)
            BR = TV - V1
            V0, V1 = V1, V1 + sum(map(lambda n: n.evaluate_controller(BR, V1, V0), controllers))

            if V is not None:
                history.append(np.max(np.abs(V1 - V)))

        if V is not None:
            plt.plot(history, label=label)

        return V1

    def bellman_operator(self, V):
        """Computes the Bellman Operator.
        The implementation depends on whether we are using policy evaluation
        or control.
        """
        raise NotImplementedError


class PolicyEvaluation(MDP):
    """
    An MDP where:
    - self.P is the transition probability kernel of a policy pi of dimension num_states * num_states,
    - self.R is the expected reward under policy pi, r^pi, of dimension num_states * 1,
    and value iteration performs policy evaluation.
    """
    def bellman_operator(self, V):
        return self.R.reshape((-1, 1)) + self.gamma * self.P @ V


class Control(MDP):
    """
    An MDP where:
    - self.P is the transition probability kernel of dimension num_states * num_states * num_actions,
    - self.R is the expected value of the reward distribution of dimension num_states * num_actions,
    and value iteration performs control.
    """
    def bellman_operator(self, V):
        return np.max(self.R + self.gamma * np.einsum('ijk,j->ik', self.P, V.reshape(-1)), axis=1).reshape(-1, 1)
