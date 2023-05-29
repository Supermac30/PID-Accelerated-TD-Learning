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

class MDP:
    """
    An abstract class that represents a discounted MDP with states
    {x_1, ..., x_{num_of_states}} and actions {a_1, ..., a_{num_of_actions}}.

    self.R is the reward distribution
    self.P is the transition probability kernel of some policy
    self.gamma is the discount factor
    """
    def __init__(self, num_states, num_actions, R, P, kp, ki, kd, alpha, beta, gamma):
        self.num_states = num_states
        self.num_actions = num_actions
        self.R = R
        self.P = P
        self.gamma = gamma

        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.alpha = alpha
        self.beta = beta

        self.reset()

    def reset(self):
        """Reset the MDP to its initial state."""
        self.V = np.zeros((self.num_states, 1))
        self.Vp = np.zeros((self.num_states, 1))
        self.z = np.zeros((self.num_states, 1))

    def value_iteration(self, controllers=[], num_iterations=500, test_function=None):
        """Compute the value function via VI using the added controllers.
        If test_function is not None, the history of test_function evaluated at V1, V0, BR is returned,
        otherwise, history is full of zeroes.
        """
        # The history of the norms
        history = np.zeros(num_iterations)

        for k in range(num_iterations):
            TV = self.bellman_operator(self.V)
            BR = TV - self.V

            # Deprecated, but keeping these here to allow the novel controller experiments to work
            # TODO: Remove these
            update = 0
            if len(controllers) == 0:
                update += sum(map(lambda n: n.evaluate_controller(BR, self.V, self.Vp), controllers))

            self.z = self.beta * self.z + self.alpha * BR
            update += self.kp * BR + self.ki * self.z + self.kd * (self.V - self.Vp)
            self.Vp, self.V = self.V, self.V + update

            if test_function is not None:
                history[k] = test_function(self.V, self.Vp, BR)

        if test_function is None:
            return self.V

        return history, self.V

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


class MDP_Q:
    """
    An MDP where we use the Q function instead of the V function.
    Here P is a probability transition kernel of dimension num_states * num_states * num_actions
    """
    def __init__(self, num_states, num_actions, R, P, kp, ki, kd, alpha, beta, gamma):
        self.num_states = num_states
        self.num_actions = num_actions
        self.R = R
        self.P = P
        self.gamma = gamma

        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.alpha = alpha
        self.beta = beta

        self.reset()

    def reset(self):
        """Reset the MDP to its initial state."""
        self.Q = np.zeros((self.num_states, self.num_actions))
        self.Qp = np.zeros((self.num_states, self.num_actions))
        self.z = np.zeros((self.num_states, self.num_actions))

    def value_iteration(self, num_iterations=500, test_function=None):
        """Compute the value function via VI using the added controllers.
        If test_function is not None, the history of test_function evaluated at V1, V0, BR is returned,
        otherwise, history is full of zeroes.
        """
        # The history of the norms
        history = np.zeros(num_iterations)

        for k in range(num_iterations):
            if k % 1000 == 0:
                print(k)
            TQ = self.bellman_operator(self.Q)
            BR = TQ - self.Q

            self.z = self.beta * self.z + self.alpha * BR
            update = self.kp * BR + self.ki * self.z + self.kd * (self.Q - self.Qp)
            self.Qp, self.Q = self.Q, self.Q + update

            if test_function is not None:
                history[k] = test_function(self.Q, self.Qp, BR)

        if test_function is None:
            return self.Q

        return history, self.Q

    def bellman_operator(self, Q):
        """Computes the Bellman Operator.
        The implementation depends on whether we are using policy evaluation
        or control.
        """
        raise NotImplementedError


class Control_Q(MDP_Q):
    """
    An MDP where:
    - self.P is the transition probability kernel of dimension num_states * num_states * num_actions,
    - self.R is the expected value of the reward distribution of dimension num_states * num_actions,
    and value iteration performs control.
    """
    def bellman_operator(self, Q):
        # Create a new matrix Q' whose entries are Q'(x, a) = sum(P(x, a, y) * max(Q(y, b) for b in A) for y in S)
        """
        Qp = np.zeros((self.num_states, self.num_actions))
        for a in range(self.num_actions):
            for s in range(self.num_states):
                Qp[s, a] = self.R[s, a] + self.gamma * self.P[s, :, a] @ np.max(Q, axis=1)
        """

        # Vectorize the above
        return self.R + self.gamma * np.einsum('ijk,j->ik', self.P, np.max(Q, axis=1))

        return Qp