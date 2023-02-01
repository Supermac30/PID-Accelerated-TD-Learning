import numpy as np

class MDP:
    """
    Represents a discounted MDP ({x_1, ..., x_n}, {a_1, ..., a_m}, R, P, gamma).
    """
    def __init__(self, num_of_states, num_of_actions, R, P, gamma):
        self.num_of_states = num_of_states
        self.num_of_actions = num_of_actions
        self.R = R
        self.P = P
        self.gamma = gamma

        self.controllers = []

    def attach_controllers(self, *controllers):
        """Attaches the inputted controllers to the MDP, so that their valuation
        is added to the dynamical system during VI.
        """
        self.controllers.extend(controllers)

    def bellman_operator(self, V):
        """Computes T^pi(V), where T^pi is the Bellman operator of the policy pi
        self.P is the transition probability kernel of."""
        return self.R + self.gamma * self.P @ V

    def value_iteration(self, num_iterations=1000):
        """Compute the value function via VI using the added controllers.

        The plant is a simple integrator with u_k as its input,
        V_{k + 1} = V_k + u_k
        where u_k is the sum of all attached controllers.
        """
        # V1 is the current value function, V0 is the previous value function
        V0 = np.zeroes((self.num_of_states, 1))
        V1 = np.zeroes((self.num_of_states, 1))

        for _ in range(num_iterations):
            TV = self.bellman_operator(V1)
            V0, V1 = V1, V1 + sum(map(lambda n: n.evaluate_controller(TV, V1, V0), self.controllers))

        return V1



class Controller:
    """
    A generic controller class to control the dynamics of value iteration.

    Recall that the plant is a simple integrator with u_k as its input,
    V_{k + 1} = V_k + u_k

    A controller is a function that will output u_k given the error. We don't have access to the
    error, and so we have the controller be a function of T(V_k), V_k, V_{k - 1},
    which is all that is needed for the P, PI, PD, and PID controllers.

    By building this, we can compose controllers together, by taking the sum of their
    outputted u_k. This allows us to build a single P, I, and D controller class, and build
    all possible controllers from there.

    This allows us to easily test different types controllers and mix and match
    """
    def evaluate_controller(self, TV_k, V_k, V_k_minus_1):
        raise NotImplementedError


class P_Controller(Controller):
    def __init__(self, Kp):
        self.Kp = Kp

    def evaluate_controller(self, TV_k, V_k, V_k_minus_1):
        BR = TV_k - V_k
        return self.Kp @ BR


class I_Controller(Controller):
    def __init__(self, alpha, beta, Ki, initial_z = 0):
        self.alpha = alpha
        self.beta = beta
        self.Ki = Ki
        self.z = initial_z

    def evaluate_controller(self, TV_k, V_k, V_k_minus_1):
        BR = TV_k - V_k
        evaluation = self.Ki * (self.beta @ self.z + self.alpha @ BR)
        self.z = self.beta + self.alpha @ BR

        return evaluation


class D_Controller(Controller):
    def __init__(self, Kd):
        self.Kd = Kd

    def evaluate_controller(self, TV_k, V_k, V_k_minus_1):
        return self.Kd @ (V_k - V_k_minus_1)