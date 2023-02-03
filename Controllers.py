class Controller:
    """
    An abstract controller class to control the dynamics of value iteration.

    Recall that the plant is a simple integrator with u_k as its input,
    V_{k + 1} = V_k + u_k

    A controller is a function that will output u_k given the error. We don't have access to the
    error, and so we have the controller be a function of T(V_k), V_k, V_{k - 1},
    which is all that is needed for the P, PI, PD, and PID controllers.
    """
    def evaluate_controller(self, TV, V, V_prev):
        raise NotImplementedError


class P_Controller(Controller):
    def __init__(self, Kp):
        self.Kp = Kp

    def evaluate_controller(self, TV, V, V_prev):
        BR = TV - V
        return self.Kp @ BR


class I_Controller(Controller):
    def __init__(self, alpha, beta, Ki, initial_z = 0):
        self.alpha = alpha
        self.beta = beta
        self.Ki = Ki
        self.z = initial_z

    def evaluate_controller(self, TV, V, V_prev):
        BR = TV - V
        evaluation = self.Ki @ (self.beta @ self.z + self.alpha @ BR)
        self.z = self.beta + self.alpha @ BR

        return evaluation


class D_Controller(Controller):
    def __init__(self, Kd):
        self.Kd = Kd

    def evaluate_controller(self, TV, V, V_prev):
        return self.Kd @ (V - V_prev)
