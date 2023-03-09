import numpy as np

class Controller:
    """
    An abstract controller class to control the dynamics of value iteration.

    Recall that the plant is a simple integrator with u_k as its input,
    V_{k + 1} = V_k + u_k

    A controller is a function that will output u_k given the error. We don't have access to the
    error, and so we have the controller be a function of T(V_k), V_k, V_{k - 1},
    which is all that is needed for the P, PI, PD, and PID controllers.
    """
    def evaluate_controller(self, BR, V, V_prev):
        raise NotImplementedError


class P_Controller(Controller):
    def __init__(self, Kp):
        self.Kp = Kp

    def evaluate_controller(self, BR, V, V_prev):
        return self.Kp @ BR


class I_Controller(Controller):
    def __init__(self, alpha, beta, Ki, initial_z=0):
        self.alpha = alpha
        self.beta = beta
        self.Ki = Ki
        self.z = initial_z

    def evaluate_controller(self, BR, V, V_prev):
        evaluation = self.Ki @ (self.beta * self.z + self.alpha * BR)
        self.z = self.beta * self.z + self.alpha * BR

        return evaluation


class D_Controller(Controller):
    def __init__(self, Kd):
        self.Kd = Kd

    def evaluate_controller(self, BR, V, V_prev):
        return self.Kd @ (V - V_prev)


# Variations on the D Controller
class Adagrad_Controller(Controller):
    def __init__(self, Kd):
        self.Kd = Kd
        self.G = 0

    def evaluate_controller(self, BR, V, V_prev):
        grad = V - V_prev
        self.G += np.outer(grad, grad)
        return self.Kd @ self.G @ grad


class Adam_Controller(Controller):
    def __init__(self, Kd, beta1, beta2, epsilon):
        self.Kd = Kd
        self.m = 0
        self.v = 0
        self.t = 0

        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def evaluate_controller(self, BR, V, V_prev):
        grad = V - V_prev
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.multiply(grad, grad)

        m_hat = self.m / (1 - pow(self.beta1, self.t))
        v_hat = self.b / (1 - pow(self.beta2, self.t))

        return self.Kd @ np.divide(m_hat, np.sqrt(v_hat) + self.epsilon)