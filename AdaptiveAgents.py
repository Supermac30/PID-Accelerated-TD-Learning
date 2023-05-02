import numpy as np
from Agents import Agent
from collections import defaultdict
import logging

class AbstractAdaptiveAgent(Agent):
    def __init__(self, gain_updater, learning_rates, meta_lr, environment, policy, gamma, update_frequency=1):
        super().__init__(environment, policy, gamma)

        self.meta_lr = meta_lr
        self.learning_rate = learning_rates[0]
        self.update_I_rate = learning_rates[1]
        self.update_D_rate = learning_rates[2]
        self.update_frequency = update_frequency

        self.gain_updater = gain_updater
        self.gain_updater.set_agent(self)

        self.kp, self.ki, self.kd = 1, 0, 0
        self.alpha, self.beta = 0.95, 0.05
        self.lr = 0

        self.replay_buffer = defaultdict(list)

        self.V, self.Vp, self.z = np.zeros((self.num_states, 1)), np.zeros((self.num_states, 1)), np.zeros((self.num_states, 1))
        self.previous_V, self.previous_Vp, self.previous_z = np.zeros((self.num_states, 1)), np.zeros((self.num_states, 1)), np.zeros((self.num_states, 1))
        self.previous_previous_V = np.zeros((self.num_states, 1))

        self.previous_state, self.current_state, self.next_state = 0, 0, 0
        self.previous_reward, self.reward = 0, 0

    def estimate_value_function(self, num_iterations=1000, test_function=None, initial_V=None):
        self.environment.reset()
        # V is the current value function, Vp is the previous value function
        # Vp stores the previous value of the x state when it was last changed
        if initial_V is not None:
            self.V = initial_V.copy()
            self.Vp = initial_V.copy()

        self.frequencies = np.zeros((self.num_states))

        history = np.zeros((num_iterations))
        gain_history = np.zeros((num_iterations, 5))

        for k in range(num_iterations):
            self.previous_state, self.current_state, self.previous_reward = self.current_state, self.next_state, self.reward
            _, self.next_state, self.reward = self.take_action()

            self.replay_buffer[self.current_state].append(
                (self.previous_reward, self.reward, self.previous_state, self.next_state)
            )

            self.frequencies[self.current_state] += 1
            self.update_value()
            if (k + 1) % self.update_frequency == 0:
                self.gain_updater.update_gains()

            # Keep a record
            gain_history[k][0] = self.kp
            gain_history[k][1] = self.ki
            gain_history[k][2] = self.kd
            gain_history[k][3] = self.alpha
            gain_history[k][4] = self.beta

            if test_function is not None:
                history[k] = test_function(self.V, self.Vp, self.BR)

        if test_function is not None:
            return self.V, gain_history, history

        return self.V, gain_history

    def update_value(self):
        """Update V, Vp, z and the previous versions"""
        raise NotImplementedError

    def BR(self):
        """Return the bellman residual"""
        raise NotImplementedError


class AdaptiveSamplerAgent(AbstractAdaptiveAgent):
    def __init__(self, gain_updater, learning_rates, meta_lr, environment, policy, gamma, update_frequency=1):
        super().__init__(gain_updater, learning_rates, meta_lr, environment, policy, gamma, update_frequency)

    def update_value(self):
        lr = self.learning_rate(self.frequencies[self.current_state])
        update_D_rate = self.update_D_rate(self.frequencies[self.current_state])
        update_I_rate = self.update_I_rate(self.frequencies[self.current_state])

        self.previous_lr, self.lr = self.lr, lr

        BR = self.BR()
        new_V = self.V + self.kp * BR + self.kd * (self.V - self.Vp) + self.ki * (self.beta * self.z + self.alpha * BR)
        new_z = self.beta * self.z + self.alpha * BR
        new_Vp = self.V

        state = self.current_state

        self.previous_previous_V[state], self.previous_V[state], self.V[state] = self.previous_V[state], self.V[state], (1 - lr) * self.V[state] + lr * new_V[state]
        self.previous_z[state], self.z[state] = self.z[state], (1 - update_I_rate) * self.z[state] + update_I_rate * new_z[state]
        self.previous_Vp[state], self.Vp[state] = self.Vp[state], (1 - update_D_rate) * self.Vp[state] + update_D_rate * new_Vp[state]

    def BR(self):
        """Return the empirical bellman"""
        return self.reward + self.gamma * self.V[self.next_state] - self.V[self.current_state]


class AdaptivePlannerAgent(AbstractAdaptiveAgent):
    def __init__(self, R, transition, gain_updater, learning_rates, meta_lr, environment, policy, gamma, update_frequency=1):
        self.transition = transition
        self.R = R
        super().__init__(gain_updater, learning_rates, meta_lr, environment, policy, gamma, update_frequency)

    def update_value(self):
        BR = self.BR()
        self.previous_V, self.V = self.V, self.V + self.kp * BR + self.kd * (self.V - self.Vp) + self.ki * (self.beta * self.z + self.alpha * BR)
        self.previous_z, self.z = self.z, self.beta * self.z + self.alpha * BR
        self.previous_Vp, self.Vp = self.Vp, self.V

    def BR(self):
        """Return the bellman"""
        return self.R + self.gamma * self.transition * self.V - self.V






class AbstractGainUpdater():
    def __init__(self):
        self.agent = None
        self.epsilon = 1e-20

    def set_agent(self, agent):
        self.agent = agent
        self.num_states = self.agent.num_states
        self.gamma = self.agent.gamma
        self.meta_lr = self.agent.meta_lr

    def update_gains(self):
        raise NotImplementedError


class SoftGainUpdater():
    def __init__(self):
        self.fp = np.zeros((self.num_states, 1))
        self.fd = np.zeros((self.num_states, 1))
        self.fi = np.zeros((self.num_states, 1))

        super().__init__()

    def update_gains(self):
        reward = self.agent.previous_reward, self.agent.reward
        next_state, current_state = self.agent.next_state, self.agent.current_state, self.agent.previous_state
        V, Vp, z = self.agent.V, self.agent.Vp, self.agent.z
        alpha, beta = self.agent.alpha, self.agent.beta
        gamma, lr = self.gamma, self.agent.lr, self.agent.previous_lr

        BR = reward + self.gamma * V[next_state] - V[current_state]
        self.agent.kp -= self.meta_lr * BR * self.fp[current_state]
        self.agent.kd -= self.meta_lr * BR * self.fd[current_state]
        self.agent.ki -= self.meta_lr * BR * self.fi[current_state]

        self.fp[current_state] += lr * BR
        self.fd[current_state] += lr * (V[current_state] - Vp[current_state])
        self.fi[current_state] += lr * (beta * z + alpha * BR)


class EmpiricalCostUpdater(AbstractGainUpdater):
    """We need agent.update_frequency = 2 for this to mathematically make sense"""
    def update_gains(self):
        previous_reward, reward = self.agent.previous_reward, self.agent.reward
        next_state, current_state, previous_state = self.agent.next_state, self.agent.current_state, self.agent.previous_state
        next_V, V, previous_V = self.agent.V, self.agent.previous_V, self.agent.previous_previous_V
        gamma, lr, previous_lr = self.gamma, self.agent.lr, self.agent.previous_lr

        def approx_diff(a, b):
            if current_state == next_state:
                return (gamma * lr - previous_lr) * a
            return gamma * lr * a - previous_lr * b

        # Find the derivative of BR with respect to kp
        next_BR = reward + gamma * next_V[next_state] - next_V[current_state]
        current_BR = reward + gamma * V[next_state] - V[current_state]
        previous_BR = previous_reward + gamma * previous_V[current_state] - previous_V[previous_state]

        BR_kp_grad = approx_diff(current_BR, previous_BR)

        # Find the derivative of BR with respect to kd
        Vp, previous_Vp = self.agent.Vp, self.agent.previous_Vp
        current_difference = V[current_state] - Vp[current_state]
        previous_difference = previous_V[previous_state] - previous_Vp[previous_state]

        BR_kd_grad = approx_diff(current_difference, previous_difference)

        # Find the derivative of BR with respect to ki, alpha, and beta
        ki, alpha, beta = self.agent.ki, self.agent.alpha, self.agent.beta
        z, previous_z = self.agent.z, self.agent.previous_z
        current_z_update = beta * z[current_state] - alpha * current_BR
        previous_z_update = beta * previous_z[previous_state] - alpha * previous_BR

        BR_ki_grad = approx_diff(current_z_update, previous_z_update)
        BR_alpha_grad = approx_diff(current_BR * ki, previous_BR * ki)
        BR_beta_grad = approx_diff(ki * z[current_state], ki * previous_z[previous_state])

        # Perform the updates
        normalizer = 1  #self.epsilon + (current_BR ** 2)
        update = lambda n: (next_BR * n) / normalizer

        logging.debug(
            f"""
            {self.agent.kp=}
            {self.agent.kd=}
            {self.agent.ki=}
            {self.agent.alpha=}
            {self.agent.beta=}

            {BR_kp_grad=}
            {BR_ki_grad=}
            {BR_kd_grad=}
            {BR_alpha_grad=}
            {BR_beta_grad=}
            """
        )

        meta_lr = self.meta_lr
        self.agent.kp -= meta_lr * update(BR_kp_grad)
        self.agent.kd -= meta_lr * update(BR_kd_grad)
        self.agent.ki -= meta_lr * update(BR_ki_grad)
        self.agent.alpha -= meta_lr * update(BR_alpha_grad)
        self.agent.beta -= meta_lr * update(BR_beta_grad)


class AbstractOriginalCostUpdater(AbstractGainUpdater):
    def __init__(self, scale_by_lr):
        super().__init__()
        self.scale_by_lr = scale_by_lr

    def update_gains(self):
        partial_br_kp, partial_br_ki, partial_br_kd, partial_br_beta, BR_current, BR_previous = self.get_gradient_terms()

        normalizer = self.epsilon + np.linalg.norm(BR_previous, 2) ** 2

        kp_grad = (BR_current.T @ partial_br_kp) / normalizer
        ki_grad = (BR_current.T @ partial_br_ki) / normalizer
        kd_grad = (BR_current.T @ partial_br_kd) / normalizer
        beta_grad = (BR_current.T @ partial_br_beta) / normalizer
        alpha_grad = self.agent.ki * kp_grad

        # Renormalize alpha and beta
        # self.alpha, self.beta = self.alpha / (self.alpha + self.beta), self.beta / (self.alpha + self.beta)

        if self.scale_by_lr:
            lr = self.agent.lr * self.meta_lr
        else:
            lr = self.meta_lr

        self.agent.alpha -= lr * alpha_grad[0][0]
        self.agent.beta -= lr * beta_grad[0][0]
        self.agent.kp -= lr * kp_grad[0][0]
        self.agent.ki -= lr * ki_grad[0][0]
        self.agent.kd -= lr * kd_grad[0][0]

    def get_gradient_terms():
        raise NotImplementedError


class ExactUpdater(AbstractOriginalCostUpdater):
    def __init__(self, transition, reward, scale):
        super().__init__(scale)
        self.transition = transition
        self.reward = reward.reshape(-1, 1)

    def get_gradient_terms(self):
        V, Vp, Vpp = self.agent.V, self.agent.previous_V, self.agent.previous_Vp
        z, zp = self.agent.z, self.agent.previous_z

        BR_current = self.gamma * self.transition @ V + self.reward - V
        BR_previous = self.gamma * self.transition @ Vp + self.reward - Vp

        partial_br_kp = (self.gamma * self.transition @ BR_previous) - BR_previous
        partial_br_kd = (self.gamma * self.transition @ (Vp - Vpp)) - (Vp - Vpp)
        partial_br_ki = (self.gamma * self.transition @ z) - z
        partial_br_beta = (self.gamma * self.transition @ zp) - zp

        return partial_br_kp, partial_br_ki, partial_br_kd, partial_br_beta, BR_current, BR_previous


class SamplerUpdater(AbstractOriginalCostUpdater):
    def __init__(self, sample_size, scale):
        super().__init__(scale)
        self.sample_size = sample_size

    def get_gradient_terms(self):
        """Find the gradient terms to update the controller gains

        Return estimates for frac{partial BR(V_k)}{partial kappa_p},
                             frac{partial BR(V_k)}{partial kappa_i},
                             frac{partial BR(V_k)}{partial kappa_d},
                             frac{partial BR(V_k)}{partial beta},
                             BR(V_k),
                             BR(V_{k - 1})
        in that order.
        """

        # \frac{\partial \text{BR}(V_k)}{\partial \kappa_p}
        partial_br_kp = np.zeros((self.num_states, 1))

        # \frac{\partial \text{BR}(V_k)}{\partial \kappa_d}
        partial_br_ki = np.zeros((self.num_states, 1))

        # \frac{\partial \text{BR}(V_k)}{\partial \kappa_d}
        partial_br_kd = np.zeros((self.num_states, 1))

        # \frac{\partial \text{BR}(V_k)}{\partial \kappa_d}
        partial_br_beta = np.zeros((self.num_states, 1))

        # BR(V_k)
        BR_current = np.zeros((self.num_states, 1))

        # BR(V_{k - 1})
        BR_previous = np.zeros((self.num_states, 1))

        replay_buffer = self.agent.replay_buffer
        V, Vp, Vpp = self.agent.V, self.agent.previous_V, self.agent.previous_Vp
        z, zp = self.agent.z, self.agent.previous_z

        for state in range(self.num_states):
            if len(replay_buffer[state]) == 0:
                continue

            for _ in range(self.sample_size):
                previous_reward, reward, previous_state, next_state = \
                    replay_buffer[state][np.random.randint(0, len(replay_buffer[state]))]

                partial_br_kp[state] += Vp[previous_state] - 2 * self.gamma * Vp[state] \
                        + (self.gamma ** 2) * Vp[next_state] - previous_reward + self.gamma * reward

            for _ in range(self.sample_size):
                previous_reward, reward, previous_state, next_state = \
                    replay_buffer[state][np.random.randint(0, len(replay_buffer[state]))]

                partial_br_ki[state] += self.gamma * z[next_state] - z[state]

            for _ in range(self.sample_size):
                previous_reward, reward, previous_state, next_state = \
                    replay_buffer[state][np.random.randint(0, len(replay_buffer[state]))]

                partial_br_kd[state] += Vpp[previous_state] - Vp[previous_state] \
                            - self.gamma * Vpp[state] + self.gamma * Vp[state]

            for _ in range(self.sample_size):
                previous_reward, reward, previous_state, next_state = \
                    replay_buffer[state][np.random.randint(0, len(replay_buffer[state]))]

                partial_br_beta[state] += self.gamma * zp[next_state] - zp[state]

            for _ in range(self.sample_size):
                previous_reward, reward, previous_state, next_state = \
                    replay_buffer[state][np.random.randint(0, len(replay_buffer[state]))]
                BR_current[state] += reward + self.gamma * V[next_state] - V[state]

            for _ in range(self.sample_size):
                previous_reward, reward, previous_state, next_state = \
                    replay_buffer[state][np.random.randint(0, len(replay_buffer[state]))]
                BR_previous[state] += reward + self.gamma * Vp[next_state] - Vp[state]

            partial_br_kp[state] /= self.sample_size
            partial_br_ki[state] /= self.sample_size
            partial_br_kd[state] /= self.sample_size
            partial_br_beta[state] /= self.sample_size
            BR_current[state] /= self.sample_size
            BR_previous[state] /= self.sample_size

        partial_br_beta *= self.agent.ki

        return partial_br_kp, partial_br_ki, partial_br_kd, partial_br_beta, BR_current, BR_previous


class HalfExactUpdater(AbstractOriginalCostUpdater):
    """Test having knowledge of the transitions to update the gradients, but not the Bellman residual"""
    def __init__(self, transition, reward, sample_size, scale):
        super().__init__(scale)
        self.transition = transition
        self.reward = reward.reshape(-1, 1)
        self.sample_size = sample_size

    def get_gradient_terms(self):
        V, Vp, Vpp = self.agent.V, self.agent.previous_V, self.agent.previous_Vp
        z, zp = self.agent.z, self.agent.previous_z
        replay_buffer = self.agent.replay_buffer

        BR_current = self.gamma * self.transition @ V + self.reward - V
        BR_previous = self.gamma * self.transition @ Vp + self.reward - Vp

        partial_br_kp = (self.gamma * self.transition @ BR_previous) - BR_previous
        partial_br_kd = (self.gamma * self.transition @ (Vp - Vpp)) - (Vp - Vpp)
        partial_br_ki = (self.gamma * self.transition @ z) - z
        partial_br_beta = (self.gamma * self.transition @ zp) - zp

        for state in range(self.num_state):
            for _ in range(self.sample_size):
                previous_reward, reward, previous_state, next_state = \
                    replay_buffer[state][np.random.randint(0, len(replay_buffer[state]))]
                BR_current[state] += reward + self.gamma * V[next_state] - V[state]

            for _ in range(self.sample_size):
                previous_reward, reward, previous_state, next_state = \
                    replay_buffer[state][np.random.randint(0, len(replay_buffer[state]))]
                BR_previous[state] += reward + self.gamma * Vp[next_state] - Vp[state]

            BR_current[state] /= self.sample_size
            BR_previous[state] /= self.sample_size

        return partial_br_kp, partial_br_ki, partial_br_kd, partial_br_beta, BR_current, BR_previous


"""
class OldAdaptiveAgent(Agent):
    def __init__(self, learning_rates, meta_lr, environment, policy, gamma, sample_size, transition=None, rewards=None, planning=False):
        \"""
        learning_rates: A triple of three learning rates, one for each component PID
        meta_lr: The meta learning rate
        transition: The transition probability matrix. If this is None, we will approximate gradient terms instead.
        \"""
        super().__init__(learning_rates, meta_lr, environment, policy, gamma)

        self.meta_lr = meta_lr
        self.learning_rate = learning_rates[0]
        self.update_I_rate = learning_rates[1]
        self.update_D_rate = learning_rates[2]
        self.transition = transition

        if self.transition is not None:
            self.policy_evaluator = PolicyEvaluation(
                self.environment.num_states,
                self.environment.num_actions,
                rewards,
                self.transition,
                self.gamma
            )

        self.epsilon = 1e-20  # For numerical stability during normalization
        self.sample_size = sample_size
        self.replay_buffer = [[] for _ in range(self.num_states)]

        self.kp, self.ki, self.kd = 1, 0, 0
        self.alpha, self.beta = 0.95, 0.05

        self.planning = planning

        # self.update_interval = 10  # Adapt gains every self.update_interval steps

    def update_gains(self):
        \"""Find the gradient terms to update the controller gains
        Vp: The previous values of each component before being put in V
        Vpp: The previous values of each component before being put in Vp
        \"""
        if self.transition is None:
            partial_br_kp, partial_br_ki, partial_br_kd, partial_br_beta, BR_current, BR_previous = self.approximate_gradient_terms(V, Vp, Vpp, z, zp)
        else:
            partial_br_kp, partial_br_ki, partial_br_kd, partial_br_beta, BR_current, BR_previous = self.find_exact_gradient_terms(V, Vp, Vpp, z, zp)

        normalizer = self.epsilon + np.linalg.norm(BR_previous, 2) ** 2

        kp_grad = (BR_current.T @ partial_br_kp) / normalizer
        ki_grad = (BR_current.T @ partial_br_ki) / normalizer
        kd_grad = (BR_current.T @ partial_br_kd) / normalizer
        beta_grad = (BR_current.T @ partial_br_beta) / normalizer
        alpha_grad = self.ki * kp_grad

        # Renormalize alpha and beta
        # self.alpha, self.beta = self.alpha / (self.alpha + self.beta), self.beta / (self.alpha + self.beta)

        self.alpha -= lr * self.meta_lr * alpha_grad
        self.beta -= lr * self.meta_lr * beta_grad
        self.kp -= lr * self.meta_lr * kp_grad
        self.ki -= lr * self.meta_lr * ki_grad
        self.kd -= lr * self.meta_lr * kd_grad


    def find_exact_gradient_terms(self, V, Vp, Vpp, z, zp):
        \"""Find the gradient terms to update the controller gains
        when we have access to the transition probabilities.
        This is used for testing.

        V: The current value function
        Vp: The previous values of each component before being put in V
        Vpp: The previous values of each component before being put in Vp

        Return estimates for frac{partial BR(V_k)}{partial kappa_p},
                             frac{partial BR(V_k)}{partial kappa_i},
                             frac{partial BR(V_k)}{partial kappa_d},
                             frac{partial BR(V_k)}{partial beta},
                             BR(V_k),
                             BR(V_{k - 1})
        in that order.
        \"""
        BR_current = self.policy_evaluator.bellman_operator(V) - V
        BR_previous = self.policy_evaluator.bellman_operator(Vp) - Vp

        partial_br_kp = (self.gamma * self.transition @ BR_previous) - BR_previous
        partial_br_kd = (self.gamma * self.transition @ (Vp - Vpp)) - (Vp - Vpp)
        partial_br_ki = (self.gamma * self.transition @ z) - z
        partial_br_beta = (self.gamma * self.transition @ zp) - zp

        return partial_br_kp, partial_br_ki, partial_br_kd, partial_br_beta, BR_current, BR_previous


    def approximate_gradient_terms(self, V, Vp, Vpp, z, zp):
        \"""Find the gradient terms to update the controller gains
        V: The current value function
        Vp: The previous values of each component before being put in V
        Vpp: The previous values of each component before being put in Vp
        z: The current value of z
        zp: The previous value of z

        Return estimates for frac{partial BR(V_k)}{partial kappa_p},
                             frac{partial BR(V_k)}{partial kappa_i},
                             frac{partial BR(V_k)}{partial kappa_d},
                             frac{partial BR(V_k)}{partial beta},
                             BR(V_k),
                             BR(V_{k - 1})
        in that order.
        \"""
        # \frac{\partial \text{BR}(V_k)}{\partial \kappa_p}
        partial_br_kp = np.zeros((self.num_states, 1))

        # \frac{\partial \text{BR}(V_k)}{\partial \kappa_d}
        partial_br_ki = np.zeros((self.num_states, 1))

        # \frac{\partial \text{BR}(V_k)}{\partial \kappa_d}
        partial_br_kd = np.zeros((self.num_states, 1))

        # \frac{\partial \text{BR}(V_k)}{\partial \kappa_d}
        partial_br_beta = np.zeros((self.num_states, 1))

        # BR(V_k)
        BR_current = np.zeros((self.num_states, 1))

        # BR(V_{k - 1})
        BR_previous = np.zeros((self.num_states, 1))

        for state in range(self.num_states):
            if len(self.replay_buffer[state]) == 0:
                continue

            for _ in range(self.sample_size):
                previous_reward, reward, previous_state, next_state = \
                    self.replay_buffer[state][np.random.randint(0, len(self.replay_buffer[state]))]

                partial_br_kp[state] += Vp[previous_state] - 2 * self.gamma * Vp[state] \
                        + (self.gamma ** 2) * Vp[next_state] - previous_reward + self.gamma * reward

            for _ in range(self.sample_size):
                previous_reward, reward, previous_state, next_state = \
                    self.replay_buffer[state][np.random.randint(0, len(self.replay_buffer[state]))]

                partial_br_ki[state] += self.gamma * z[next_state] - z[state]

            for _ in range(self.sample_size):
                previous_reward, reward, previous_state, next_state = \
                    self.replay_buffer[state][np.random.randint(0, len(self.replay_buffer[state]))]
                partial_br_kd[state] += Vpp[previous_state] - Vp[previous_state] \
                            - self.gamma * Vpp[state] + self.gamma * Vp[state]

            for _ in range(self.sample_size):
                previous_reward, reward, previous_state, next_state = \
                    self.replay_buffer[state][np.random.randint(0, len(self.replay_buffer[state]))]

                partial_br_beta[state] += self.gamma * zp[next_state] - zp[state]

            for _ in range(self.sample_size):
                previous_reward, reward, previous_state, next_state = \
                    self.replay_buffer[state][np.random.randint(0, len(self.replay_buffer[state]))]
                BR_current[state] += reward + self.gamma * V[next_state] - V[state]

            for _ in range(self.sample_size):
                previous_reward, reward, previous_state, next_state = \
                    self.replay_buffer[state][np.random.randint(0, len(self.replay_buffer[state]))]
                BR_previous[state] += reward + self.gamma * Vp[next_state] - Vp[state]

            partial_br_kp[state] /= self.sample_size
            partial_br_ki[state] /= self.sample_size
            partial_br_kd[state] /= self.sample_size
            partial_br_beta[state] /= self.sample_size
            BR_current[state] /= self.sample_size
            BR_previous[state] /= self.sample_size

        partial_br_beta *= self.ki

        return partial_br_kp, partial_br_ki, partial_br_kd, partial_br_beta, BR_current, BR_previous

"""