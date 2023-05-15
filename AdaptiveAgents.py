import numpy as np
from Agents import Agent
from collections import defaultdict
import logging
import matplotlib.pyplot as plt

# TODO: Rewrite this class to use the agent class instead of rewriting the updates here.

class AbstractAdaptiveAgent(Agent):
    def __init__(self, gain_updater, learning_rates, meta_lr, environment, policy, gamma, update_frequency=1, kp=1, kd=0, ki=0, alpha=0.05, beta=0.95):
        super().__init__(environment, policy, gamma)

        self.meta_lr = meta_lr
        self.learning_rate = learning_rates[0]
        self.update_I_rate = learning_rates[1]
        self.update_D_rate = learning_rates[2]
        self.update_frequency = update_frequency

        self.gain_updater = gain_updater

        self.original_kp, self.original_ki, self.original_kd = kp, ki, kd
        self.original_alpha, self.original_beta = alpha, beta

        self.reset()

    def reset(self):
        self.environment.reset()
        self.kp, self.ki, self.kd = self.original_kp, self.original_ki, self.original_kd
        self.alpha, self.beta = self.original_alpha, self.original_beta
        self.lr, self.previous_lr, self.update_D_rate_value, self.previous_update_D_rate_value, self.update_I_rate_value, self.previous_update_I_rate_value = 0, 0, 0, 0, 0, 0

        self.replay_buffer = defaultdict(list)
        self.frequencies = np.zeros((self.num_states))

        self.V, self.Vp, self.z, self.previous_V, self.previous_Vp, self.previous_z, self.previous_previous_V \
            = (np.zeros((self.num_states, 1), dtype=np.longdouble) for _ in range(7))

        self.previous_previous_state, self.previous_state, self.current_state, self.next_state = 0, 0, 0, 0
        self.previous_reward, self.reward = 0, 0

        self.gain_updater.set_agent(self)

    def estimate_value_function(self, num_iterations=1000, test_function=None, initial_V=None, stop_if_diverging=True, follow_trajectory=True):
        self.reset()
        # V is the current value function, Vp is the previous value function
        # Vp stores the previous value of the x state when it was last changed
        if initial_V is not None:
            self.V = initial_V.copy()
            self.Vp = initial_V.copy()

        history = np.zeros((num_iterations))
        gain_history = np.zeros((num_iterations, 5))

        for k in range(num_iterations):
            self.previous_previous_state, self.previous_state, self.previous_reward = self.previous_state, self.current_state, self.reward
            self.current_state, _, self.next_state, self.reward = self.take_action(follow_trajectory)

            self.replay_buffer[self.current_state].append(
                (self.previous_reward, self.reward, self.previous_state, self.next_state)
            )

            self.frequencies[self.current_state] += 1
            if (k + 1) % self.update_frequency == 0:
                self.gain_updater.calculate_updated_values()
                self.gain_updater.update_gains()
                self.update_value()
            else:
                self.gain_updater.intermediate_update()
                self.update_value()


            # Keep a record
            try:
                gain_history[k][0] = self.kp
                gain_history[k][1] = self.ki
                gain_history[k][2] = self.kd
                gain_history[k][3] = self.alpha
                gain_history[k][4] = self.beta
            except:
                pass

            if test_function is not None:
                history[k] = test_function(self.V, self.Vp, self.BR)
                if stop_if_diverging and history[k] > 1.5 * history[0]:
                    history[k:] = float("inf")
                    break

        logging.info(f"Final gains are: kp: {self.kp}, ki: {self.ki}, kd: {self.kd}, alpha: {self.alpha}, beta: {self.beta}")
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
    def update_value(self):
        lr = self.learning_rate(self.frequencies[self.current_state])
        update_D_rate = self.update_D_rate(self.frequencies[self.current_state])
        update_I_rate = self.update_I_rate(self.frequencies[self.current_state])

        self.previous_lr, self.lr = self.lr, lr
        self.previous_update_D_rate_value, self.update_D_rate_value = self.update_D_rate_value, update_D_rate
        self.previous_update_I_rate_value, self.update_I_rate_value = self.update_I_rate_value, update_I_rate

        state = self.current_state

        self.previous_previous_V, self.previous_V = self.previous_V.copy(), self.V.copy()
        self.previous_Vp, self.previous_z = self.Vp.copy(), self.z.copy()

        #Update the value function using the floats kp, ki, kd
        BR = self.BR()
        new_V = self.V[state][0] + self.kp * BR + self.kd * (self.V[state][0] - self.Vp[state][0]) + self.ki * (self.beta * self.z[state][0] + self.alpha * BR)
        new_z = self.beta * self.z[state][0] + self.alpha * BR
        new_Vp = self.V[state][0]

        self.previous_previous_V[state], self.previous_V[state], self.V[state] = self.previous_V[state][0], self.V[state][0], (1 - lr) * self.V[state][0] + lr * new_V
        self.previous_z[state], self.z[state] = self.z[state][0], (1 - update_I_rate) * self.z[state][0] + update_I_rate * new_z
        self.previous_Vp[state], self.Vp[state] = self.Vp[state][0], (1 - update_D_rate) * self.Vp[state][0] + update_D_rate * new_Vp


        #self.z[current_state] = (1 - update_I_rate) * self.z[current_state][0] + update_I_rate * (self.beta * self.z[current_state][0] + self.alpha * BR)
        #update = self.kp * BR + self.ki * self.z[current_state] + self.kd * (self.V[current_state] - self.Vp[current_state])
        #self.Vp[current_state] = (1 - update_D_rate) * self.Vp[current_state][0] + update_D_rate * self.V[current_state][0]
        #self.V[current_state] = self.V[current_state][0] + lr * update

    def BR(self):
        """Return the empirical bellman residual"""
        return self.reward + self.gamma * self.V[self.next_state][0] - self.V[self.current_state][0]


class DiagonalAdaptiveSamplerAgent(AbstractAdaptiveAgent):
    def __init__(self, gain_updater, learning_rates, meta_lr, environment, policy, gamma, update_frequency=1, kp=1, kd=0, ki=0, alpha=0.05, beta=0.95):
        super().__init__(gain_updater, learning_rates, meta_lr, environment, policy, gamma, update_frequency, kp, kd, ki, alpha, beta)
        
        gain_updater.set_agent(self)
    
    def reset(self):
        self.environment.reset()
        self.kp = np.full((self.num_states), self.original_kp, dtype=np.longdouble)
        self.ki = np.full((self.num_states), self.original_ki, dtype=np.longdouble)
        self.kd = np.full((self.num_states), self.original_kd, dtype=np.longdouble)

        self.alpha, self.beta = self.original_alpha, self.original_beta
        self.lr, self.previous_lr, self.update_D_rate_value, self.previous_update_D_rate_value, self.update_I_rate_value, self.previous_update_I_rate_value = 0, 0, 0, 0, 0, 0

        self.replay_buffer = defaultdict(list)
        self.frequencies = np.zeros((self.num_states))

        self.V, self.Vp, self.z, self.previous_V, self.previous_Vp, self.previous_z, self.previous_previous_V \
            = (np.zeros((self.num_states, 1), dtype=np.longdouble) for _ in range(7))

        self.previous_previous_state, self.previous_state, self.current_state, self.next_state = 0, 0, 0, 0
        self.previous_reward, self.reward = 0, 0

        self.gain_updater.set_agent(self)

    def update_value(self):
        lr = self.learning_rate(self.frequencies[self.current_state])
        update_D_rate = self.update_D_rate(self.frequencies[self.current_state])
        update_I_rate = self.update_I_rate(self.frequencies[self.current_state])

        self.previous_lr, self.lr = self.lr, lr

        state = self.current_state

        BR = self.BR()
        new_V = self.V[state][0] + self.kp[state] * BR + self.kd[state] * (self.V[state][0] - self.Vp[state][0]) + self.ki[state] * (self.beta * self.z[state][0] + self.alpha * BR)
        new_z = self.beta * self.z[state][0] + self.alpha * BR
        new_Vp = self.V[state][0]

        self.previous_previous_V[state], self.previous_V[state], self.V[state] = self.previous_V[state][0], self.V[state][0], (1 - lr) * self.V[state][0] + lr * new_V
        self.previous_z[state], self.z[state] = self.z[state][0], (1 - update_I_rate) * self.z[state][0] + update_I_rate * new_z
        self.previous_Vp[state], self.Vp[state] = self.Vp[state][0], (1 - update_D_rate) * self.Vp[state][0] + update_D_rate * new_Vp

    def BR(self):
        """Return the empirical bellman residual"""
        return self.reward + self.gamma * self.V[self.next_state][0] - self.V[self.current_state][0]


class AdaptivePlannerAgent(AbstractAdaptiveAgent):
    def __init__(self, R, transition, gain_updater, learning_rates, meta_lr, environment, policy, gamma, update_frequency=1, kp=1, kd=0, ki=0, alpha=0.05, beta=0.95):
        self.transition = transition
        self.R = R.reshape((-1, 1))
        super().__init__(gain_updater, learning_rates, meta_lr, environment, policy, gamma, update_frequency, kp, kd, ki, alpha, beta)

    def update_value(self):
        BR = self.BR()
        self.previous_z, self.z = self.z, self.beta * self.z + self.alpha * BR
        self.previous_Vp, self.Vp, self.previous_previous_V, self.previous_V, self.V = \
            self.Vp, self.V, self.previous_V, self.V, self.V + self.kp * BR + self.kd * (self.V - self.previous_V) + self.ki * self.z

    def BR(self):
        """Return the bellman residual"""
        return self.R + self.gamma * self.transition @ self.V - self.V



class AbstractGainUpdater():
    def __init__(self, lambd):
        self.agent = None
        self.epsilon = 1e-20
        self.lambd = lambd

    def set_agent(self, agent):
        self.agent = agent
        self.num_states = self.agent.num_states
        self.gamma = self.agent.gamma
        self.meta_lr = self.agent.meta_lr

        self.kp = self.agent.kp
        self.ki = self.agent.ki
        self.kd = self.agent.kd
        self.alpha = self.agent.alpha
        self.beta = self.agent.beta

    def intermediate_update(self):
        """This is an update during a delay in case some internal variables need to change while V is changing."""
        self.calculate_updated_values(intermediate=True)

    def calculate_updated_values(self, intermediate=False):
        raise NotImplementedError

    def update_gains(self):
        self.agent.kp = self.kp
        self.agent.ki = self.ki
        self.agent.kd = self.kd

        # For now, don't update alpha and beta, later add functionality to be able to choose whether or not to update them
        # self.agent.alpha = self.alpha
        # self.agent.beta = self.beta



class LogSpaceUpdater(AbstractGainUpdater):
    def __init__(self, num_states, N_p=0.75, N_I=1, N_d=0.1, lambd=0):
        self.fp, self.fd, self.fi = (np.zeros((num_states, 1)) for _ in range(3))
        self.fp_next, self.fd_next, self.fi_next = (np.zeros((num_states, 1)) for _ in range(3))
        self.BR = (np.zeros((num_states, 1)))

        super().__init__(lambd)

        self.N_p = N_p
        self.N_I = N_I
        self.N_d = N_d

        self.lambda_p = np.log(1 - N_p)
        self.lambda_I = self.lambda_d = 0

    def calculate_updated_values(self, intermediate=False):
        reward = self.agent.reward
        next_state, current_state = self.agent.next_state, self.agent.current_state
        V, Vp, z = self.agent.V, self.agent.Vp, self.agent.z
        alpha, beta = self.alpha, self.beta
        gamma, lr = self.gamma, self.agent.lr
        BR = reward + gamma * V[next_state] - V[current_state]
        if not intermediate:
            self.kp = np.exp(self.lambda_p) + self.N_p
            self.ki = (2 * self.N_I) / (1 + np.exp(-self.lambda_I)) - self.N_I
            self.kd = (2 * self.N_d) / (1 + np.exp(-self.lambda_d)) - self.N_d

        self.lambda_p -= lr * self.meta_lr * BR * (gamma * self.fp[next_state] - self.fp[current_state]) \
            * (self.kp - self.N_p)
        self.lambda_d -= lr * self.meta_lr * BR * (gamma * self.fd[next_state] - self.fd[current_state]) \
            * (self.kd + self.N_d) * (1/2 - (self.kd / (2 * self.N_d)))
        self.lambda_I -= lr * self.meta_lr * BR * (gamma * self.fi[next_state] - self.fi[current_state]) \
            * (self.ki + self.N_I) * (1/2 - (self.ki / (2 * self.N_I)))

        self.fp[current_state] = BR
        self.fd[current_state] = (V[current_state] - Vp[current_state])
        self.fi[current_state] = (beta * z[current_state] + alpha * BR)



class DiagonalSoftGainUpdater(AbstractGainUpdater):
    def __init__(self, num_states, lambd=0):
        self.fp, self.fd, self.fi = (np.zeros((num_states)) for _ in range(3))
        self.fp_next, self.fd_next, self.fi_next = (np.zeros((num_states, 1)) for _ in range(3))
        self.BR = (np.zeros((num_states, 1)))

        super().__init__(lambd)

        self.kp = np.ones((num_states))
        self.ki = np.zeros((num_states))
        self.kd = np.zeros((num_states))

    def calculate_updated_values(self, intermediate=False):
        reward = self.agent.reward
        next_state, current_state = self.agent.next_state, self.agent.current_state
        V, Vp, z = self.agent.V, self.agent.Vp, self.agent.z
        alpha, beta = self.alpha, self.beta
        gamma, lr = self.gamma, self.agent.lr
        BR = reward + gamma * V[next_state] - V[current_state]
        if not intermediate:
            self.kp[current_state] -= self.meta_lr * BR * (gamma * self.fp[next_state] - self.fp[current_state])
            self.kd[current_state] -= self.meta_lr * BR * (gamma * self.fd[next_state] - self.fd[current_state])
            self.ki[current_state] -= self.meta_lr * BR * (gamma * self.fi[next_state] - self.fi[current_state])

        self.fp[current_state] = lr * BR
        self.fd[current_state] = lr * (V[current_state] - Vp[current_state])
        self.fi[current_state] = lr * (beta * z[current_state] + alpha * BR)


class DiagonalLogSpaceUpdater(AbstractGainUpdater):
    def __init__(self, num_states, N_p=0.75, N_I=1, N_d=0.1, lambd=0):
        self.fp, self.fd, self.fi = (np.zeros((num_states, 1)) for _ in range(3))
        self.fp_next, self.fd_next, self.fi_next = (np.zeros((num_states, 1)) for _ in range(3))
        self.BR = (np.zeros((num_states, 1)))

        super().__init__(lambd)

        self.kp = np.ones((num_states, 1))
        self.ki = np.zeros((num_states, 1))
        self.kd = np.zeros((num_states, 1))

        self.N_p = N_p
        self.N_I = N_I
        self.N_d = N_d

        self.lambda_p = np.full((num_states, 1), np.log(1 - N_p))
        self.lambda_I = np.zeros((num_states, 1))
        self.lambda_d = np.zeros((num_states, 1))

    def calculate_updated_values(self, intermediate=False):
        reward = self.agent.reward
        next_state, current_state = self.agent.next_state, self.agent.current_state
        V, Vp, z = self.agent.V, self.agent.Vp, self.agent.z
        alpha, beta = self.alpha, self.beta
        gamma, lr = self.gamma, self.agent.lr
        BR = reward + gamma * V[next_state][0] - V[current_state][0]
        if not intermediate:
            self.kp[current_state] = np.exp(self.lambda_p[current_state][0]) + self.N_p
            self.ki[current_state] = (2 * self.N_I) / (1 + np.exp(-self.lambda_I[current_state][0])) - self.N_I
            self.kd[current_state] = (2 * self.N_d) / (1 + np.exp(-self.lambda_d[current_state][0])) - self.N_d

        self.lambda_p[current_state] -= self.meta_lr * BR * (gamma * self.fp[next_state][0] - self.fp[current_state][0]) \
            * max(0.01, self.lambda_p[current_state][0] + 1)
        self.lambda_d[current_state] -= self.meta_lr * BR * (gamma * self.fd[next_state][0] - self.fd[current_state][0]) \
            * (self.kd[current_state][0] + self.N_d) * (1/2 - (self.kd[current_state][0] / (2 * self.N_d)))
        self.lambda_I[current_state] -= self.meta_lr * BR * (gamma * self.fi[next_state][0] - self.fi[current_state][0]) \
            * (self.ki[current_state][0] + self.N_I) * (1/2 - (self.ki[current_state][0] / (2 * self.N_I)))

        self.fp[current_state] = BR
        self.fd[current_state] = (V[current_state][0] - Vp[current_state][0])
        self.fi[current_state] = (beta * z[current_state][0] + alpha * BR)



class NaiveSoftGainUpdater(AbstractGainUpdater):
    def __init__(self, num_states, lambd=0):
        self.fp, self.fd, self.fi = (np.zeros((num_states)) for _ in range(3))
        self.fp_next, self.fd_next, self.fi_next = (np.zeros((num_states, 1)) for _ in range(3))

        super().__init__(lambd)

        self.p_update = 0
        self.d_update = 0
        self.i_update = 0

    def calculate_updated_values(self, intermediate=False):
        reward = self.agent.reward
        next_state, current_state = self.agent.next_state, self.agent.current_state
        V, Vp, z = self.agent.previous_V, self.agent.previous_Vp, self.agent.previous_z
        alpha, beta = self.alpha, self.beta
        gamma, lr = self.gamma, self.agent.lr

        BR = reward + gamma * V[next_state][0] - V[current_state][0]
        if intermediate:
            self.p_update += BR * (gamma * self.fp[next_state] - self.fp[current_state])
            self.d_update += BR * (gamma * self.fd[next_state] - self.fd[current_state])
            self.i_update += BR * (gamma * self.fi[next_state] - self.fi[current_state])
        else:
            self.kp = (1 - self.lambd) * self.kp - self.meta_lr * self.p_update / self.agent.update_frequency
            self.kd = (1 - self.lamdb) * self.kd - self.meta_lr * self.d_update / self.agent.update_frequency
            self.ki = (1 - self.lambd) * self.ki - self.meta_lr * self.i_update / self.agent.update_frequency

            self.p_update = 0
            self.d_update = 0
            self.i_update = 0

        self.fp[current_state] = lr * BR
        self.fd[current_state] = lr * (V[current_state] - Vp[current_state])
        self.fi[current_state] = lr * (beta * z[current_state][0] + alpha * BR)


class TrueSoftGainUpdater(AbstractGainUpdater):
    def __init__(self, num_states, lambd=0):
        # The order is [p, i, d]
        self.num_states = num_states
        self.reset_partials()

        self.BR = (np.zeros((num_states)))

        super().__init__(lambd)

    def reset_partials(self):
        self.fs = [np.zeros((self.num_states)) for _ in range(3)]
        self.gs = [np.zeros((self.num_states)) for _ in range(3)]
        self.hs = [np.zeros((self.num_states)) for _ in range(3)]

        self.total_kp, self.total_ki, self.total_kd = 0, 0, 0

    def calculate_updated_values(self, intermediate=False):
        reward = self.agent.reward
        next_state, current_state = self.agent.next_state, self.agent.current_state
        V, Vp, z = self.agent.previous_V, self.agent.previous_Vp, self.agent.previous_z
        alpha, beta = self.alpha, self.beta
        gamma, lr = self.gamma, self.agent.lr
        update_D_rate, update_I_rate = self.agent.update_D_rate_value, self.agent.update_I_rate_value

        BR = reward + gamma * V[next_state] - V[current_state]

        self.total_kp += BR * (gamma * self.fs[0][next_state] - self.fs[0][current_state])
        self.total_ki += BR * (gamma * self.fs[1][next_state] - self.fs[1][current_state])
        self.total_kd += BR * (gamma * self.fs[2][next_state] - self.fs[2][current_state])
        if not intermediate:
            self.kp -= self.meta_lr * self.total_kp
            self.ki -= self.meta_lr * self.total_ki
            self.kd -= self.meta_lr * self.total_kd

            self.reset_partials()

        common_updates = [0, 0, 0]
        for i in range(3):
            common_updates[i] = self.kp * (gamma * self.fs[i][next_state] - self.fs[i][current_state]) \
                + self.ki * (beta * self.hs[i][current_state] + alpha * (gamma * self.fs[i][next_state] - self.fs[i][current_state])) \
                + self.kd * (self.fs[i][current_state] - self.gs[i][current_state])
        for i in range(3):
            self.gs[i][current_state] = (1 - update_D_rate) * self.gs[i][current_state] + update_D_rate * self.fs[i][current_state]
            self.hs[i][current_state] = (1 - update_I_rate) * self.hs[i][current_state] \
                + update_I_rate * (beta * self.hs[i][current_state] + alpha * (gamma * self.fs[i][next_state] - self.fs[i][current_state]))

        self.fs[0][current_state] += lr * (BR + common_updates[0])
        self.fs[1][current_state] += lr * ((beta * z[current_state] + alpha * BR) + common_updates[1])
        self.fs[2][current_state] += lr * ((V[current_state] - Vp[current_state]) +  common_updates[2])


class LogisticExactUpdater(AbstractGainUpdater):
    def __init__(self, transition, reward, num_states, N_p=0.75, N_d=0.5, N_I=1, lambd=0):
        super().__init__(lambd)
        self.transition = transition
        self.reward = reward.reshape(-1, 1)
        self.N_p = N_p
        self.N_d = N_d
        self.N_I = N_I

        self.lambda_p = -np.log(4 * N_p - 1)
        self.lambda_d = 0
        self.lambda_I = 0

    def calculate_updated_values(self, intermediate=False):
        V, Vp, z = self.agent.V, self.agent.Vp, self.agent.z
        previous_V = self.agent.previous_V
        alpha, beta = self.alpha, self.beta
        gamma, lr = self.gamma, self.agent.lr

        BR = self.reward + gamma * self.transition @ V - V
        if not intermediate:
            self.kp = self.N_p / (1 + np.exp(-self.lambda_p)) + 0.75
            self.ki = (2 * self.N_I) / (1 + np.exp(-self.lambda_I)) - self.N_I
            self.kd = (2 * self.N_d) / (1 + np.exp(-self.lambda_d)) - self.N_d

        self.lambda_p -= lr * self.meta_lr * BR.T @ (gamma * self.transition @ BR - BR) \
            * (self.kp - 0.75) * (1 - (self.kp - 0.75) / self.N_p)
        self.lambda_d = max(0, self.lambda_d - lr * self.meta_lr * BR.T @ (gamma * self.transition @ (V - Vp) - (V - Vp)) \
            * (self.kd + self.N_d) * (1/2 - (self.kd / (2 * self.N_d))))
        self.lambda_I -= lr * self.meta_lr * BR.T @ (gamma * self.transition @ (beta * z + alpha * BR) - (beta * z + alpha * BR)) \
            * (self.ki + self.N_I) * (1/2 - (self.ki / (2 * self.N_I)))

class SemiGradientUpdater(AbstractGainUpdater):
    def __init__(self, num_states, lambd=0):
        super().__init__(lambd)

        self.d_update = 0
        self.i_update = 0
        self.p_update = 0

        self.running_BR = 0
        self.previous_average_BR = float("inf")

    def set_agent(self, agent):
        self.running_BR = 0
        self.previous_average_BR = float("inf")
        self.d_update = 0
        self.i_update = 0
        self.p_update = 0
        return super().set_agent(agent)

    def calculate_updated_values(self, intermediate=False):
        reward = self.agent.reward
        next_state, current_state = self.agent.next_state, self.agent.current_state
        V, Vp, z = self.agent.V, self.agent.Vp, self.agent.z
        alpha, beta = self.alpha, self.beta
        gamma, lr = self.gamma, self.agent.lr

        BR = reward + gamma * V[next_state] - V[current_state]
        if intermediate:
            self.p_update += lr * BR * BR
            self.d_update += lr * BR * (V[current_state] - Vp[current_state])
            self.i_update += lr * BR * (beta * z[next_state] + alpha * BR)

            self.running_BR += BR
        if not intermediate:
            # Decay if we are not improving, else adapt the gains
            if abs(self.running_BR / self.agent.update_frequency) < abs(self.previous_average_BR):
                self.d_update = 0
                self.i_update = 0
                self.p_update = 0

                self.previous_average_BR = self.running_BR / self.agent.update_frequency

            self.kp = 1 + (1 - self.lambd) * (self.kp - 1) + self.meta_lr * self.p_update / self.agent.update_frequency
            self.kd = (1 - self.lambd) * self.kd + self.meta_lr * self.d_update / self.agent.update_frequency
            self.ki = (1 - self.lambd) * self.ki + self.meta_lr * self.i_update / self.agent.update_frequency

            self.d_update = 0
            self.i_update = 0
            self.p_update = 0

            self.running_BR = 0



class DiagonalSemiGradient(AbstractGainUpdater):
    def __init__(self, num_states, lambd=0):
        super().__init__(lambd)

    def calculate_updated_values(self, intermediate=False):
        reward = self.agent.reward
        next_state, current_state = self.agent.next_state, self.agent.current_state
        V, Vp, z = self.agent.V, self.agent.Vp, self.agent.z
        alpha, beta = self.alpha, self.beta
        gamma, lr = self.gamma, self.agent.lr

        BR = reward + gamma * V[next_state] - V[current_state]
        if not intermediate:
            self.kp[current_state] = max(1, (1 - self.lambd) * self.kp[current_state] + lr * self.meta_lr * BR * BR)
            self.kd[current_state] = max(0, (1 - self.lambd) * self.kd[current_state] + lr * self.meta_lr * BR * (V[current_state] - Vp[current_state]))
            self.ki[current_state] = (1 - self.lambd) * self.ki[current_state] + lr * self.meta_lr * BR * (beta * z[current_state] + alpha * BR)

class EmpiricalCostUpdater(AbstractGainUpdater):
    """We need agent.update_frequency = 2 for this to mathematically make sense"""
    def calculate_updated_values(self, intermediate=False):
        if intermediate: return
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
        ki, alpha, beta = self.ki, self.alpha, self.beta
        z, previous_z = self.agent.z, self.agent.previous_z
        current_z_update = beta * z[current_state] - alpha * current_BR
        previous_z_update = beta * previous_z[previous_state] - alpha * previous_BR

        BR_ki_grad = approx_diff(current_z_update, previous_z_update)
        BR_alpha_grad = approx_diff(current_BR * ki, previous_BR * ki)
        BR_beta_grad = approx_diff(ki * z[current_state], ki * previous_z[previous_state])

        # Perform the updates
        normalizer = 1  # self.epsilon + (current_BR ** 2)
        update = lambda n: (next_BR * n) / normalizer

        logging.debug(
            f"""
            kp={self.agent.kp}
            kd={self.agent.kd}
            ki={self.agent.ki}
            alpha={self.agent.alpha}
            beta={self.agent.beta}

            BR_kp_grad={BR_kp_grad}
            BR_ki_grad={BR_ki_grad}
            BR_kd_grad={BR_kd_grad}
            BR_alpha_grad={BR_alpha_grad}
            BR_beta_grad={BR_beta_grad}
            """
        )


        meta_lr = self.meta_lr
        self.kp -= meta_lr * update(BR_kp_grad)
        self.kd -= meta_lr * update(BR_kd_grad)
        self.ki -= meta_lr * update(BR_ki_grad)
        self.alpha -= meta_lr * update(BR_alpha_grad)
        self.beta -= meta_lr * update(BR_beta_grad)


class AbstractOriginalCostUpdater(AbstractGainUpdater):
    def __init__(self, scale_by_lr, lambd=0):
        super().__init__(lambd)
        self.scale_by_lr = scale_by_lr
        self.partial_br_kp = []
        self.partial_br_ki = []
        self.partial_br_kd = []
        self.partial_br_alpha = []
        self.partial_br_beta = []

    def calculate_updated_values(self, intermediate=False):
        if intermediate: return
        partial_br_kp, partial_br_ki, partial_br_kd, partial_br_beta, BR_current, BR_previous = self.get_gradient_terms()

        if self.scale_by_lr:
            normalizer = 1
        else:
            normalizer = self.epsilon + BR_previous.T @ BR_previous

        kp_grad = (BR_current.T @ partial_br_kp) / normalizer
        ki_grad = (BR_current.T @ partial_br_ki) / normalizer
        kd_grad = (BR_current.T @ partial_br_kd) / normalizer
        beta_grad = (BR_current.T @ partial_br_beta) / normalizer
        alpha_grad = self.ki * kp_grad

        # store a list of partial derivatives to be plotted later at a specific state
        self.partial_br_kp.append(partial_br_kp[10][0])
        self.partial_br_ki.append(partial_br_ki[10][0])
        self.partial_br_kd.append(partial_br_kd[10][0])
        self.partial_br_beta.append(partial_br_beta[10][0])
        self.partial_br_alpha.append(self.ki * partial_br_kp[10][0])

        # Renormalize alpha and beta
        # self.alpha, self.beta = self.alpha / (self.alpha + self.beta), self.beta / (self.alpha + self.beta)
        if self.scale_by_lr:
            lr = self.agent.lr * self.meta_lr
        else:
            lr = self.meta_lr
        self.kp = (1 - self.lambd) * self.kp - lr * kp_grad
        self.ki = (1 - self.lambd) * self.ki - lr * ki_grad
        self.kd = (1 - self.lambd) * self.kd - lr * kd_grad
        self.alpha = (1 - self.lambd) * self.alpha - lr * alpha_grad
        self.beta = (1 - self.lambd) * self.beta - lr * beta_grad

    def plot(self):
        """Plot the partial derivatives of the BR function with respect to each gain"""
        fig, axs = plt.subplots(2, 3)
        axs[0, 0].plot(self.partial_br_kp)
        axs[0, 0].set_title("kp")
        axs[0, 1].plot(self.partial_br_ki)
        axs[0, 1].set_title("ki")
        axs[0, 2].plot(self.partial_br_kd)
        axs[0, 2].set_title("kd")
        axs[1, 0].plot(self.partial_br_alpha)
        axs[1, 0].set_title("alpha")
        axs[1, 1].plot(self.partial_br_beta)
        axs[1, 1].set_title("beta")
        fig.delaxes(axs[1, 2])
        plt.show()

    def get_gradient_terms():
        raise NotImplementedError


class ExactUpdater(AbstractOriginalCostUpdater):
    def __init__(self, transition, reward, scale, lambd=0):
        super().__init__(scale, lambd)
        self.transition = transition.astype(np.float64)
        self.reward = reward.reshape(-1, 1).astype(np.float64)

    def get_gradient_terms(self):
        V, Vp, Vpp = self.agent.V, self.agent.previous_V, self.agent.previous_Vp
        z, zp = self.agent.z, self.agent.previous_z

        BR = lambda n: self.gamma * self.transition @ n + self.reward - n

        BR_current = BR(V)
        BR_previous = BR(Vp)

        mult = lambda n: -n

        partial_br_kp = mult(BR_previous)
        partial_br_kd = mult(Vp - Vpp)
        partial_br_ki = mult(z)
        partial_br_beta = self.ki * mult(zp)

        return partial_br_kp, partial_br_ki, partial_br_kd, partial_br_beta, BR_current, BR_previous


class DiagonalExactUpdater(AbstractGainUpdater):
    def __init__(self, transition, reward, lambd=0):
        super().__init__(lambd)
        self.transition = transition.astype(np.float64)
        self.reward = reward.reshape(-1, 1).astype(np.float64)

    def calculate_updated_values(self, intermediate=False):
        reward = self.agent.reward
        next_state, current_state = self.agent.next_state, self.agent.current_state
        V, Vp, z = self.agent.V, self.agent.Vp, self.agent.z
        alpha, beta = self.alpha, self.beta
        gamma, lr = self.gamma, self.agent.lr

        BR = ((self.gamma * self.transition @ V) + self.reward - V)[current_state]
        if not intermediate:
            # self.kp[current_state] += self.meta_lr * BR * BR
            self.kd[current_state] = max(0, self.kd[current_state] + lr * self.meta_lr * BR * (V[current_state] - Vp[current_state]))
            self.ki[current_state] += lr * self.meta_lr * BR * (beta * z[current_state] + alpha * BR)


class SamplerUpdater(AbstractOriginalCostUpdater):
    def __init__(self, sample_size, scale, lambd=0):
        super().__init__(scale, lambd)
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

        partial_br_beta *= self.ki

        return partial_br_kp, partial_br_ki, partial_br_kd, partial_br_beta, BR_current, BR_previous


class BellmanExactUpdater(AbstractOriginalCostUpdater):
    """Test having knowledge of the transitions to update the gradients, but not the Bellman residual"""
    def __init__(self, transition, reward, sample_size, scale, lambd=0):
        super().__init__(scale, lambd)
        self.transition = transition
        self.reward = reward.reshape(-1, 1)
        self.sample_size = sample_size

    def get_gradient_terms(self):
        V, Vp, Vpp = self.agent.V, self.agent.previous_V, self.agent.previous_previous_V
        z, zp = self.agent.z, self.agent.previous_z

        BR_current = self.gamma * self.transition @ V + self.reward - V
        BR_previous = self.gamma * self.transition @ Vp + self.reward - Vp

        previous_reward, reward, previous_state, current_state, next_state = \
            self.agent.previous_reward, self.agent.reward, self.agent.previous_state, self.agent.current_state, self.agent.next_state

        partial_br_kp = Vpp[previous_state] - 2 * self.gamma * Vpp[current_state] \
                + (self.gamma ** 2) * Vpp[next_state] - previous_reward + self.gamma * reward

        partial_br_ki = self.gamma * z[next_state] - z[current_state]

        partial_br_kd = Vpp[previous_state] - Vp[previous_state] \
                    - self.gamma * Vpp[current_state] + self.gamma * Vp[current_state]
        partial_br_beta = self.ki * (self.gamma * zp[next_state] - zp[current_state])

        return partial_br_kp, partial_br_ki, partial_br_kd, partial_br_beta, BR_current[current_state], BR_previous[current_state]



class PartialsExactUpdater(AbstractOriginalCostUpdater):
    """Test having knowledge of the transitions to update the gradients, but not the Bellman residual"""
    def __init__(self, transition, reward, sample_size, scale, lambd=0):
        super().__init__(scale, lambd)
        self.transition = transition
        self.reward = reward.reshape(-1, 1)
        self.sample_size = sample_size

    def get_gradient_terms(self):
        V, Vp, Vpp = self.agent.V, self.agent.previous_V, self.agent.previous_previous_V
        z, zp = self.agent.z, self.agent.previous_z

        BR = lambda n: self.gamma * self.transition @ n + self.reward - n

        BR_current = BR(V)
        BR_previous = BR(Vp)

        mult = lambda n: (self.gamma * self.transition @ n) - n

        partial_br_kp = mult(BR_previous)
        partial_br_kd = mult(Vp - Vpp)
        partial_br_ki = mult(z)
        partial_br_beta = self.ki * mult(zp)

        replay_buffer = self.agent.replay_buffer
        for state in range(self.num_states):
            for _ in range(self.sample_size):
                _, reward, _, next_state = \
                    replay_buffer[state][np.random.randint(0, len(replay_buffer[state]))]
                BR_current[state] += reward + self.gamma * V[next_state] - V[state]

            for _ in range(self.sample_size):
                _, reward, _, next_state = \
                    replay_buffer[state][np.random.randint(0, len(replay_buffer[state]))]
                BR_previous[state] += reward + self.gamma * Vp[next_state] - Vp[state]

        return partial_br_kp, partial_br_ki, partial_br_kd, partial_br_beta, BR_current, BR_previous