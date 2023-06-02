import numpy as np
from collections import defaultdict
import logging
import matplotlib.pyplot as plt

from TabularPID.Agents.Agents import Agent

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

        self.Q, self.Qp, self.z, self.previous_Q, self.previous_Qp, self.previous_z, self.previous_previous_Q \
            = (np.zeros((self.num_states, self.num_actions), dtype=np.longdouble) for _ in range(7))

        self.previous_previous_state, self.previous_state, self.current_state, self.next_state = 0, 0, 0, 0
        self.previous_reward, self.reward = 0, 0

        self.gain_updater.set_agent(self)

    def estimate_value_function(self, num_iterations=1000, test_function=None, initial_Q=None, stop_if_diverging=True, follow_trajectory=True):
        self.reset()
        # Q is the current value function, Qp is the previous value function
        # Qp stores the previous value of the x state when it was last changed
        if initial_Q is not None:
            self.Q = initial_Q.copy()
            self.Qp = initial_Q.copy()

        history = np.zeros((num_iterations))
        gain_history = np.zeros((num_iterations, 5))

        for k in range(num_iterations):
            self.previous_previous_state, self.previous_state, self.previous_reward = self.previous_state, self.current_state, self.reward
            self.current_state, self.action, self.next_state, self.reward = self.take_action(follow_trajectory)

            self.replay_buffer[self.current_state].append(
                (self.previous_reward, self.reward, self.previous_state, self.next_state, self.action)
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
                history[k] = test_function(self.Q, self.Qp, self.BR)
                if stop_if_diverging and history[k] > 1.5 * history[0]:
                    history[k:] = float("inf")
                    break

        logging.info(f"Final gains are: kp: {self.kp}, ki: {self.ki}, kd: {self.kd}, alpha: {self.alpha}, beta: {self.beta}")
        if test_function is not None:
            return self.Q, gain_history, history

        return self.Q, gain_history

    def update_value(self):
        """Update Q, Qp, z and the previous versions"""
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
        action = self.action

        self.previous_previous_Q, self.previous_Q = self.previous_Q.copy(), self.Q.copy()
        self.previous_Qp, self.previous_z = self.Qp.copy(), self.z.copy()

        #Update the value function using the floats kp, ki, kd
        BR = self.BR()
        new_Q = self.Q[state][action] + self.kp * BR + self.kd * (self.Q[state][action] - self.Qp[state][action]) + self.ki * (self.beta * self.z[state][action] + self.alpha * BR)
        new_z = self.beta * self.z[state][action] + self.alpha * BR
        new_Qp = self.Q[state][action]

        self.previous_previous_Q[state][action], self.previous_Q[state][action], self.Q[state][action] = self.previous_Q[state][0], self.Q[state][0], (1 - lr) * self.Q[state][0] + lr * new_Q
        self.previous_z[state][action], self.z[state][action] = self.z[state][action], (1 - update_I_rate) * self.z[state][action] + update_I_rate * new_z
        self.previous_Qp[state][action], self.Qp[state][action] = self.Qp[state][action], (1 - update_D_rate) * self.Qp[state][action] + update_D_rate * new_Qp

    def BR(self):
        """Return the empirical bellman residual"""
        return self.reward + self.gamma * np.max(self.Q[self.next_state]) - self.Q[self.current_state][self.action]


class DiagonalAdaptiveSamplerAgent(AbstractAdaptiveAgent):
    def __init__(self, gain_updater, learning_rates, meta_lr, environment, policy, gamma, update_frequency=1, kp=1, kd=0, ki=0, alpha=0.05, beta=0.95):
        super().__init__(gain_updater, learning_rates, meta_lr, environment, policy, gamma, update_frequency, kp, kd, ki, alpha, beta)
        
        gain_updater.set_agent(self)
    
    def reset(self):
        self.environment.reset()
        self.kp = np.full((self.num_states, self.num_actions), self.original_kp, dtype=np.longdouble)
        self.ki = np.full((self.num_states, self.num_actions), self.original_ki, dtype=np.longdouble)
        self.kd = np.full((self.num_states, self.num_actions), self.original_kd, dtype=np.longdouble)

        self.alpha, self.beta = self.original_alpha, self.original_beta
        self.lr, self.previous_lr, self.update_D_rate_value, self.previous_update_D_rate_value, self.update_I_rate_value, self.previous_update_I_rate_value = 0, 0, 0, 0, 0, 0

        self.replay_buffer = defaultdict(list)
        self.frequencies = np.zeros((self.num_states))

        self.Q, self.Qp, self.z, self.previous_Q, self.previous_Qp, self.previous_z, self.previous_previous_Q \
            = (np.zeros((self.num_states, self.num_actions), dtype=np.longdouble) for _ in range(7))

        self.previous_previous_state, self.previous_state, self.current_state, self.next_state = 0, 0, 0, 0
        self.previous_reward, self.reward = 0, 0

        self.gain_updater.set_agent(self)

    def update_value(self):
        lr = self.learning_rate(self.frequencies[self.current_state])
        update_D_rate = self.update_D_rate(self.frequencies[self.current_state])
        update_I_rate = self.update_I_rate(self.frequencies[self.current_state])

        self.previous_lr, self.lr = self.lr, lr

        state = self.current_state
        action = self.action

        BR = self.BR()
        new_Q = self.Q[state][action] + self.kp[state][action] * BR + self.kd[state][action] * (self.Q[state][action] - self.Qp[state][action]) + self.ki[state][action] * (self.beta * self.z[state][action] + self.alpha * BR)
        new_z = self.beta * self.z[state][action] + self.alpha * BR
        new_Qp = self.Q[state][action]

        self.previous_previous_Q[state][action], self.previous_Q[state][action], self.Q[state][action] = self.previous_Q[state][action], self.Q[state][action], (1 - lr) * self.Q[state][action] + lr * new_Q
        self.previous_z[state][action], self.z[state][action] = self.z[state][action], (1 - update_I_rate) * self.z[state][action] + update_I_rate * new_z
        self.previous_Qp[state][action], self.Qp[state][action] = self.Qp[state][action], (1 - update_D_rate) * self.Qp[state][action] + update_D_rate * new_Qp

    def BR(self):
        """Return the empirical bellman residual"""
        return self.reward + self.gamma * np.max(self.Q[self.next_state]) - self.Q[self.current_state][self.action]


class AdaptivePlannerAgent(AbstractAdaptiveAgent):
    def __init__(self, R, transition, gain_updater, learning_rates, meta_lr, environment, policy, gamma, update_frequency=1, kp=1, kd=0, ki=0, alpha=0.05, beta=0.95):
        self.transition = transition
        self.R = R.reshape((-1, 1))
        super().__init__(gain_updater, learning_rates, meta_lr, environment, policy, gamma, update_frequency, kp, kd, ki, alpha, beta)

    def update_value(self):
        BR = self.BR()
        self.previous_z, self.z = self.z, self.beta * self.z + self.alpha * BR
        self.previous_Qp, self.Qp, self.previous_previous_Q, self.previous_Q, self.Q = \
            self.Qp, self.Q, self.previous_Q, self.Q, self.Q + self.kp * BR + self.kd * (self.Q - self.previous_Q) + self.ki * self.z

    def BR(self):
        """Return the bellman residual"""
        max_future_reward = 0
        for i in range(self.num_actions):
            expected_reward = sum(self.P[i, :, j] * self.Q[j] for j in range(self.num_states))
            max_future_reward = max(max_future_reward, expected_reward)
        return self.R + self.gamma * max_future_reward - self.Q

