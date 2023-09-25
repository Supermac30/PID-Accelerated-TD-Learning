import numpy as np
from collections import defaultdict
import logging
import matplotlib.pyplot as plt
from TabularPID.MDPs.Policy import Policy

from TabularPID.Agents.Agents import Agent, learning_rate_function
        

class AbstractAdaptiveAgent(Agent):
    def __init__(self, gain_updater, learning_rates, meta_lr, environment, gamma, update_frequency=1, kp=1, kd=0, ki=0, alpha=0.05, beta=0.95):
        super().__init__(environment, None, gamma)

        self.meta_lr = meta_lr
        self.learning_rate = learning_rates[0]
        self.update_I_rate = learning_rates[1]
        self.update_D_rate = learning_rates[2]
        self.update_frequency = update_frequency

        self.gain_updater = gain_updater

        self.original_kp, self.original_ki, self.original_kd = kp, ki, kd
        self.original_alpha, self.original_beta = alpha, beta

        self.reset()

    def set_seed(self, seed):
        self.environment.set_seed(seed)

    def reset(self, reset_environment=True):
        if reset_environment:
            self.environment.reset()
        self.kp_A, self.ki_A, self.kd_A = self.original_kp, self.original_ki, self.original_kd
        self.kp_B, self.ki_B, self.kd_B = self.original_kp, self.original_ki, self.original_kd
        self.alpha, self.beta = self.original_alpha, self.original_beta
        self.lr, self.previous_lr, self.update_D_rate_value, self.previous_update_D_rate_value, self.update_I_rate_value, self.previous_update_I_rate_value = 0, 0, 0, 0, 0, 0

        self.replay_buffer = defaultdict(list)
        self.frequencies = np.zeros((self.num_states))

        self.Q_A, self.Qp_A, self.z_A, self.previous_Q_A, self.previous_Qp_A, self.previous_z_A, self.previous_previous_Q_A \
            = (np.zeros((self.num_states, self.num_actions), dtype=np.longdouble) for _ in range(7))
        
        self.Q_B, self.Qp_B, self.z_B, self.previous_Q_B, self.previous_Qp_B, self.previous_z_B, self.previous_previous_Q_B \
            = (np.zeros((self.num_states, self.num_actions), dtype=np.longdouble) for _ in range(7))

        self.previous_previous_state, self.previous_state, self.current_state, self.next_state = 0, 0, 0, 0
        self.previous_reward, self.reward = 0, 0

        self.num_steps = 0
        
        self.gain_updater.set_agent(self)

        self.policy = Policy(self.num_actions, self.num_states, self.environment.prg, None)

    def estimate_value_function(self, num_iterations=1000, test_function=None, initial_Q=None, stop_if_diverging=True, follow_trajectory=False, reset_environment=True):
        self.reset(reset_environment)
        # Q is the current value function, Qp is the previous value function
        # Qp stores the previous value of the x state when it was last changed
        if initial_Q is not None:
            self.Q_A = initial_Q.copy()
            self.Qp_A = initial_Q.copy()
            self.Q_B = initial_Q.copy()
            self.Qp_B = initial_Q.copy()

        history = np.zeros((num_iterations))
        gain_history = np.zeros((num_iterations, 5))

        for k in range(num_iterations):
            self.num_steps += 1
            self.previous_previous_state, self.previous_state, self.previous_reward = self.previous_state, self.current_state, self.reward
            self.current_state, self.action, self.next_state, self.reward = self.take_action(follow_trajectory, is_q=True)

            self.replay_buffer[self.current_state].append(
                (self.previous_reward, self.reward, self.previous_state, self.next_state, self.action)
            )

            self.frequencies[self.current_state] += 1
            self.update_value()
            if (k + 1) % self.update_frequency == 0:
                self.gain_updater.calculate_updated_values()
                self.gain_updater.update_gains()
            else:
                self.gain_updater.intermediate_update()

            # Keep a record
            try:
                gain_history[k][0] = self.kp_A
                gain_history[k][1] = self.ki_A
                gain_history[k][2] = self.kd_A
                gain_history[k][3] = self.alpha
                gain_history[k][4] = self.beta
            except:
                pass

            if test_function is not None:
                history[k] = test_function(self.Q_A, self.Qp_A, self.BR)
                if stop_if_diverging and history[k] > 1.5 * history[0]:
                    history[k:] = float("inf")
                    break

        logging.info(f"Final gains are: kp: {self.kp_A}, ki: {self.ki_A}, kd: {self.kd_A}, alpha: {self.alpha}, beta: {self.beta}")
        if test_function is not None:
            return self.Q_A, gain_history, history

        return self.Q_A, gain_history

    def update_value(self):
        """Update Q, Qp, z and the previous versions"""
        raise NotImplementedError

    def BR(self):
        """Return the bellman residual"""
        raise NotImplementedError
    
    def plot(self, directory=""):
        return self.gain_updater.plot(directory)

    def set_learning_rates(self, a, b, c, d, e, f):
        self.learning_rate = learning_rate_function(a, b)
        self.update_I_rate = learning_rate_function(c, d)
        self.update_D_rate = learning_rate_function(e, f)


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

        self.previous_previous_Q_B, self.previous_Q_B = self.previous_Q_B.copy(), self.Q_B.copy()
        self.previous_Qp_B, self.previous_z_B = self.Qp_B.copy(), self.z_B.copy()

        #Update the value function using the floats kp, ki, kd
        if np.random.random() < 0.5:
            self.updated_A = True

            self.previous_previous_Q_A, self.previous_Q_A = self.previous_Q_A.copy(), self.Q_A.copy()
            self.previous_Qp_A, self.previous_z_A = self.Qp_A.copy(), self.z_A.copy()
            BR = self.BR(True)
            self.previous_BR = BR

            self.p_update = BR
            self.i_update = self.beta * self.z_A[state][action] + self.alpha * BR
            self.d_update = self.Q_A[state][action] - self.Qp_A[state][action]

            new_Q = self.Q_A[state][action] + self.kp_A * self.p_update + self.ki_A * self.i_update + self.kd_A * self.d_update
            new_z = self.beta * self.z_A[state][action] + self.alpha * BR
            new_Qp = self.Q_A[state][action]

            self.previous_previous_Q_A = self.previous_Q_A.copy()
            self.previous_Q_A = self.Q_A.copy()
            self.previous_z_A = self.z_A.copy()
            self.previous_Qp_A = self.Qp_A.copy()

            self.Q_A[state][action] = (1 - lr) * self.Q_A[state][action] + lr * new_Q
            self.z_A[state][action] = (1 - update_I_rate) * self.z_A[state][action] + update_I_rate * new_z
            self.Qp_A[state][action] = (1 - update_D_rate) * self.Qp_A[state][action] + update_D_rate * new_Qp

            self.next_BR = self.BR(True)

        else:
            self.updated_A = False

            self.previous_previous_Q_B, self.previous_Q_B = self.previous_Q_B.copy(), self.Q_B.copy()
            self.previous_Qp_B, self.previous_z_B = self.Qp_B.copy(), self.z_B.copy()
            BR = self.BR(False)
            self.previous_BR = BR

            self.p_update = BR
            self.i_update = self.beta * self.z_B[state][action] + self.alpha * BR
            self.d_update = self.Q_B[state][action] - self.Qp_B[state][action]

            new_Q = self.Q_B[state][action] + self.kp_B * self.p_update + self.ki_B * self.i_update + self.kd_B * self.d_update
            new_z = self.beta * self.z_B[state][action] + self.alpha * BR
            new_Qp = self.Q_B[state][action]

            self.previous_previous_Q_B = self.previous_Q_B.copy()
            self.previous_Q_B = self.Q_B.copy()
            self.previous_z_B = self.z_B.copy()
            self.previous_Qp_B = self.Qp_B.copy()

            self.Q_B[state][action] = (1 - lr) * self.Q_B[state][action] + lr * new_Q
            self.z_B[state][action] = (1 - update_I_rate) * self.z_B[state][action] + update_I_rate * new_z
            self.Qp_B[state][action] = (1 - update_D_rate) * self.Qp_B[state][action] + update_D_rate * new_Qp

            self.next_BR = self.BR(False)

        """
        self.previous_previous_Q[state][action], self.previous_Q[state][action] = self.previous_Q[state][action], self.Q[state][action]
        self.previous_z[state][action] = self.z[state][action]
        self.previous_Qp[state][action] = self.Qp[state][action]
        """

    def BR(self, is_A):
        """Return the empirical bellman residual"""
        if is_A:
            best_action = np.argmax(self.Q_A[self.next_state])
            return int(self.reward) + self.gamma * self.Q_B[self.next_state][best_action] - self.Q_A[self.current_state][self.action]
        else:
            best_action = np.argmax(self.Q_B[self.next_state])
            return int(self.reward) + self.gamma * self.Q_A[self.next_state][best_action] - self.Q_B[self.current_state][self.action]


class DiagonalAdaptiveSamplerAgent(AbstractAdaptiveAgent):
    def __init__(self, gain_updater, learning_rates, meta_lr, environment, gamma, update_frequency=1, kp=1, kd=0, ki=0, alpha=0.05, beta=0.95):
        super().__init__(gain_updater, learning_rates, meta_lr, environment, gamma, update_frequency, kp, kd, ki, alpha, beta)
        
        gain_updater.set_agent(self)
    
    def reset(self, reset_environment=True):
        if reset_environment:
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
        self.policy = Policy(self.num_actions, self.num_states, self.environment.prg, None)

    def update_value(self):
        lr = self.learning_rate(self.frequencies[self.current_state])
        update_D_rate = self.update_D_rate(self.frequencies[self.current_state])
        update_I_rate = self.update_I_rate(self.frequencies[self.current_state])

        self.previous_lr, self.lr = self.lr, lr

        state = self.current_state
        action = self.action

        # if np.random.random() < 0.01:
        #     breakpoint()

        BR = self.BR()
        self.previous_BR = BR

        self.p_update = BR
        self.d_update = self.Q[state][action] - self.Qp[state][action]
        self.i_update = self.beta * self.z[state][action] + self.alpha * BR

        new_Q = self.Q[state][action] + self.kp[state][action] * self.p_update + self.kd[state][action] * self.d_update + self.ki[state][action] * self.i_update
        new_z = self.beta * self.z[state][action] + self.alpha * BR
        new_Qp = self.Q[state][action]

        self.previous_previous_Q = self.previous_Q.copy()
        self.previous_Q = self.Q.copy()
        self.previous_z = self.z.copy()
        self.previous_Qp = self.Qp.copy()

        """
        self.previous_previous_Q[state][action], self.previous_Q[state][action], self.Q[state][action] = self.previous_Q[state][action], self.Q[state][action], (1 - lr) * self.Q[state][action] + lr * new_Q
        self.previous_z[state][action], self.z[state][action] = self.z[state][action], (1 - update_I_rate) * self.z[state][action] + update_I_rate * new_z
        self.previous_Qp[state][action], self.Qp[state][action] = self.Qp[state][action], (1 - update_D_rate) * self.Qp[state][action] + update_D_rate * new_Qp
        """

        # Try modifying this to copy previous_Q into Q, instead of doing it delayed state wise
        # This fits the math better, and might fix the problems we are having
        self.Q[state][action] = (1 - lr) * self.Q[state][action] + lr * new_Q
        self.z[state][action] = (1 - update_I_rate) * self.z[state][action] + update_I_rate * new_z
        self.Qp[state][action] = (1 - update_D_rate) * self.Qp[state][action] + update_D_rate * new_Qp

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


class AbstractGainUpdater():
    def __init__(self, lambd, epsilon=0.1):
        self.agent = None
        self.epsilon = epsilon
        self.lambd = lambd

    def set_agent(self, agent):
        self.agent = agent
        self.num_states = self.agent.num_states
        self.gamma = self.agent.gamma
        self.meta_lr = self.agent.meta_lr

        self.kp_A = self.agent.kp_A
        self.ki_A = self.agent.ki_A
        self.kd_A = self.agent.kd_A
        self.kp_B = self.agent.kp_B
        self.ki_B = self.agent.ki_B
        self.kd_B = self.agent.kd_B

        self.alpha = self.agent.alpha
        self.beta = self.agent.beta

    def intermediate_update(self):
        """This is an update during a delay in case some internal variables need to change while V is changing."""
        self.calculate_updated_values(intermediate=True)

    def calculate_updated_values(self, intermediate=False):
        raise NotImplementedError

    def update_gains(self):
        self.agent.kp_A = self.kp_A
        self.agent.ki_A = self.ki_A
        self.agent.kd_A = self.kd_A

        self.agent.kp_B = self.kp_B
        self.agent.ki_B = self.ki_B
        self.agent.kd_B = self.kd_B

        #self.agent.alpha = self.alpha
        #self.agent.beta = self.beta

    def plot(self, directory):
        """Plot any relavant information"""
        # raise NotImplementedError
        return None


class NoGainUpdater(AbstractGainUpdater):
    def __init__(self, lambd=0, epsilon=0.1):
        super().__init__(lambd, epsilon)

    def calculate_updated_values(self, intermediate=False):
        pass


class SemiGradientUpdater(AbstractGainUpdater):
    def __init__(self, lambd=0, epsilon=0.1):
        super().__init__(lambd, epsilon)

        self.d_update = 0
        self.i_update = 0
        self.p_update = 0

        self.running_BR = 0
        self.previous_average_BR = float("inf")

    def set_agent(self, agent):
        self.running_BR_A = np.zeros((agent.num_states, agent.num_actions))
        self.running_BR_B = np.zeros((agent.num_states, agent.num_actions))
        self.previous_average_BR = float("inf")
        self.d_update = 0
        self.i_update = 0
        self.p_update = 0

        self.BR_plot = [0]
        self.i_update_plot = [0]
        self.z_plot = [0]

        self.plot_state = 25

        return super().set_agent(agent)

    def calculate_updated_values(self, intermediate=False):
        current_state, action = self.agent.current_state, self.agent.action
        lr = self.agent.lr

        BR = self.agent.previous_BR
        next_BR = self.agent.next_BR

        scale = 0.5
        if self.agent.updated_A:
            self.running_BR_A[current_state][action] = (1 - scale) * self.running_BR_A[current_state][action] + scale * BR * BR
            normalization = self.epsilon + self.running_BR_A[current_state][action]
        else:
            self.running_BR_B[current_state][action] = (1 - scale) * self.running_BR_B[current_state][action] + scale * BR * BR
            normalization = self.epsilon + self.running_BR_B[current_state][action]

        self.p_update += lr * next_BR * self.agent.p_update / normalization
        self.d_update += lr * next_BR * self.agent.d_update / normalization
        self.i_update += lr * next_BR * self.agent.i_update / normalization

        if not intermediate:
            if self.agent.updated_A:
                self.kp_A = 1 + (1 - self.lambd) * (self.kp_A - 1) + self.meta_lr * self.p_update / self.agent.update_frequency
                self.kd_A = self.kd_A * (1 - self.lambd) + self.meta_lr * self.d_update / self.agent.update_frequency
                self.ki_A = self.ki_A * (1 - self.lambd) + self.meta_lr * self.i_update / self.agent.update_frequency
            else:
                self.kp_B = 1 + (1 - self.lambd) * (self.kp_B - 1) + self.meta_lr * self.p_update / self.agent.update_frequency
                self.kd_B = self.kd_B * (1 - self.lambd) + self.meta_lr * self.d_update / self.agent.update_frequency
                self.ki_B = self.ki_B * (1 - self.lambd) + self.meta_lr * self.i_update / self.agent.update_frequency

            self.d_update = 0
            self.i_update = 0
            self.p_update = 0

    def plot(self, directory):
        return
        # Plot BR_plot and i_update_plot on separate sub-plots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        ax1.plot(self.BR_plot, color="red", label="BR")
        ax2.plot(self.i_update_plot, color="blue", label="i_update")
        ax3.plot(self.z_plot, color="green", label="z")
        # Add the legend
        ax1.legend()
        ax2.legend()
        ax3.legend()
        # Add the title
        ax1.set_title("BR")
        ax2.set_title("i_update")
        ax3.set_title("z")
        # Save the figure
        fig.savefig("BR_plot.png")
        plt.show()


class DiagonalSemiGradient(AbstractGainUpdater):
    def __init__(self, lambd=0, epsilon=0.1):
        super().__init__(lambd, epsilon)

    def set_agent(self, agent):
        super().set_agent(agent)
        self.previous_BRs = np.zeros((agent.num_states, agent.num_actions))
        self.running_BR = np.zeros((agent.num_states, agent.num_actions))
        self.p_update = np.zeros((agent.num_states, agent.num_actions))
        self.i_update = np.zeros((agent.num_states, agent.num_actions))
        self.d_update = np.zeros((agent.num_states, agent.num_actions))
        self.alpha_update = np.zeros((agent.num_states, agent.num_actions))
        self.beta_update = np.zeros((agent.num_states, agent.num_actions))

        self.update_frequency = self.agent.update_frequency
        self.frequencies = np.zeros((agent.num_states, agent.num_actions))

        self.plot_state = 25
        self.BR_plot = [0]
        self.kp_plot = [1]
        self.kd_plot = [0]
        self.ki_plot = [0]
        self.d_update_plot = [0]
        self.i_update_plot = [0]


    def calculate_updated_values(self, intermediate=False):
        current_state, action = self.agent.current_state, self.agent.action
        lr = self.agent.lr

        BR = self.agent.previous_BR
        next_BR = self.agent.BR()

        scale = 1 / self.agent.num_steps
        self.running_BR[current_state][action] = (1 - scale) * self.running_BR[current_state][action] + scale * BR * BR
        normalization = self.epsilon + self.running_BR[current_state][action]

        self.p_update[current_state][action] += lr * next_BR * self.agent.p_update / normalization
        self.d_update[current_state][action] += lr * next_BR * self.agent.d_update / normalization
        self.i_update[current_state][action] += lr * next_BR * self.agent.i_update / normalization  
        #self.alpha_update[current_state][action] += lr * self.ki[current_state][action] * BR * BR_previous / normalization
        #self.beta_update[current_state][action] += lr * self.ki[current_state][action] * BR * z[current_state][action] / normalization


        self.frequencies[current_state][action] += 1
        if self.frequencies[current_state][action] == self.update_frequency:
            self.kp[current_state][action] = 1 + (1 - self.lambd) * (self.kp[current_state][action] - 1)
            self.kd[current_state][action] *= 1 - self.lambd
            self.ki[current_state][action] *= 1 - self.lambd
            #self.alpha[current_state] *= 1 - self.lambd
            #self.beta[current_state] = 1 + (1 - self.lambd) * (self.beta[current_state] - 1)

            self.kp[current_state][action] += self.meta_lr * self.p_update[current_state][action] / self.update_frequency
            self.kd[current_state][action] += self.meta_lr * self.d_update[current_state][action] / self.update_frequency
            self.ki[current_state][action] += self.meta_lr * self.i_update[current_state][action] / self.update_frequency
            #self.alpha[current_state] += self.meta_lr * self.alpha_update[current_state] / self.update_frequency
            #self.beta[current_state] += self.meta_lr * self.beta_update[current_state] / self.update_frequency

            self.p_update[current_state][action] = 0
            self.d_update[current_state][action] = 0
            self.i_update[current_state][action] = 0
            self.alpha_update[current_state][action] = 0
            self.beta_update[current_state][action] = 0

            self.frequencies[current_state][action] = 0


        if current_state == 0 and action == 1:
            self.BR_plot.append(BR)
            self.kp_plot.append(self.kp[current_state][action])
            self.kd_plot.append(self.kd[current_state][action])
            self.ki_plot.append(self.ki[current_state][action])
            self.d_update_plot.append(self.d_update[current_state][action])
            self.i_update_plot.append(self.i_update[current_state][action])

    def plot(self, directory):
        #Create a subplot for each BR_plot, kp_plot kd_plot, ki_plot, d_update_plot, i_update_plot
        # They should be in a 3 x 2 grid, and labelled properly, ideally all in squares

        fig, axs = plt.subplots(3, 2)
        fig.suptitle('Diagonal Semi-Gradient Updater')

        axs[0, 0].plot(self.BR_plot, label="BR")
        axs[0, 0].legend()
        axs[0, 1].plot(self.kp_plot, label="kp")
        axs[0, 1].legend()
        axs[1, 0].plot(self.kd_plot, label="kd")
        axs[1, 0].legend()
        axs[1, 1].plot(self.ki_plot, label="ki")
        axs[1, 1].legend()
        axs[2, 0].plot(self.d_update_plot, label="d_update")
        axs[2, 0].legend()
        axs[2, 1].plot(self.i_update_plot, label="i_update")
        axs[2, 1].legend()
        # Add more space between the plots
        plt.subplots_adjust(hspace=0.5)  

        # Make the figure larger
        fig.set_size_inches(10, 10)

        # Save the figure before showing it
        plt.savefig(f"{directory}/semi_gradient_updater.png")

        plt.show()

        # Create a new plot just with the BR_plot, with a different name
        fig, axs = plt.subplots(1, 1)

        # Title it
        fig.suptitle('Chain Walk BR on State 25')

        # Set the axes
        axs.set_xlabel("Iterations")
        axs.set_ylabel("Bellman Residual")

        axs.plot(self.BR_plot, label="BR")
        # Plot and save this
        plt.savefig(f"{directory}/semi_gradient_updater_BR.png")
        plt.show()
