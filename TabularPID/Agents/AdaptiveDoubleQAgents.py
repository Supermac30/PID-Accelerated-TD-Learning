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
        self.kp, self.ki, self.kd = self.original_kp, self.original_ki, self.original_kd
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
        self.updated_A = True

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

            state = self.current_state
            action = self.action

            if self.updated_A:
                self.previous_Q_A[state][action] = self.Q_A[state][action]
                self.previous_Qp_A[state][action] = self.Qp_A[state][action]
                self.previous_z_A[state][action] = self.z_A[state][action]
            else:
                self.previous_Q_B[state][action] = self.Q_B[state][action]
                self.previous_Qp_B[state][action] = self.Qp_B[state][action]
                self.previous_z_B[state][action] = self.z_B[state][action]

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
                history[k] = test_function(self.Q_A, self.Qp_A, self.BR)
                if stop_if_diverging and history[k] > 1.5 * history[0]:
                    history[k:] = float("inf")
                    break

        logging.info(f"Final gains are: kp: {self.kp}, ki: {self.ki}, kd: {self.kd}, alpha: {self.alpha}, beta: {self.beta}")
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

        #Update the value function using the floats kp, ki, kd
        if np.random.random() < 0.5:
            self.updated_A = True
            z = self.z_A
            Q = self.Q_A
            Qp = self.Qp_A
        else:
            self.updated_A = False
            z = self.z_B
            Q = self.Q_B
            Qp = self.Qp_B

        BR = self.BR(self.updated_A)
        self.previous_BR = BR

        self.p_update = BR
        self.i_update = self.beta * z[state][action] + self.alpha * BR
        self.d_update = Q[state][action] - Qp[state][action]

        new_Q = Q[state][action] + self.kp * self.p_update + self.ki * self.i_update + self.kd * self.d_update
        new_z = self.beta * z[state][action] + self.alpha * BR
        new_Qp = Q[state][action]

        Q[state][action] = (1 - lr) * Q[state][action] + lr * new_Q
        z[state][action] = (1 - update_I_rate) * z[state][action] + update_I_rate * new_z
        Qp[state][action] = (1 - update_D_rate) * Qp[state][action] + update_D_rate * new_Qp

        self.next_BR = self.BR(self.updated_A)

    def BR(self, is_A):
        """Return the empirical bellman residual"""
        if is_A:
            Q = self.Q_A
            action_Q = self.Q_B
        else:
            Q = self.Q_B
            action_Q = self.Q_A

        best_action = np.argmax(Q[self.next_state])
        return self.reward + self.gamma * action_Q[self.next_state][best_action] - Q[self.current_state][self.action]


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

    def find_BR(self, is_previous, is_A):
        """Return the bellman residual"""
        current_state, action, reward, next_state = self.agent.current_state, self.agent.action, self.agent.reward, self.agent.next_state
        if is_A:
            if is_previous:
                Q = self.agent.previous_Q_A
                action_Q = self.agent.previous_Q_B
            else:
                Q = self.agent.Q_A
                action_Q = self.agent.Q_B
        else:
            if is_previous:
                Q = self.agent.previous_Q_B
                action_Q = self.agent.previous_Q_A
            else:
                Q = self.agent.Q_B
                action_Q = self.agent.Q_A

        best_action = np.argmax(Q[next_state])

        return reward + self.gamma * action_Q[next_state][best_action] - Q[current_state][action]

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

        BR = self.find_BR(True, self.agent.updated_A)
        next_BR = self.find_BR(False, self.agent.updated_A)

        scale = 0.5
        if self.agent.updated_A:
            self.running_BR_A[current_state][action] = (1 - scale) * self.running_BR_A[current_state][action] + scale * BR * BR
            normalization = self.epsilon + self.running_BR_A[current_state][action]
            previous_Q = self.agent.Q_A
            previous_Qp = self.agent.previous_Q_A
            previous_z = self.agent.previous_z_A
        else:
            self.running_BR_B[current_state][action] = (1 - scale) * self.running_BR_B[current_state][action] + scale * BR * BR
            normalization = self.epsilon + self.running_BR_B[current_state][action]
            previous_Q = self.agent.Q_B
            previous_Qp = self.agent.previous_Q_B
            previous_z = self.agent.previous_z_B

        self.p_update += next_BR * BR / normalization
        self.d_update += next_BR * (previous_Q[current_state][action] - previous_Qp[current_state][action]) / normalization
        self.i_update += next_BR * (self.beta * previous_z[current_state][action] + self.alpha * BR) / normalization

        if not intermediate:
            self.kp = 1 + (1 - self.lambd) * (self.kp - 1) + self.meta_lr * self.p_update / self.agent.update_frequency
            self.kd = self.kd * (1 - self.lambd) + self.meta_lr * self.d_update / self.agent.update_frequency
            self.ki = self.ki * (1 - self.lambd) + self.meta_lr * self.i_update / self.agent.update_frequency

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
