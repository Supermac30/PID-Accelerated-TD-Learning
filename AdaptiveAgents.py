import numpy as np
from Agents import Agent
from MDP import PolicyEvaluation

class AdaptiveAgent(Agent):
    def __init__(self, learning_rates, meta_lr, environment, policy, gamma, sample_size, transition=None, rewards=None, planning=False):
        """
        learning_rates: A triple of three learning rates, one for each component PID
        meta_lr: The meta learning rate
        transition: The transition probability matrix. If this is None, we will approximate gradient terms instead.
        """
        super().__init__(environment, policy, gamma)

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

        self.epsilon = 1e-8  # For numerical stability during normalization
        self.sample_size = sample_size
        self.replay_buffer = [[] for _ in range(self.num_states)]

        self.kp, self.ki, self.kd = 1, 0, 0
        self.alpha, self.beta = 0.95, 0.05

        self.running_BRs = 0  # Used to estimate the partials with respect to beta and kappa_I
        self.planning = planning

    def estimate_value_function(self, num_iterations=1000, test_function=None, initial_V=None):
        self.environment.reset()
        # V is the current value function, Vp is the previous value function
        # Vp stores the previous value of the x state when it was last changed
        if initial_V is None:
            V = np.zeros((self.num_states, 1))
            Vp = np.zeros((self.num_states, 1))
            Vpp = np.zeros((self.num_states, 1))
        else:
            V = initial_V.copy()
            Vp = initial_V.copy()
            Vpp = initial_V.copy()

        z = np.zeros((self.num_states, 1))

        frequencies = np.zeros((self.num_states))

        history = np.zeros((num_iterations))
        gain_history = np.zeros((num_iterations, 5))

        previous_state, current_state, next_state = 0, 0, 0
        previous_reward, reward = 0, 0

        for k in range(num_iterations):
            previous_state, current_state, previous_reward = current_state, next_state, reward
            _, next_state, reward = self.take_action()

            self.replay_buffer[current_state].append((previous_reward, reward, previous_state, next_state))

            frequencies[current_state] += 1
            lr = self.learning_rate(frequencies[current_state])
            update_D_rate = self.update_D_rate(frequencies[current_state])
            update_I_rate = self.update_I_rate(frequencies[current_state])

            self.update_gradients(V, Vp, Vpp, lr)

            # P Component:
            BR = reward + self.gamma * V[next_state] - V[current_state]
            # I Component:
            average = self.beta * z[current_state] + self.alpha * BR
            z = (1 - update_I_rate) * z + update_I_rate * (self.beta * z + self.alpha * BR)
            # D Component:
            difference = V[current_state] - Vp[current_state]

            # A soft update of the value functions
            if self.planning:
                Vpp, Vp, V = Vp, V, V + self.kp * (self.policy_evaluator.bellman_operator(V) - V) + self.kd * (V - Vp)
            else:
                update = V[current_state] + self.kp * BR + self.ki * average + self.kd * difference
                Vpp[current_state] = Vp[current_state]
                Vp[current_state] = (1 - update_D_rate) * Vp[current_state] + update_D_rate * V[current_state]
                V[current_state] = (1 - lr) * V[current_state] + lr * update


            # Keep a record
            gain_history[k][0] = self.kp
            gain_history[k][1] = self.ki
            gain_history[k][2] = self.kd
            gain_history[k][3] = self.alpha
            gain_history[k][4] = self.beta

            if test_function is not None:
                history[k] = test_function(V, Vp, BR)

        if test_function is not None:
            return V, history, gain_history
        return V, gain_history


    def update_gradients(self, V, Vp, Vpp, lr):
        """Find the gradient terms to update the controller gains
        Vp: The previous values of each component before being put in V
        Vpp: The previous values of each component before being put in Vp
        """
        if self.transition is None:
            partial_br_kp, partial_br_kd, BR_current, BR_previous = self.approximate_gradient_terms(V, Vp, Vpp)
        else:
            partial_br_kp, partial_br_kd, BR_current, BR_previous = self.find_exact_gradient_terms(V, Vp, Vpp)

        normalizer = self.epsilon + np.linalg.norm(BR_previous, 2) ** 2

        kp_grad = (BR_current.T @ partial_br_kp) / normalizer
        kd_grad = (BR_current.T @ partial_br_kd) / normalizer

        self.alpha -= self.meta_lr * self.ki * kp_grad

        self.beta, self.running_BRs = \
            self.beta - self.meta_lr * self.ki * self.running_BRs, \
            self.beta * self.running_BRs + self.beta

        self.kp -= lr * self.meta_lr * kp_grad
        self.ki -= lr * self.meta_lr * (self.alpha * self.running_BRs)
        self.kd -= lr * self.meta_lr * kd_grad


    def find_exact_gradient_terms(self, V, Vp, Vpp):
        """Find the gradient terms to update the controller gains
        when we have access to the transition probabilities.
        This is used for testing.

        Vp: The previous values of each component before being put in V
        Vpp: The previous values of each component before being put in Vp

        Return estimates for frac{partial BR(V_k)}{partial kappa_p},
                             frac{partial BR(V_k)}{partial kappa_d},
                             BR(V_k),
                             BR(V_{k - 1})
        in that order.
        """
        BR_current = self.policy_evaluator.bellman_operator(V) - V
        BR_previous = self.policy_evaluator.bellman_operator(Vp) - Vp

        partial_br_kp = (self.gamma * self.transition @ BR_previous) - BR_previous
        partial_br_kd = (self.gamma * self.transition @ (Vp - Vpp)) - (Vp - Vpp)

        return partial_br_kp, partial_br_kd, BR_current, BR_previous


    def approximate_gradient_terms(self, V, Vp, Vpp):
        """Find the gradient terms to update the controller gains
        Vp: The previous values of each component before being put in V
        Vpp: The previous values of each component before being put in Vp

        Return estimates for frac{partial BR(V_k)}{partial kappa_p},
                             frac{partial BR(V_k)}{partial kappa_d},
                             BR(V_k),
                             BR(V_{k - 1})
        in that order.
        """
        # \frac{\partial \text{BR}(V_k)}{\partial \kappa_p}
        partial_br_kp = np.zeros((self.num_states, 1))

        # \frac{\partial \text{BR}(V_k)}{\partial \kappa_d}
        partial_br_kd = np.zeros((self.num_states, 1))

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
                partial_br_kd[state] += Vpp[previous_state] - Vp[previous_state] \
                            - self.gamma * Vpp[state] + self.gamma * Vp[state]

            for _ in range(self.sample_size):
                previous_reward, reward, previous_state, next_state = \
                    self.replay_buffer[state][np.random.randint(0, len(self.replay_buffer[state]))]
                BR_current[state] += reward + self.gamma * V[next_state] - V[state]

            for _ in range(self.sample_size):
                previous_reward, reward, previous_state, next_state = \
                    self.replay_buffer[state][np.random.randint(0, len(self.replay_buffer[state]))]
                BR_previous[state] += reward + self.gamma * Vp[next_state] - Vp[state]

            partial_br_kp[state] /= self.sample_size
            partial_br_kd[state] /= self.sample_size
            BR_current[state] /= self.sample_size
            BR_previous[state] /= self.sample_size


        return partial_br_kp, partial_br_kd, BR_current, BR_previous