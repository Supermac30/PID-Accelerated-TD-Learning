import numpy as np

from TabularPID.Agents.Agents import learning_rate_function

class LinearTDQ():
    def __init__(self, env, policy, gamma, basis, kp, ki, kd, alpha, beta, lr_Q, lr_z, lr_Qp, adapt_gains=False, meta_lr=0.1, epsilon=0.001, solved_agent=None):
        """
        Initialize the agent.

        Parameters

        env: gym.Env
            The environment to learn from.
        policy: Policy object
            The policy to use.
        gamma: float
            The discount factor.
        basis: LinearFuncSpace
            The basis functions to use.
        lr_V: float
            The learning rate for the P component.
        lr_z: float
            The learning rate for the I component.
        lr_Vp: float
            The learning rate for the D component.
        solved_agent: <has the query_agent method>
            An agent that can be queried with a state to get the v-value for that state, given the policy we are evaluating.
        """
        self.env = env

        # An easy way of checking if the environment is a gym environment, the final piece of hacky code I promise
        self.is_gym_env = hasattr(env, "step")

        self.policy = policy
        self.gamma = gamma
        self.basis = basis
        self.lr_Q = lr_Q
        self.lr_Qp = lr_Qp
        self.lr_z = lr_z

        self.default_kp = kp
        self.default_ki = ki
        self.default_kd = kd

        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.alpha = alpha
        self.beta = beta

        self.current_state = self.env.reset()
        self.solved_agent = solved_agent

        # Gain adaptation params
        self.adapt_gains = adapt_gains
        self.meta_lr = meta_lr
        self.epsilon = epsilon
    
    def take_action(self):
        action = self.policy.get_action(self.current_state)
        current_state = self.current_state
        if self.is_gym_env:
            next_state, reward, done, _, _ = self.env.step(action.item())
            if done:
                next_state = self.env.reset()[0]
        else:
            next_state, reward = self.env.take_action(action)

        self.current_state = next_state
        self.action = action
        return current_state, next_state, reward, action

    def reset(self, reset_environment=True):
        num_features = self.basis.num_features
        if reset_environment:
            if self.is_gym_env:
                self.current_state = self.env.reset()[0]
            else:
                self.current_state = self.env.reset()

        self.w_Q = np.zeros((num_features, 1))
        self.w_Qp = np.zeros((num_features, 1))
        self.w_z = np.zeros((num_features, 1))

        self.kp = self.default_kp
        self.ki = self.default_ki
        self.kd = self.default_kd

        self.running_BR = 0
        self.num_steps = 0
    
    def set_seed(self, seed):
        """Set the seed for the environment and the agent.
        """
        if self.is_gym_env:
            self.env.seed(seed)
        else:
            self.env.set_seed(seed)
        self.policy.set_seed(seed)

    def estimate_value_function(self, num_iterations, test_function=None, reset_environment=True, stop_if_diverging=True):
        self.reset(reset_environment)

        # The history of the gains
        self.gain_history = [np.zeros(num_iterations // (num_iterations // 100)) for _ in range(5)]
        self.history = np.zeros(num_iterations // (num_iterations // 100))
        index = 0

        for k in range(num_iterations):
            self.num_steps += 1
            current_state, next_state, reward, action = self.take_action()

            # Update the value function using the floats kp, ki, kd
            current_state_value = self.query_agent(current_state, action, component="Q")
            # Loop over all actions to find the next_state_value
            next_state_value = np.max([
                self.query_agent(next_state, next_action, component="Q")
                for next_action in range(self.env.action_space.n)
            ])
            current_state_Qp_value = self.query_agent(current_state, action, component="Qp")
            current_state_z_value = self.query_agent(current_state, action, component="z")

            self.BR = reward + self.gamma * next_state_value - current_state_value
            Q_update = current_state_value + self.kp * self.BR \
                + self.kd * (current_state_value - current_state_Qp_value) \
                + self.ki * (self.beta * current_state_z_value + self.alpha * self.BR)
            Qp_update = current_state_value
            z_update = self.beta * current_state_z_value + self.alpha * self.BR

            lr_Q, lr_Qp, lr_z = self.lr_Q(k), self.lr_Qp(k), self.lr_z(k)
            self.learning_rate = lr_Q

            self.w_Q += lr_Q * (Q_update.item() - current_state_value.item()) * self.basis_value(current_state, action)
            self.w_Qp += lr_Qp * (Qp_update.item() - current_state_Qp_value.item()) * self.basis_value(current_state, action)
            self.w_z += lr_z * (z_update.item() - current_state_z_value.item()) * self.basis_value(current_state, action)

            new_next_state_value = np.max([
                self.query_agent(next_state, next_action, component="Q")
                for next_action in range(self.env.action_space.n)
            ])
            new_current_state_value = self.query_agent(current_state, action, component="Q")
            self.next_BR = reward + self.gamma * new_next_state_value - new_current_state_value

            if self.solved_agent is not None and k % (num_iterations // 100) == 0:
                self.history[index] = self.solved_agent.measure_performance(self.query_agent)
                if self.adapt_gains:
                    self.update_gain_history(index)
                if stop_if_diverging and self.history[index] > 2 * self.history[0]:
                    # If we are too large, stop learning
                    self.history[index:] = float('inf')
                    break

                index += 1

            if self.adapt_gains:
                self.update_gains()

        if self.solved_agent is None:
            return self.w_Q

        self.history = np.array(self.history)

        if self.adapt_gains:
            return self.history, self.gain_history, self.w_Q

        return self.history, self.w_Q

    def update_gains(self):
        """Update the gains kp, ki, and kd.
        """
        scale = 1 / self.num_steps
        self.running_BR = (1 - scale) * self.running_BR + scale * self.BR * self.BR
        normalizer = self.epsilon + self.running_BR

        Q = self.query_agent(self.current_state, self.action, component="Q")
        z = self.query_agent(self.current_state, self.action, component="z")
        Qp = self.query_agent(self.current_state, self.action, component="Qp")

        l2_basis = self.basis_value(self.current_state, self.action).T @ self.basis_value(self.current_state, self.action)

        self.kp += self.learning_rate * self.meta_lr * self.next_BR * self.BR * l2_basis / normalizer
        self.ki += self.learning_rate * self.meta_lr * self.next_BR * (self.beta * z + self.alpha * self.BR) * l2_basis / normalizer
        self.kd += self.learning_rate * self.meta_lr * self.next_BR * (Q - Qp) * l2_basis / normalizer

    def update_gain_history(self, index):
        """Update the gain history.
        """
        self.gain_history[0][index] = self.kp
        self.gain_history[1][index] = self.ki
        self.gain_history[2][index] = self.kd
        self.gain_history[3][index] = self.alpha
        self.gain_history[4][index] = self.beta

    def basis_value(self, state, action):
        # If action isn't already an numpy array, make it one
        if not isinstance(action, np.ndarray):
            action = np.array([action])
        if not isinstance(state, np.ndarray):
            state = np.array([state])
 
        return self.basis.value(np.concatenate((state, action)))

    def query_agent(self, state, action, component="Q"):
        """Query the agent for the value at a state"""
        if component == "Q":
            vector = self.w_Q
        elif component == "Qp":
            vector = self.w_Qp
        elif component == "z":
            vector = self.w_z
        
        return self.basis_value(state, action).T @ vector
    
    def set_learning_rates(self, a, b, c, d, e, f):
        self.lr_V = learning_rate_function(a, b)
        self.lr_z = learning_rate_function(c, d)
        self.lr_Vp = learning_rate_function(e, f)
