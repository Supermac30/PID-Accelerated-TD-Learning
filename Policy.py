import numpy as np

class Policy():
    def __init__(self, num_actions, num_states, prg, policy=None):
        """
        num_actions: number of actions
        num_states: number of states
        prg: numpy random number generator
        """
        self.num_actions = num_actions
        self.num_states = num_states
        self.prg = prg

        if policy is None:
            self.policy = np.full((num_states, num_actions), 1/self.num_actions)
        else:
            self.policy = policy

    def set_policy(self, policy):
        """Set the policy"""
        self.policy = policy

    def get_policy(self):
        """Get the policy"""
        return self.policy

    def set_policy_from_Q(self, Q, epsilon):
        """Set the policy from a Q function, setting a probability of epsilon picking an action that isn't optimal"""
        self.policy = np.zeros((self.num_states, self.num_actions))
        self.policy = np.where(
            np.arange(self.num_actions) == np.argmax(Q, axis=1)[:, None],
            1 - epsilon,
            epsilon / (self.num_actions - 1)
        )

    def get_action(self, state, epsilon=0):
        """Get an action from the policy, with a probability of epsilon of choosing a random action"""
        if self.prg.random() < epsilon:
            return self.prg.randint(self.num_actions)

        return self.prg.choice(self.num_actions, p=self.policy[state])

    def get_uniformly_random_sample(self, epsilon):
        """Get a uniformly random sample from the policy, with a probability of epsilon of choosing a random action"""
        if self.prg.random() < epsilon:
            return self.prg.randint(self.num_states), self.prg.randint(self.num_actions)

        state = self.prg.choice(self.num_states)
        action = self.prg.choice(self.num_actions, p=self.policy[state])
        return state, action