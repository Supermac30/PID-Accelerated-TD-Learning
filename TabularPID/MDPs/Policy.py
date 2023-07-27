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

    def set_policy_from_Q(self, Q):
        """Set the policy from a Q function"""
        self.policy = np.eye(self.num_actions)[np.argmax(Q, axis=1)]

    def get_action(self, state):
        """Get an action from the policy"""
        return self.prg.choice(self.num_actions, p=self.policy[state])

    def get_on_policy_sample(self):
        """Get a uniformly random sample from the policy"""
        state = self.prg.choice(self.num_states)
        action = self.prg.choice(self.num_actions, p=self.policy[state])
        return state, action
    
    def get_random_sample(self):
        """Get a random state and action pair"""
        state = self.prg.choice(self.num_states)
        action = self.prg.choice(self.num_actions)
        return state, action