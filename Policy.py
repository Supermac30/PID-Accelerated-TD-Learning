import numpy as np

class Policy():
    def __init__(self, num_actions, num_states, prg):
        self.policy = {}
        self.num_actions = num_actions
        self.num_states = num_states
        self.prg = prg

        self.policy = np.zeros((num_states, num_actions))
    
    def get_action(self, state, epsilon):
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