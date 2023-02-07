import numpy as np


class InvalidAction(Exception):
    """Raised when the agent tries to take an invalid action"""

    def __init__(self, invalid_action):
        super().__init__(f"Agent tried to take invalid action {invalid_action}")


class Environment:
    def take_action(self, action):
        """Take action action, updating the current state, and returning a reward
        and the next state.
        If the action is invalid, raise an invalid action error.
        """
        raise NotImplementedError

    def build_reward_matrix(self):
        """Return a vector of dimensions n by m where the
        (i, j)th entry is the expected reward of taking action j in state i
        """
        raise NotImplementedError

    def build_probability_transition_kernel(self):
        """Return a matrix of dimensions n by n by m where the (i, j, k)
        entry is the probability of going from state i to state j when taking action k.
        """
        raise NotImplementedError

    def build_policy_reward_vector(self, policy):
        """Return a vector of dimensions n where the ith
        entry is the expected reward of entering state i when following policy k.
        """
        return self.build_reward_matrix() @ policy

    def build_policy_probability_transition_kernel(self, policy):
        """Return a matrix of dimensions n by n where the (i, j)
        entry is the probability of going from state i to j when following policy k.
        """
        return self.build_probability_transition_kernel() @ policy


class Garnet(Environment):
    """An implementation of the Garnet found, as described in section H.2

    self.transitions: An n by m by n matrix, where entry (i, j, k)
        is the probability of going to state k from state i after taking action j
    self.rewards: An n dimensional vector, where the ith entry is the
        reward of being in state i.
    """
    def __init__(self, n, m, bP, bR):
        self.n = n
        self.m = m
        self.bP = bP
        self.bR = bR

        self.current_state = 0

        self.transitions = np.zeroes((self.n, self.m, self.n), dtype=int)
        for i in range(self.n):
            for j in range(self.m):
                next_states = np.random.choice(self.n, self.bP, replace=False)
                self.transitions[i, j, next_states] = 1/self.bP

        rewarded_states = np.random.choice(self.n, self.bR, replace=False)
        self.rewards = np.zeroes((self.n, 1))
        self.rewards[rewarded_states] = 1
        self.rewards *= np.random.uniform(0, 1, (self.n, 1))

    def take_action(self, action):
        """Take action action, updating the current state,
        and replacing a state with probability self.tau * self.n,
        and returning a (next_state, reward) pair.

        InvalidActionError is raised if 0 <= a < self.m is false.
        """
        if not 0 <= action < self.m:
            raise InvalidAction(action)

        # Find next state and reward
        random_transition = np.random.randint(0, self.b)

        self.current_state = self.transitions[self.current_state, action, random_transition]
        reward = self.reward[self.current_state]

        return self.current_state, reward

    def build_reward_matrix(self):
        """Return a vector of dimensions n by m where the
        (i, j)th entry is the expected reward of taking action j in state i.
        """
        return self.transitions @ self.rewards

    def build_probability_transition_kernel(self):
        """Return a matrix of dimensions n by n by m where the (i, j, k)
        entry is the probability of going from state i to state j when taking action k.
        """
        return self.transitions


class ChainWalk(Environment):
    """An implementation of the Chain Walk problem as described in
    appendix H.

    self.N: The number of states, we assume self.N > 10
    """
    def __init__(self, N):
        self.N = N
        self.current_state = 0

    def take_action(self, action):
        """Moves left if action is -1, and right if action is 1,
        and raises an InvalidAction error otherwise.

        Returns the next state and reward.
        """
        random_number = np.random.uniform()
        if random_number < 0.7:
            self.current_state = (self.current_state + action) % self.N
        elif random_number < 0.9:
            self.current_state = (self.current_state - action) % self.N
        # else: Don't move

        reward = 0
        if self.current_state == 10:
            reward = -1
        if self.current_state == self.N - 10:
            reward = 1

        return self.current_state, reward

    def build_reward_matrix(self):
        """Return a vector of dimensions n by m where the
        (i, j)th entry is the expected reward of taking action j in state i
        """
        rewards = np.zeroes((self.N, 2))

        rewards[9, 1] = -0.7
        rewards[9, 0] = -0.2
        rewards[11, 0] = -0.7
        rewards[11, 1] = -0.2
        rewards[10, 0] = -0.1
        rewards[10, 1] = -0.1

        rewards[self.N - 1, 1] = 0.7
        rewards[self.N - 1, 0] = 0.2
        rewards[self.N, 0] = 0.1
        rewards[self.N, 1] = 0.1
        rewards[self.N + 1, 0] = 0.7
        rewards[self.N + 1, 1] = 0.2

        return rewards

    def build_probability_transition_kernel(self):
        """Return a matrix of dimensions n by n by m where the (i, j, k)
        entry is the probability of going from state i to state j when taking action k.
        """
        transitions = np.zeroes((self.N, self.N, 2))
        for i in range(self.N):
            transitions[i, i, 0] = 0.1
            transitions[i, i, 1] = 0.1
            transitions[i, (i - 1) % self.N, 0] = 0.7
            transitions[i, (i - 1) % self.N, 1] = 0.2
            transitions[i, (i + 1) % self.N, 0] = 0.2
            transitions[i, (i + 1) % self.N, 1] = 0.7

        return transitions