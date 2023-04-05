import numpy as np
import logging


class InvalidAction(Exception):
    """Raised when the agent tries to take an invalid action"""

    def __init__(self, invalid_action):
        super().__init__(f"Agent tried to take invalid action {invalid_action}")


class Environment:
    """An abstract class that represents the finite environment the agent will run in."""
    def __init__(self, num_states, num_actions, start_state, seed=-1):
        """Create the Environment. Every environment must store the number
        of states and actions.

        If the seed is uninitialized (equal to -1), then create a random seed
        and log it.
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.start_state = start_state
        self.current_state = start_state

        if seed == -1:
            # Create a random seed
            self.seed = np.random.randint(1, 1000000)
            # Log it for reproducibility
            logger = logging.getLogger(__name__)
            logger.info(f"Random seed for environment: {self.seed}")
        else:
            self.seed = seed

        self.prg = np.random.default_rng(self.seed)

    def reset(self):
        """Reset the Environment back to the initial state.
        Reset the prg to ensure fair experiments.
        Of course, by the MDP property, this is equivalent to starting
        from a blank slate.
        """
        self.current_state = self.start_state
        self.prg = np.random.default_rng(self.seed)

    def take_action(self, action):
        """Take action action, updating the current state, and returning a reward
        and the next state.
        If the action is invalid, raise an invalid action error.
        """
        raise NotImplementedError

    def build_reward_matrix(self):
        """Return a vector of dimensions self.num_states by self.num_actions where the
        (i, j)th entry is the expected reward of taking action j in state i
        """
        raise NotImplementedError

    def build_probability_transition_kernel(self):
        """Return a matrix of dimensions self.num_states by self.num_states
        by self.num_actions where the (i, j, k) entry is the probability of
        going from state i to state j when taking action k.
        """
        raise NotImplementedError

    def build_policy_reward_vector(self, policy):
        """Return a vector of dimension self.num_states where the ith
        entry is the expected reward of entering state i when following policy k.
        """
        return np.einsum('ij,ij->i', self.build_reward_matrix(), policy)

    def build_policy_probability_transition_kernel(self, policy):
        """Return a matrix of dimension self.num_states by self.num_states where the (i, j)
        entry is the probability of going from state i to j when following policy k.
        """
        return np.einsum('ijk,ik->ij', self.build_probability_transition_kernel(), policy)


class Garnet(Environment):
    """An implementation of the Garnet found, as described in section H.2

    self.transitions: An n by m by n matrix, where entry (i, j, k)
        is the probability of going to state j from state i after taking action k
    self.rewards: An n dimensional vector, where the ith entry is the
        reward of being in state i.
    """
    def __init__(self, num_states, num_actions, bP, bR, seed=-1):
        super().__init__(num_states, num_actions, 0, seed)

        self.bP = bP
        self.bR = bR

        self.transitions = np.zeros((num_states, num_states, num_actions), dtype=float)
        for i in range(num_states):
            for j in range(num_actions):
                next_states = np.random.choice(num_states, bP, replace=False)
                self.transitions[i, next_states, j] = 1/self.bP

        rewarded_states = self.prg.choice(num_states, bR, replace=False)
        self.rewards = np.zeros((num_states, 1))
        self.rewards[rewarded_states] = 1
        self.rewards *= self.prg.uniform(0, 1, (num_states, 1))

    def take_action(self, action):
        """Take action action, updating the current state,
        and returning a (next_state, reward) pair.

        InvalidActionError is raised if 0 <= a < self.num_states is false.
        """
        if not 0 <= action < self.num_actions:
            raise InvalidAction(action)

        # Find next state and reward
        transition_probs = self.transitions[self.current_state, :, action]
        self.current_state = np.random.choice(len(transition_probs), p=transition_probs)
        reward = self.rewards[self.current_state]

        return self.current_state, reward

    def build_reward_matrix(self):
        """Return a vector of dimensions self.num_states by self.num_actions where the
        (i, j)th entry is the expected reward of taking action j in state i.
        """
        return np.einsum('ijk,il->i', self.transitions, self.rewards).reshape((-1, 1))

    def build_probability_transition_kernel(self):
        """Return a matrix of dimensions self.num_states by self.num_states
        by self.num_actions where the (i, j, k) entry is the probability of
        going from state i to state j when taking action k.
        """
        return self.transitions


class ChainWalk(Environment):
    """An implementation of the Chain Walk problem as described in
    appendix H.

    We assume self.num_states > 10
    """
    def __init__(self, num_states, seed=-1):
        super().__init__(num_states, 2, num_states - 1, seed)

    def take_action(self, action):
        """Moves left if action is 0, and right if action is 1,
        and raises an InvalidAction error otherwise.

        Returns the reward.
        """
        shift = -1 if action == 0 else 1
        random_number = self.prg.uniform()
        if random_number < 0.7:
            self.current_state = (self.current_state + shift) % self.num_states
        elif random_number < 0.9:
            self.current_state = (self.current_state - shift) % self.num_states
        # else: Don't move

        reward = 0
        if self.current_state == 10:
            reward = -1
        if self.current_state == self.num_states - 10:
            reward = 1

        return self.current_state, reward

    def build_reward_matrix(self):
        """Return a vector of dimensions self.num_states by self.num_actions where the
        (i, j)th entry is the expected reward of taking action j in state i
        """
        rewards = np.zeros((self.num_states, 2))

        rewards[9, 1] = -0.7
        rewards[9, 0] = -0.2
        rewards[11, 0] = -0.7
        rewards[11, 1] = -0.2
        rewards[10, 0] = -0.1
        rewards[10, 1] = -0.1

        rewards[self.num_states - 11, 1] = 0.7
        rewards[self.num_states - 11, 0] = 0.2
        rewards[self.num_states - 10, 0] = 0.1
        rewards[self.num_states - 10, 1] = 0.1
        rewards[self.num_states - 9, 0] = 0.7
        rewards[self.num_states - 9, 1] = 0.2

        return rewards

    def build_probability_transition_kernel(self):
        """Return a matrix of dimensions self.num_states by self.num_states
        by self.num_actions where the (i, j, k) entry is the probability of
        going from state i to state j when taking action k.
        """
        transitions = np.zeros((self.num_states, self.num_states, 2))
        for i in range(self.num_states):
            transitions[i, i, 0] = 0.1
            transitions[i, i, 1] = 0.1
            transitions[i, (i - 1) % self.num_states, 0] = 0.7
            transitions[i, (i - 1) % self.num_states, 1] = 0.2
            transitions[i, (i + 1) % self.num_states, 0] = 0.2
            transitions[i, (i + 1) % self.num_states, 1] = 0.7

        return transitions