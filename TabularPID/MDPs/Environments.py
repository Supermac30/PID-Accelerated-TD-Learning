import numpy as np

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

        # A duck typing hack to make code compatible with gymnasium with the least amount of refactoring required
        self.observation_space = type(
            'observation_space', 
            (object,), 
            {
                'shape': (1,),
                'high': np.array([num_states]),
                'low': np.array([0]),
                'n': num_states
            }
        )
        self.action_space = type(
            'action_space',
            (object,),
            {
                'n': num_actions
            }
        )

        self.seed = seed
        self.prg = np.random.default_rng(seed)

    def set_seed(self, seed):
        self.prg = np.random.default_rng(seed)

    def reset(self):
        """Reset the Environment back to the initial state.
        """
        self.current_state = self.start_state
        
        return self.current_state

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
        return np.einsum('ij,ij->i', self.build_reward_matrix(), policy.get_policy())

    def build_policy_probability_transition_kernel(self, policy):
        """Return a matrix of dimension self.num_states by self.num_states where the (i, j)
        entry is the probability of going from state i to j when following policy k.
        """
        return np.einsum('ijk,ik->ij', self.build_probability_transition_kernel(), policy.get_policy())


class ZapMDP(Environment):
    def __init__(self, seed=-1):
        super().__init__(6, 18, 0, seed)
        self.actions = [
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 4),
            (5, 5),
            (0, 4), (4, 0),
            (4, 5), (5, 4),
            (3, 4), (4, 3),
            (2, 3), (3, 2),
            (1, 3), (3, 1),
            (1, 5), (5, 1)
        ]
    
    def take_action(self, action):
        if action > 17 or action < 0:
            raise InvalidAction(action)
        
        start, end = self.actions[action]

        next_state = self.current_state
        if start == self.current_state and self.prg.random() < 0.8:
            next_state = end
        
        if start == self.current_state and end == self.current_state and start != 5:
            reward = 0
        elif action == 10:
            reward = -100
        elif start == self.current_state and end == 5:
            reward = 100
        else:
            reward = -5

        self.current_state = next_state

        return next_state, reward
    
    def build_reward_matrix(self):
        rewards = np.ones((self.num_states, self.num_actions)) * -5
        for i in range(self.num_states):
            rewards[i, i] = 0
            rewards[i, 10] = -100

        rewards[5, 5] = 100
        rewards[4, 8] = 100
        rewards[1, 16] = 100

        return rewards
    
    def build_probability_transition_kernel(self):
        transitions = np.zeros((self.num_states, self.num_states, self.num_actions), dtype=float)
        for a in range(self.num_actions):
            start, end = self.actions[a]
            if start == end:
                transitions[start, end, a] = 1
            else:
                transitions[start, end, a] = 0.8
                transitions[start, start, a] = 0.2
            for s in range(self.num_states):
                if s != start:
                    transitions[s, s, a] = 1
        
        return transitions


class GridWorld(Environment):
    def __init__(self, square_size, seed=-1):
        self.square_size = square_size
        num_states = square_size ** 2
        super().__init__(num_states, 4, (0, 0), seed)
 
        self.teleport_rules = {
            (0, 1): ((4, 1), 10),
            (0, 3): ((2, 3), 5)
        }

    def index(self, i, j):
        if i < 0:
            i = 0
        if i >= self.square_size:
            i = self.square_size - 1
        if j < 0:
            j = 0
        if j >= self.square_size:
            j = self.square_size - 1
        return i * self.square_size + j

    def unindex(self, index_num):
        return (index_num // self.square_size, index_num % self.square_size)

    def take_action(self, action):
        if action > 3 or action < 0:
            raise InvalidAction(action)
        
        if self.current_state in self.teleport_rules:
            next_state, reward = self.teleport_rules[self.current_state]
            self.current_state = next_state
            return next_state, reward
        
        x, y = self.unindex(self.current_state)
        
        if action == 0:
            x -= 1
        if action == 1:
            x += 1
        if action == 2:
            y -= 1
        if action == 3:
            y += 1

        next_state = self.index(x, y)
        self.current_state = next_state
        
        if x < 0 or y < 0 or x >= self.square_size or y >= self.square_size:
            return next_state, -1
        return next_state, 0

    def build_reward_matrix(self):
        rewards = np.zeros((self.num_states, self.num_actions))
        for start in self.teleport_rules:
            _, reward = self.teleport_rules[start]
            rewards[self.index(*start), :] = reward
        
        # Out of bounds penalties
        for i in range(self.square_size):
            # Action order: Up (0) Down (1) Left (2) Right (3)
            rewards[self.index(0, i), 0] = -1
            rewards[self.index(self.square_size - 1, i), 1] = -1
            rewards[self.index(i, 0), 2] = -1
            rewards[self.index(i, self.square_size - 1), 3] = -1

        return rewards

    def build_probability_transition_kernel(self):
        transitions = np.zeros((self.num_states, self.num_states, 4), dtype=float)
        for i in range(self.square_size):
            for j in range(self.square_size):
                if (i, j) in self.teleport_rules: continue
                # Action order: Up (0) Down (1) Left (2) Right (3)
                transitions[self.index(i, j), self.index(i - 1, j), 0] = 1
                transitions[self.index(i, j), self.index(i + 1, j), 1] = 1
                transitions[self.index(i, j), self.index(i, j - 1), 2] = 1
                transitions[self.index(i, j), self.index(i, j + 1), 3] = 1
        
        for start in self.teleport_rules:
            end, _ = self.teleport_rules[start]
            transitions[self.index(*start), self.index(*end), :] = 1
        
        return transitions


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
                next_states = self.prg.choice(num_states, min(bP, num_states), replace=False)
                self.transitions[i, next_states, j] = 1/min(self.bP, num_states)

        rewarded_states = self.prg.choice(num_states, min(bR, num_states), replace=False)
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
        reward = self.rewards[self.current_state]
        transition_probs = self.transitions[self.current_state, :, action]
        self.current_state = self.prg.choice(len(transition_probs), p=transition_probs)

        return self.current_state, reward

    def build_reward_matrix(self):
        """Return a vector of dimensions self.num_states by self.num_actions where the
        (i, j)th entry is the expected reward of taking action j in state i.
        """
        return np.tile(self.rewards, (1, self.num_actions))

    def build_probability_transition_kernel(self):
        """Return a matrix of dimensions self.num_states by self.num_states
        by self.num_actions where the (i, j, k) entry is the probability of
        going from state i to state j when taking action k.
        """
        return self.transitions


class ChainWalk(Environment):
    """An implementation of the Chain Walk problem as described in
    appendix H.
    """
    def __init__(self, num_states, seed=-1, goal_state=40, punish_state=10):
        self.goal_state = goal_state
        self.punish_state = punish_state
        super().__init__(num_states, 2, num_states - 1, seed)

    def take_action(self, action):
        """Moves left if action is 0, and right if action is 1,
        and raises an InvalidAction error otherwise.

        Returns the reward.
        """
        if action == 0:
            shift = -1
        elif action == 1:
            shift = 1
        else:
            raise InvalidAction(action)

        random_number = self.prg.uniform()
        if random_number < 0.7:
            self.current_state = (self.current_state + shift) % self.num_states
        elif random_number < 0.9:
            self.current_state = (self.current_state - shift) % self.num_states
        # else: Don't move

        reward = 0
        if self.current_state == self.goal_state:
            reward = 1
        if self.current_state == self.punish_state:
            reward = -1

        return self.current_state, reward

    def build_reward_matrix(self):
        """Return a matrix of dimensions self.num_states by self.num_actions where the
        (i, j)th entry is the expected reward of taking action j in state i
        """
        rewards = np.zeros((self.num_states, 2))

        rewards[self.punish_state - 1, 1] += -0.7
        rewards[self.punish_state - 1, 0] += -0.2
        rewards[self.punish_state, 0] += -0.1
        rewards[self.punish_state, 1] += -0.1
        rewards[self.punish_state + 1, 0] += -0.7
        rewards[self.punish_state + 1, 1] += -0.2

        rewards[self.goal_state - 1, 1] += 0.7
        rewards[self.goal_state - 1, 0] += 0.2
        rewards[self.goal_state, 0] += 0.1
        rewards[self.goal_state, 1] += 0.1
        rewards[self.goal_state + 1, 0] += 0.7
        rewards[self.goal_state + 1, 1] += 0.2

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


class CliffWalk(Environment):
    """Taken from the OS-Dyna Code"""
    def __init__(self, success_prob, seed):
        self.n_columns = 6
        self.n_rows = 6

        self.terminal_states = [self.n_columns-1]

        for state in range(self.n_columns * self.n_rows):
            if state // self.n_columns in [0,2,4] and state % self.n_columns in range(1, self.n_columns-1):
                self.terminal_states.append(state)

        self.walls = []
        self.success_prob = success_prob

        self.reward_matrix = self.build_reward_matrix()

        super().__init__(self.n_columns * self.n_rows, 4, 0, seed)

    def build_probability_transition_kernel(self):
        n_states = self.n_columns * self.n_rows
        P = np.zeros((n_states, n_states, 4))
        unif_prob = (1 - self.success_prob) / 3
        for r in range(self.n_rows):
            for c in range(self.n_columns):
                state = r * self.n_columns + c
                if state in self.terminal_states:
                    P[state, state, :] = 1
                else:
                    for a in range(4):
                        for dir in range(4):
                            target = self.get_target(state, dir)
                            if dir == a:
                                P[state, target, a] += self.success_prob
                            else:
                                P[state, target, a] += unif_prob

        return P

    def build_reward_matrix(self):
        goal_state = self.n_columns - 1
        n_states = self.n_columns * self.n_rows

        R = np.zeros((n_states, 4))

        for state in range(n_states):
            if state in self.terminal_states:
                if state == goal_state:
                    R[state, :] = 20
                elif state % self.n_columns in range(1, self.n_columns-1):
                    if state // self.n_columns == 0:
                        R[state, :] = -32
                    if state // self.n_columns == 2:
                        R[state, :] = -16
                    if state // self.n_columns == 4:
                        R[state, :] = -8
                    if state // self.n_columns == 6:
                        R[state, :] = -4
                    if state // self.n_columns == 8:
                        R[state, :] = -2
                    if state // self.n_columns == 10:
                        R[state, :] = -1
            else:
                R[state, :] = -1
        return R

    def take_action(self, action):
        if self.prg.random() > self.success_prob:
            action = self.prg.choice([a for a in range(4) if a != action])
        reward = self.reward_matrix[self.current_state, action]
        if self.current_state in self.terminal_states:
            return self.current_state, reward

        target = self.get_target(self.current_state, action)
        self.current_state = target

        return target, reward

    def get_target(self, state, action):
        column = state % self.n_columns
        row = int((state - column) / self.n_columns)

        if action == 0:
            top_c = column
            top_r = max(row - 1, 0)
            target = top_r * self.n_columns + top_c
        elif action == 1:
            right_c = min(column + 1, self.n_columns - 1)
            right_r = row
            target = right_r * self.n_columns + right_c
        elif action == 2:
            bottom_c = column
            bottom_r = min(row + 1, self.n_rows - 1)
            target = bottom_r * self.n_columns + bottom_c
        elif action == 3:
            left_c = max(column - 1, 0)
            left_r = row
            target = left_r * self.n_columns + left_c
        else:
            raise InvalidAction(action)

        return target


class IdentityEnv(Environment):
    def __init__(self, num_states, seed):
        self.reward = 1e10
        super().__init__(num_states, 1, 0, seed)

    def build_probability_transition_kernel(self):
        P = np.zeros((self.num_states, self.num_states, 1))
        for i in range(self.num_states):
            P[i, i, 0] = 1
        return P

    def build_reward_matrix(self):
        return np.full((self.num_states, 1), self.reward)

    def take_action(self, action):
        return self.current_state, self.reward


class NormalApproximation(Environment):
    def __init__(self, variance, seed):
        self.variance = variance
        self.reward = 10
        super().__init__(1, 1, 0, seed)

    def build_probability_transition_kernel(self):
        return np.array([[[1]]])

    def build_reward_matrix(self):
        return np.array([[self.reward]])

    def take_action(self, action):
        # Sample from a normal distribution with variance self.variance and mean self.reward
        return self.current_state, self.prg.normal(self.reward, self.variance)

class BernoulliApproximation(Environment):
    def __init__(self, seed):
        self.reward = 0
        super().__init__(3, 2, 2, seed)

    def build_probability_transition_kernel(self):
        return np.array([[[1]]])

    def build_reward_matrix(self):
        return np.array([[self.reward]])

    def take_action(self, action):
        # Sample from a beta distribution with alpha and beta parameters
        random_number = self.prg.random()
        if random_number < 0.05:
            return self.current_state, 200
        else:
            return self.current_state, -100/9

class ComplexMDP(Environment):
    def __init__(self, seed):
        self.reward = 0
        super().__init__(1, 1, 0, seed)

    def build_probability_transition_kernel(self, policy):
        """
        Return the np matrix:
        0 0 1
        1 0 0
        0 1/2 1/2
        """
        return np.array([[[0, 1/2, 1/2]], [[1/2, 0, 1/2]], [[0, 1/2, 1/2]]])
    
    def build_reward_matrix(self):
        """
        Return the np vector:
        0
        0
        0
        """
        return np.array([[0, -1, 1]])
    
    def take_action(self, action):
        if self.current_state == 2:
            if action == 0:
                self.current_state = 0
                return self.current_state, 0
            else:
                self.current_state = 1
                return self.current_state, 0
        elif self.current_state == 1:
            return self.current_state, -1
        else:
            return self.current_state, 1