from TabularPID.Agents.Agents import Agent

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
from torchviz import make_dot

from gym.wrappers import RecordVideo

from warnings import filterwarnings
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool` is a deprecated alias')

"""
TODOs:
    - Implement seeds
"""

Sample = namedtuple('Sample', ('state', 'action', 'next_state', 'reward'))

class DQN(nn.Module):
    """
    A DQN
    """
    def __init__(self, num_features, num_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(num_features, 16)
        self.layer2 = nn.Linear(16, 16)
        self.layer3 = nn.Linear(16, num_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class PID_DQN():
    """
    An agent that uses PID to learn the optimal policy with DQN
    """
    def __init__(self, kp, ki, kd, alpha, beta, environment, gamma, optimizer="Adam",
                 replay_memory_size=10000, batch_size=128, learning_rate=0.001,
                 tau=0.005, epsilon=0.1, epsilon_decay=0.999,
                 epsilon_min=0.01, epsilon_decay_step=1000, train_steps=1000,
                 device="cuda"):
        """
        Initialize the agent:

        environment: a gym environment

        Replay memory size: the size of the replay memory
        Batch size: the size of the batch to sample from the replay memory
        Learning rate: the learning rate for the optimizer
        Tau: the tau for soft updates
        Epsilon: the probability of choosing a random action
        Epsilon decay: the decay of epsilon
        Epsilon min: the minimum epsilon
        Epsilon decay step: the number of steps to decay epsilon
        Train steps: the number of iterations to train the DQN after each sample
        """
        self.env = environment
        self.device = device

        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Get num_features from the envirnoment
        self.num_features = self.env.observation_space.shape[0]

        # Get num_actions from the environment
        self.num_actions = self.env.action_space.n

        # Build the loss function
        self.loss_function = nn.MSELoss()

        # Build the replay memory
        self.replay_memory = ReplayMemory(replay_memory_size, batch_size, self.device)

        self.reset(True)

        # Build the optimizer
        if optimizer == "Adam":
            self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate, amsgrad=True)
        elif optimizer == "RMSprop":
            self.optimizer = optim.RMSprop(self.policy_net.parameters())
        elif optimizer == "SGD":
            self.optimizer = optim.SGD(self.policy_net.parameters(), lr=learning_rate)
        else:
            raise ValueError("Optimizer must be Adam, RMSprop, or SGD")

        # Build the tau
        self.tau = tau

        # Build the epsilon
        self.epsilon = epsilon

        # Build the epsilon decay
        self.epsilon_decay = epsilon_decay

        # Build the epsilon minimum
        self.epsilon_min = epsilon_min

        # Build the epsilon decay step
        self.epsilon_decay_step = epsilon_decay_step

        # Build the number of steps to train
        self.train_steps = train_steps

    def reset(self, reset_environment):
        if reset_environment:
            self.env.reset()
        
        # Reset the replay memory
        self.replay_memory.reset()
        # Reset the policy net
        self.policy_net = DQN(self.num_features, self.num_actions).to(self.device)
        # Reset the test net
        self.test_net = DQN(self.num_features, 1).to(self.device)
        # Reset the target net
        self.target_net = DQN(self.num_features, self.num_actions).to(self.device)
        # Reset the D network
        self.D = DQN(self.num_features, self.num_actions).to(self.device)
        # Set the target net to the policy net
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.D.load_state_dict(self.target_net.state_dict())
        self.average_loss = 0

    def rollout(self, max_num_iterations=1000, max_num_episodes=1000, reset=True, reset_environment=True, use_episodes=True, debug=True):
        """
        Rollout the agent for a number of iterations

        max_num_iterations: the maximum number of iterations to rollout
        reset: whether to reset the agent
        reset_environment: whether to reset the environment
        use_episodes: whether to return episodes or iterations
        """
        if reset:
            self.reset(reset_environment)

        # Recorded rewards
        history = []

        episode_reward = 0
        iteration_count = 0
        episode_count = 0
        k = 0

        current_state = torch.tensor(self.env.state, dtype=torch.float32).to(self.device).unsqueeze(0)

        while episode_count < max_num_episodes and iteration_count < max_num_iterations:
            k += 1
            action, done, next_state, reward = self.take_action(current_state)

            if done:
                if use_episodes:
                    history.append(episode_reward)
                    episode_count += 1
                if debug:
                    if episode_count % 50 == 0:
                        print("Episode: {} | Reward: {} | epsilon: {}".format(episode_count, episode_reward, self.epsilon))
                        print("Average loss: {}".format(self.average_loss))
                episode_reward = 0
                # reset the environment
                self.env.reset()

            # Add the current state, action, next state, and reward to the replay memory
            self.replay_memory.add(current_state, action, next_state, reward)
            self.train()

            # Update the target net
            self.update_target_net()
            self.update_D_net()

            # Update epsilon
            if k % self.epsilon_decay_step == 0:
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if not use_episodes:
                iteration_count += 1
                history.append(reward.item())

            episode_reward += reward.item()
            current_state = next_state
        
        history = np.array(history)
        return history, self.policy_net
    
    def train(self):
        for _ in range(self.train_steps):
            # Sample a batch from the replay memory
            current_states, actions, next_states, rewards = self.replay_memory.sample()

            # Compute the Q values for the current states and actions
            current_Q_values = self.test_net(torch.zeros(self.num_features, device=self.device))
            """
            # Create an empty tensor with the same shape as the next states
            #next_Q_values = torch.zeros(next_states.shape[0], device=self.device)
            #current_Q_target_values = torch.zeros(next_states.shape[0], device=self.device)
            #D_values = torch.zeros(next_states.shape[0], device=self.device)
            with torch.no_grad():
                # Compute the Q values for the next states
                next_Q_values = self.target_net(next_states).max(1)[0]
                current_Q_target_values = self.target_net(current_states).gather(1, actions).squeeze(1)
                D_values = self.D(current_states).gather(1, actions).squeeze(1)

                # Compute the expected Q values
                target = (1 - self.kp) * current_Q_target_values + self.kp * (rewards + self.gamma * next_Q_values) \
                    + self.kd * (current_Q_target_values - D_values) \
                #   + self.ki * zs
            """
            
            # Make target a tensor of ones
            target = torch.ones(1, device=self.device)

            # Compute the loss
            if np.random.rand() < 0.001:
                breakpoint()
            loss_function = torch.nn.MSELoss()
            loss = loss_function(current_Q_values, target)

            self.optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
            self.optimizer.step()

            self.average_loss = loss.item() * 0.01 + self.average_loss * 0.99

    def update_target_net(self):
        with torch.no_grad():
            target_net_state_dict = self.target_net.state_dict()
            policy_net_state_dict = self.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
            self.target_net.load_state_dict(target_net_state_dict)

    def update_D_net(self):
        with torch.no_grad():
            D_net_state_dict = self.D.state_dict()
            target_net_state_dict = self.target_net.state_dict()
            for key in D_net_state_dict:
                D_net_state_dict[key] = target_net_state_dict[key]*self.tau + D_net_state_dict[key]*(1-self.tau)
            self.D.load_state_dict(D_net_state_dict)

    def take_action(self, current_state):
        # Choose an action
        action = self.choose_action(current_state)

        # Take the action
        next_state, reward, done, _, _ = self.env.step(action)

        next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        reward = torch.tensor([reward], dtype=torch.float32, device=self.device)
        action = torch.tensor([[action]], dtype=torch.long, device=self.device)

        return action, done, next_state, reward

    def choose_action(self, state):
        # With probability epsilon, choose a random action
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        # Otherwise, choose the best action
        else:
            return self.choose_best_action(state)

    def choose_best_action(self, state):
        with torch.no_grad():
            return self.policy_net(state).max(1)[1].item()

    def visualize_episode(self, file_name="episode"):
        """Render the environment until the episode is done.

        Args:
            file_name (str, optional): The name of the file. Defaults to "episode".
        """
        # Reset the environment
        self.env.reset()

        env = RecordVideo(self.env, file_name)

        env.reset()
        done = False

        while not done:
            # Take an action
            action = self.choose_best_action(torch.tensor(self.env.state, dtype=torch.float32).unsqueeze(0).to(self.device))
            # Take the action
            _, _, done, _, _ = env.step(action)

        env.close()


class ReplayMemory():
    def __init__(self, replay_memory_size, batch_size, device):
        self.batch_size = batch_size
        self.device = device
        self.replay_memory = deque([], maxlen=replay_memory_size)

    def add(self, current_state, action, next_state, reward):
        self.replay_memory.append((current_state, action, next_state, reward))

    def sample(self):
        if self.batch_size > len(self.replay_memory):
            batch = self.replay_memory
        else:
            batch = random.sample(self.replay_memory, self.batch_size)

        sample = Sample(*zip(*batch))

        current_states = torch.cat(sample.state)
        actions = torch.cat(sample.action)
        rewards = torch.cat(sample.reward)
        next_states = torch.cat(sample.next_state)

        return current_states, actions, next_states, rewards
    
    def reset(self):
        self.replay_memory = []