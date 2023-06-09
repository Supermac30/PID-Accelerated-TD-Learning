from TabularPID.Agents.Agents import Agent

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
from recordclass import recordclass
from torchviz import make_dot
import logging

from gym.wrappers import RecordVideo

from warnings import filterwarnings
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool` is a deprecated alias')

"""
TODOs:
    - Implement seeds
"""

Sample = recordclass('Sample', ('state', 'action', 'next_state', 'reward', 'z'))

class DQN(nn.Module):
    """
    A DQN
    """
    def __init__(self, num_features, num_actions, inner_size=128):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(num_features, inner_size)
        self.layer2 = nn.Linear(inner_size, inner_size)
        self.layer3 = nn.Linear(inner_size, num_actions)

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
                 D_tau=0.5, adapt_gains=False, meta_lr=0.1, inner_size=128,
                 device="cuda"):
        """
        Initialize the agent:

        environment: a gym environment

        Replay memory size: the size of the replay memory
        Batch size: the size of the batch to sample from the replay memory
        Learning rate: the learning rate for the optimizer
        Tau: the tau for soft updates
        D_tau: The update rate for the D controller
        Epsilon: the probability of choosing a random action
        Epsilon decay: the decay of epsilon
        Epsilon min: the minimum epsilon
        Epsilon decay step: the number of steps to decay epsilon
        Train steps: the number of iterations to train the DQN after each sample
        adapt_gains: whether to adapt the gains or not
        meta_lr: the learning rate for the meta optimizer
        inner_size: the size of the hidden layers
        """
        self.env = environment
        self.device = device

        self.inner_size = inner_size

        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        if self.env.observation_space.shape is None:
            self.num_features = len(self.env.observation_space)
        else:
            self.num_features = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n
        self.loss_function = nn.MSELoss()  # nn.SmoothL1Loss()

        self.replay_memory = ReplayMemory(replay_memory_size, batch_size, self.device)
        self.batch_size = batch_size

        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.reset(True)

        self.tau = tau
        self.D_tau = D_tau
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.epsilon_decay_step = epsilon_decay_step
        self.train_steps = train_steps

        self.adapt_gains = adapt_gains
        self.meta_lr = meta_lr

    def reset(self, reset_environment):
        if reset_environment:
            self.env.reset()
        
        self.replay_memory.reset()
        self.policy_net = DQN(self.num_features, self.num_actions, self.inner_size).to(self.device)
        self.target_net = DQN(self.num_features, self.num_actions, self.inner_size).to(self.device)
        self.D = DQN(self.num_features, self.num_actions, self.inner_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.D.load_state_dict(self.target_net.state_dict())
        self.average_loss = 0

        # Build the optimizer
        if self.optimizer_name == "Adam":
            self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.learning_rate, amsgrad=True)
        elif self.optimizer_name == "RMSprop":
            self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == "SGD":
            self.optimizer = optim.SGD(self.policy_net.parameters(), lr=self.learning_rate)
        else:
            raise ValueError("Optimizer must be Adam, RMSprop, or SGD")

    def rollout(self, max_num_iterations=1000, max_num_episodes=1000, reset=True, reset_environment=True, use_episodes=True, debug=True, adapt=False, debug_num_steps=10):
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

        current_state = torch.tensor(self.env.reset()[0], dtype=torch.float32).to(self.device).unsqueeze(0)

        while (use_episodes and episode_count < max_num_episodes) or (not use_episodes and iteration_count < max_num_iterations):
            k += 1
            action, done, next_state, reward = self.take_action(current_state)

            self.replay_memory.add(current_state, action, next_state, reward)
            self.train()

            if self.adapt_gains:
                self.update_gains()

            self.update_target_net()
            self.update_D_net()

            if k % self.epsilon_decay_step == 0:
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if not use_episodes:
                iteration_count += 1
                history.append(reward.item())

            episode_reward += reward.item()
            current_state = next_state

            if done:
                if use_episodes:
                    history.append(episode_reward)
                if debug:
                    if episode_count % debug_num_steps == 0:
                        logging.info("Episode: {} | Reward: {} | epsilon: {}".format(episode_count, episode_reward, self.epsilon))
                        logging.info("Average loss: {}".format(self.average_loss))


                episode_count += 1
                episode_reward = 0
                state, _ = self.env.reset()
                current_state = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)

        history = np.array(history)
        return history, self.policy_net
    
    def train(self):
        if len(self.replay_memory.replay_memory) < self.batch_size:
            return
        for _ in range(self.train_steps):
            # Sample a batch from the replay memory
            samples = self.replay_memory.sample()

            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, samples.next_state)), dtype=torch.bool).to(self.device)
            non_final_next_states = [s for s in samples.next_state if s is not None]
            if non_final_next_states != []:
                non_final_next_states = torch.cat([s for s in samples.next_state if s is not None]).to(self.device)
            else:
                non_final_next_states = torch.zeros(0).to(self.device)
            states = torch.cat(samples.state).to(self.device)
            actions = torch.cat(samples.action).to(self.device)
            rewards = torch.cat(samples.reward).to(self.device)
            zs = torch.cat(samples.z).to(self.device)

            # Compute the Q values for the current states and actions
            current_Q_values = self.policy_net(states).gather(1, actions)

            # Compute the Q values for the next states
            next_Q_values = torch.zeros(self.batch_size).to(self.device)
            with torch.no_grad():
                # Compute the Q values for the next states
                next_Q_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
                current_Q_target_values = self.target_net(states).gather(1, actions).squeeze(1)
                D_values = self.D(states).gather(1, actions).squeeze(1)

                self.BRs = rewards + self.gamma * next_Q_values - current_Q_target_values
                new_zs = self.beta * zs + self.alpha * self.BRs

            self.p_update = rewards + self.gamma * next_Q_values - current_Q_target_values
            self.d_update = current_Q_target_values - D_values
            self.i_update = new_zs

            # Compute the expected Q values
            target = current_Q_target_values + self.kp * self.p_update + self.kd * self.d_update + self.ki * self.i_update

            loss = self.loss_function(current_Q_values, target.unsqueeze(1))

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
            self.optimizer.step()

            self.average_loss = loss.item() * 0.01 + self.average_loss * 0.99

            if self.ki != 0:
                self.update_zs(samples, new_zs)

    def update_zs(self, samples, new_zs):
        """In all of the samples, replace the zs with the new zs"""
        # TODO: Figrue out how to make this more efficient
        for i in range(len(samples)):
            samples.z[i][0] = new_zs[i]

    def update_gains(self):
        """Update the gains"""
        self.running_BRs = 0.5 * self.running_BRs + 0.5 * self.BRs.T @ self.BRs
        self.kp += self.meta_lr * self.BRs.T @ self.p_update / (self.epsilon + self.running_BRs)
        self.ki += self.meta_lr * self.BRs.T @ self.i_update / (self.epsilon + self.running_BRs)
        self.kd += self.meta_lr * self.BRs.T @ self.d_update / (self.epsilon + self.running_BRs)

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
                D_net_state_dict[key] = target_net_state_dict[key]*self.D_tau + D_net_state_dict[key]*(1-self.D_tau)
            self.D.load_state_dict(D_net_state_dict)

    def take_action(self, current_state):
        # Choose an action
        action = self.choose_action(current_state)

        # Take the action
        next_state, reward, done, truncated, _ = self.env.step(action)
        terminated = done or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        reward = torch.tensor([reward], dtype=torch.float32, device=self.device)
        action = torch.tensor([[action]], dtype=torch.long, device=self.device)

        return action, terminated, next_state, reward

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

    def visualize_episode(self, file_name="episode", max_length=10000):
        """Render the environment until the episode is done.

        Args:
            file_name (str, optional): The name of the file. Defaults to "episode".
        """
        # Reset the environment
        self.env.reset()

        env = RecordVideo(self.env, file_name + f"{self.kp},{self.ki},{self.kd}.mp4")

        state = env.reset()[0]
        done = False
        k = 0

        while not done and k < max_length:
            # Take an action
            action = self.choose_best_action(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device))
            # Take the action
            state, reward, done, _, _ = env.step(action)
            k += 1

        env.close()


class ReplayMemory():
    def __init__(self, replay_memory_size, batch_size, device):
        self.batch_size = batch_size
        self.device = device
        self.replay_memory = deque([], maxlen=replay_memory_size)

    def add(self, current_state, action, next_state, reward):
        z = torch.zeros(1, device=self.device)
        self.replay_memory.append((current_state, action, next_state, reward, z))

    def sample(self):
        if self.batch_size > len(self.replay_memory):
            batch = self.replay_memory
        else:
            batch = random.sample(self.replay_memory, self.batch_size)

        return Sample(*zip(*batch))

    def reset(self):
        self.replay_memory = []



