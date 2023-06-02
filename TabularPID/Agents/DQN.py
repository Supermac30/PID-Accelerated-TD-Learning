from TabularPID.Agents.Agents import Agent

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DQN(nn.Module):
    """
    A DQN
    """
    def __init__(self, num_states, num_actions):
        super().__init__()

        # Build the first layer
        self.fc1 = nn.Linear(num_states, 128)
        # Build the second layer
        self.fc2 = nn.Linear(128, 128)
        # Build the third layer
        self.fc3 = nn.Linear(128, num_actions)

    def forward(self, x):
        # Pass the input through the first layer
        x = F.relu(self.fc1(x))
        # Pass the input through the second layer
        x = F.relu(self.fc2(x))
        # Pass the input through the third layer
        x = self.fc3(x)
        return x

class PID_DQN(Agent):
    """
    An agent that uses PID to learn the optimal policy with DQN
    """
    def __init__(self, environment, policy, gamma, follow_trajectory=True,
                 replay_memory_size=10000, batch_size=128, learning_rate=0.001,
                 target_net_update_steps=100, epsilon=0.1, epsilon_decay=0.999,
                 epsilon_min=0.01, epsilon_decay_step=1000, train_steps=1000):
        """
        Initialize the agent:

        Replay memory size: the size of the replay memory
        Batch size: the size of the batch to sample from the replay memory
        Learning rate: the learning rate for the optimizer
        Target net update steps: the number of steps to update the target net
        Epsilon: the probability of choosing a random action
        Epsilon decay: the decay of epsilon
        Epsilon min: the minimum epsilon
        Epsilon decay step: the number of steps to decay epsilon
        Train steps: the number of iterations to train the DQN after each sample
        """
        super().__init__(environment, policy, gamma, follow_trajectory)

        # Build the DQN
        # Build the policy net
        self.policy_net = DQN(self.num_states, self.num_actions).to(self.device)
        # Build the target net
        self.target_net = DQN(self.num_states, self.num_actions).to(self.device)
        # Set the target net to the policy net
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # Set the target net to evaluation mode
        self.target_net.eval()

        # Build the optimizer
        self.optimizer = optim.RMSprop(self.policy_net.parameters())

        # Build the loss function
        self.loss_function = nn.MSELoss()

        # Build the replay memory as a set
        self.replay_memory = set()

        # Build the replay memory size
        self.replay_memory_size = replay_memory_size

        # Build the batch size
        self.batch_size = batch_size

        # Build the learning rate
        self.learning_rate = learning_rate

        # Build the number of steps to update the target net
        self.target_net_update_steps = target_net_update_steps

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
            self.environment.reset()
        
        # Reset the replay memory
        self.replay_memory = {}
        # Reset the policy net
        self.policy_net = DQN(self.num_states, self.num_actions).to(self.device)
        # Reset the target net
        self.target_net = DQN(self.num_states, self.num_actions).to(self.device)
        # Set the target net to the policy net
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # Set the target net to evaluation mode
        self.target_net.eval()

    def estimate_value_function(self, follow_trajectory=True, num_iterations=1000, reset=True, reset_environment=True, stop_if_diverging=True):
        if reset:
            self.reset(reset_environment)

        # The history of test_function
        history = np.zeros(num_iterations)

        for k in range(num_iterations):
            current_state, action, next_state, reward = self.take_action(follow_trajectory)

            # Add the current state, action, next state, and reward to the replay memory
            self.replay_memory.add((current_state, action, next_state, reward))


            # Sample a batch from the replay memory
            batch = np.random.choice(list(self.replay_memory), self.batch_size)

            # Build the current states
            current_states = torch.tensor([x[0] for x in batch], dtype=torch.float32).to(self.device)
            # Build the actions

            actions = torch.tensor([x[1] for x in batch], dtype=torch.long).to(self.device)
            # Build the next states
            next_states = torch.tensor([x[2] for x in batch], dtype=torch.float32).to(self.device)
            # Build the rewards
            rewards = torch.tensor([x[3] for x in batch], dtype=torch.float32).to(self.device)

            # Compute the Q values for the current states
            current_Q_values = self.policy_net(current_states)
            # Compute the Q values for the next states
            next_Q_values = self.target_net(next_states)
            # Compute the Q values for the current states and actions
            current_Q_values = current_Q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            # Compute the Q values for the next states
            next_Q_values = next_Q_values.max(1)[0].detach()
            # Compute the expected Q values
            expected_Q_values = rewards + self.gamma * next_Q_values

            # Compute the loss
            loss = self.loss_function(current_Q_values, expected_Q_values)

            # Zero the gradients
            self.optimizer.zero_grad()
            # Compute the gradients
            loss.backward()
            # Update the weights
            self.optimizer.step()

            # Update the target net
            if k % self.target_net_update_steps == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            # Update epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            history[k] = reward

        return history, self.policy_net