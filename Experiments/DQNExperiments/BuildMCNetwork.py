"""
Given the ((state, action), reward) pairs in models/$env/bufferQValues.npy,
write a network that takes as input states, and outputs the Q value of each action,
trained on the bufferQValues in a supervised learning fashion.
"""

import hydra
import torch as th
import numpy as np
import logging
import gymnasium as gym


def train_network(states, actions, rewards, learning_rate, batch_size, epochs, num_actions):
    """
    Trains a network to predict the Q value of each action given a state.
    """
    # Convert to tensors
    states = th.tensor(states, dtype=th.float32)
    actions = th.tensor(actions, dtype=th.float32)
    rewards = th.tensor(rewards, dtype=th.float32)

    # Split the data into train and test
    train_size = int(0.8 * len(states))
    train_indices = np.random.choice(len(states), train_size, replace=False)
    test_indices = np.array([i for i in range(len(states)) if i not in train_indices])
    states_train, states_test = states[train_indices], states[test_indices]
    actions_train, actions_test = actions[train_indices], actions[test_indices]
    rewards_train, rewards_test = rewards[train_indices], rewards[test_indices]

    # Build the network. This takes in a state and outputs the Q value of each action.
    network = th.nn.Sequential(
        th.nn.Linear(states.shape[1], 64),
        th.nn.ReLU(),
        th.nn.Linear(64, 64),
        th.nn.ReLU(),
        th.nn.Linear(64, num_actions)
    )

    # Build the optimizer
    optimizer = th.optim.Adam(network.parameters(), lr=learning_rate)

    # Train the network
    for epoch in range(epochs):
        # Sample a batch
        indices = np.random.choice(len(states_train), batch_size)
        states_batch = states_train[indices]
        actions_batch = actions_train[indices]
        rewards_batch = rewards_train[indices]

        # Compute the loss. We should only be looking at the difference between teh Q value of the action taken and the reward.
        predictions = network(states_batch)
        loss = th.nn.functional.mse_loss(predictions[th.arange(batch_size), actions_batch.long()], rewards_batch)

        # Backpropagate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the loss
        logging.info(f"Epoch {epoch}: {loss}")

        # Compute the test loss
        predictions = network(states_test)
        loss = th.nn.functional.mse_loss(predictions[th.arange(len(states_test)), actions_test.long()], rewards_test)

        # Print the test loss
        logging.info(f"Test loss: {loss}")

        # If test loss is lower than 0.1, stop training
        if loss < 0.1:
            break

    return network


@hydra.main(version_base=None, config_path='../../config/DQNExperiments', config_name='BuildMCNetwork')
def main(cfg):
    env_name = cfg['env_name']
    env = gym.make(env_name)
    num_actions = env.action_space.n
    
    buffer = np.load(f'models/{env_name}/buffer.npy')
    bufferQValues = np.load(f'models/{env_name}/bufferQValues.npy')
    states, actions, rewards = buffer[:, :-1], buffer[:, -1], bufferQValues

    network = train_network(states, actions, rewards, cfg['learning_rate'], cfg['batch_size'], cfg['epochs'], num_actions)

    th.save(network, f'models/{env_name}/network.pth')


if __name__ == '__main__':
    main()