"""
Given the ((state, action), reward) pairs in models/$env/bufferQValues.npy,
write a network that takes as input states, and outputs the Q value of each action,
trained on the bufferQValues in a supervised learning fashion.
"""

import hydra
import torch as th
import numpy as np
import logging


def train_network(states, actions, rewards, learning_rate, batch_size, epochs):
    """
    Trains a network to predict the Q value of each action given a state.
    """
    # Convert to tensors
    states = th.tensor(states, dtype=th.float32)
    actions = th.tensor(actions, dtype=th.float32)
    rewards = th.tensor(rewards, dtype=th.float32)

    # Build the network
    network = th.nn.Sequential(
        th.nn.Linear
    )

    # Build the optimizer
    optimizer = th.optim.Adam(network.parameters(), lr=learning_rate)

    # Train the network
    for epoch in range(epochs):
        # Sample a batch
        indices = np.random.choice(len(states), batch_size)
        states_batch = states[indices]
        actions_batch = actions[indices]
        rewards_batch = rewards[indices]

        # Compute the loss
        predictions = network(states_batch)
        loss = th.nn.functional.mse_loss(predictions, rewards_batch)

        # Backpropagate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the loss
        logging.info(f"Epoch {epoch}: {loss}")

    return network


@hydra.main(config_path='../../config/DQNExperiments', config_name='DQNExperiment')
def main(cfg):
    """
    Main method.
    """
    env_name = cfg['env']
    bufferQValues = np.load(f'models/{env_name}/bufferQValues.npy')
    states, actions, rewards = bufferQValues[:, 0], bufferQValues[:, 1], bufferQValues[:, 2]

    network = train_network(states, actions, rewards, cfg['learning_rate'], cfg['batch_size'], cfg['epochs'])

    th.save(network, f'models/{env_name}/network.pth')


if __name__ == '__main__':
    main()