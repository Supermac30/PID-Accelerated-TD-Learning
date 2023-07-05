import torch as th
import torch.nn as nn


class GainAdapter():
    def __init__(self, meta_lr, epsilon, use_previous_BRs=False, batch_size=32, device="cuda"):
        self.device = device
        self.running_BRs = 0

        self.epsilon = epsilon
        self.meta_lr = meta_lr
        self.use_previous_BRs = use_previous_BRs
        self.batch_size = batch_size

    def get_gains(self, model, states, actions, replay_sample):
        raise NotImplementedError
    
    def adapt_gains(self, model, loss, replay_sample):
        """Update the gains"""
        raise NotImplementedError


class NoGainAdapter(GainAdapter):
    """Doesn't adapt the gains at all."""
    def __init__(self, kp, ki, kd, alpha, beta, meta_lr, epsilon, use_previous_BRs=False, batch_size=32):
        super().__init__(meta_lr, epsilon, use_previous_BRs, batch_size)

        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.alpha = alpha
        self.beta = beta

    def get_gains(self, model, states, actions, replay_sample):
        return th.full((self.batch_size, 1), self.kp, device=self.device), th.full((self.batch_size, 1), self.ki, device=self.device), \
            th.full((self.batch_size, 1), self.kd, device=self.device), th.full((self.batch_size, 1), self.alpha, device=self.device), \
            th.full((self.batch_size, 1), self.beta, device=self.device)
    
    def adapt_gains(self, model, loss, replay_sample):
        """Update the gains"""
        return


class SingleGainAdapter(GainAdapter):
    """The regular gain adaptation algorithm for PID-DQN.
    Maintains a single gain for all states and actions, and updates it using the
    """
    def __init__(self, kp, ki, kd, alpha, beta, meta_lr, epsilon, use_previous_BRs=False, batch_size=32):
        super().__init__(meta_lr, epsilon, use_previous_BRs, batch_size)
        
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.alpha = alpha
        self.beta = beta

    def get_gains(self, model, states, actions, replay_sample):
        return th.full((self.batch_size, 1), self.kp, device=self.device), th.full((self.batch_size, 1), self.ki, device=self.device), \
            th.full((self.batch_size, 1), self.kd, device=self.device), th.full((self.batch_size, 1), self.alpha, device=self.device), \
            th.full((self.batch_size, 1), self.beta, device=self.device)
    
    def adapt_gains(self, model, loss, replay_sample):
        """Update the gains"""
        if self.use_previous_BRs:
            BRs = replay_sample.previous_BRs
        else:
            BRs = model.BRs
        self.running_BRs = 0.5 * self.running_BRs + 0.5 * BRs.T @ BRs
        learning_rate = self.meta_lr * model.learning_rate / self.batch_size

        self.kp += learning_rate * (BRs.T @ model.p_update / (self.epsilon + self.running_BRs)).item()
        self.ki += learning_rate * (BRs.T @ model.i_update / (self.epsilon + self.running_BRs)).item()
        self.kd += learning_rate * (BRs.T @ model.d_update / (self.epsilon + self.running_BRs)).item()


class DiagonalGainAdapter(GainAdapter):
    """The diagonal gain adaptation algorithm for PID-DQN"""
    def __init__(self, kp, ki, kd, alpha, beta, meta_lr, epsilon, use_previous_BRs=False, batch_size=32):
        super().__init__(meta_lr, epsilon, use_previous_BRs, batch_size)

        self.initial_kp = kp
        self.initial_ki = ki
        self.initial_kd = kd
        self.initial_alpha = alpha
        self.initial_beta = beta

    def get_gains(self, model, states, actions, replay_sample):
        return th.from_numpy(replay_sample.kps), th.from_numpy(replay_sample.kis), \
            th.from_numpy(replay_sample.kds), th.from_numpy(replay_sample.alphas), \
            th.from_numpy(replay_sample.betas)
    
    def update_gains(self, model, loss, replay_sample):
        """Update the gains"""
        if self.use_previous_BRs:
            BRs = replay_sample.previous_BRs
        else:
            BRs = model.BRs
        self.running_BRs = 0.5 * self.running_BRs + 0.5 * BRs.T @ BRs
        learning_rate = self.meta_lr * model.learning_rate / self.batch_size

        new_kps = replay_sample.kps + learning_rate * (BRs.T @ model.p_update / (self.epsilon + self.running_BRs))
        new_kis = replay_sample.kis + learning_rate * (BRs.T @ model.i_update / (self.epsilon + self.running_BRs))
        new_kds = replay_sample.kds + learning_rate * (BRs.T @ model.d_update / (self.epsilon + self.running_BRs))

        model.replay_buffer.update(replay_sample.indices, kps=new_kps, kis=new_kis, kds=new_kds)


class GainAdapter(nn.Module):
    def __init__(self, input_dim, range_min, range_max):
        super(GainAdapter, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.range_min = range_min
        self.range_max = range_max
    
    def forward(self, x):
        logits = self.linear(x)
        output = (self.range_max - self.range_min) * th.sigmoid(logits) + self.range_min
        return output


class NetworkGainAdapter(GainAdapter):
    def __init__(self, kp, ki, kd, alpha, beta, meta_lr, epsilon, use_previous_BRs=False, state_dim=4, action_dim=1, batch_size=32):
        super().__init__(kp, ki, kd, alpha, beta, meta_lr, epsilon, use_previous_BRs, batch_size)
        # Create a logistic regression model for each gain that takes in the state and
        # action as input, and outputs the gains in some range (chosen by heuristic).
        self.kp_model = GainAdapter(state_dim + action_dim, 0.5, 1.5)
        self.ki_model = GainAdapter(state_dim + action_dim, -1, 1)
        self.kd_model = GainAdapter(state_dim + action_dim, -1, 1)
        self.alpha_model = GainAdapter(state_dim + action_dim, 0, 1)
        self.beta_model = GainAdapter(state_dim + action_dim, 0, 1)

        self.kp_optimizer = th.optim.Adam(self.kp_model.parameters(), lr=0.001)
        self.ki_optimizer = th.optim.Adam(self.ki_model.parameters(), lr=0.001)
        self.kd_optimizer = th.optim.Adam(self.kd_model.parameters(), lr=0.001)
        self.alpha_optimizer = th.optim.Adam(self.alpha_model.parameters(), lr=0.001)
        self.beta_optimizer = th.optim.Adam(self.beta_model.parameters(), lr=0.001)

    def get_gains(self, model, states, actions, replay_sample):
        """Run a single gradient step to update the gains"""
        kps = self.kp_model(th.cat((states, actions), dim=1))
        kis = self.ki_model(th.cat((states, actions), dim=1))
        kds = self.kd_model(th.cat((states, actions), dim=1))
        alpha = self.alpha_model(th.cat((states, actions), dim=1))
        beta = self.beta_model(th.cat((states, actions), dim=1))
        
        return kps, kis, kds, alpha, beta

    def update_gains(self, model, loss, replay_sample):
        self.kp_optimizer.zero_grad()
        self.ki_optimizer.zero_grad()
        self.kd_optimizer.zero_grad()
        self.alpha_optimizer.zero_grad()
        self.beta_optimizer.zero_grad()
        loss.backward()

        # Update the gains
        self.kp_optimizer.step()
        self.ki_optimizer.step()
        self.kd_optimizer.step()
        self.alpha_optimizer.step()
        self.beta_optimizer.step()