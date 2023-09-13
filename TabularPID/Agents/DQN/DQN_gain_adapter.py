import torch as th
import torch.nn as nn


class GainAdapter():
    def __init__(self, meta_lr, epsilon, use_previous_BRs=True):
        self.running_BRs = 0

        self.epsilon = epsilon
        self.meta_lr = meta_lr
        self.use_previous_BRs = use_previous_BRs
    
        self.adapts_single_gains = False
        self.num_steps = 0
    
    def set_model(self, model):
        self.model = model
        self.device = model.device
        self.batch_size = model.batch_size

    def get_gains(self, states, actions, replay_sample):
        raise NotImplementedError
    
    def adapt_gains(self, loss, replay_sample):
        """Update the gains"""
        raise NotImplementedError


class NoGainAdapter(GainAdapter):
    """Doesn't adapt the gains at all."""
    def __init__(self, kp, ki, kd, alpha, beta, meta_lr, epsilon, use_previous_BRs=True):
        super().__init__(meta_lr, epsilon, use_previous_BRs)

        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.alpha = alpha
        self.beta = beta

        self.adapts_single_gains = True

    def get_gains(self, states, actions, replay_sample):
        return th.full((self.batch_size, 1), self.kp, device=self.device), th.full((self.batch_size, 1), self.ki, device=self.device), \
            th.full((self.batch_size, 1), self.kd, device=self.device), th.full((self.batch_size, 1), self.alpha, device=self.device), \
            th.full((self.batch_size, 1), self.beta, device=self.device)
    
    def adapt_gains(self, replay_sample):
        """Update the gains"""
        return


class SingleGainAdapter(GainAdapter):
    """The regular gain adaptation algorithm for PID-DQN.
    Maintains a single gain for all states and actions, and updates it using the
    """
    def __init__(self, kp, ki, kd, alpha, beta, meta_lr, epsilon, use_previous_BRs=True):
        super().__init__(meta_lr, epsilon, use_previous_BRs)
        
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.alpha = alpha
        self.beta = beta

        self.adapts_single_gains = True

    def get_gains(self, states, actions, replay_sample):
        return th.full((self.batch_size, 1), self.kp, device=self.device), th.full((self.batch_size, 1), self.ki, device=self.device), \
            th.full((self.batch_size, 1), self.kd, device=self.device), th.full((self.batch_size, 1), self.alpha, device=self.device), \
            th.full((self.batch_size, 1), self.beta, device=self.device)
    
    def adapt_gains(self, replay_sample):
        """Update the gains"""
        if self.use_previous_BRs:
            BRs = replay_sample.BRs
        else:
            BRs = self.model.BRs
        self.num_steps += 1
        scale = 1 / self.num_steps
        self.running_BRs = (1 - scale) * self.running_BRs + scale * BRs.T @ BRs
        learning_rate = self.meta_lr * self.model.learning_rate / self.batch_size

        self.kp += learning_rate * (BRs.T @ self.model.p_update / (self.epsilon + self.running_BRs)).item()
        self.ki += learning_rate * (BRs.T @ self.model.i_update / (self.epsilon + self.running_BRs)).item()
        self.kd += learning_rate * (BRs.T @ self.model.d_update / (self.epsilon + self.running_BRs)).item()


class DiagonalGainAdapter(GainAdapter):
    """The diagonal gain adaptation algorithm for PID-DQN"""
    def __init__(self, kp, ki, kd, alpha, beta, meta_lr, epsilon, use_previous_BRs=True):
        super().__init__(meta_lr, epsilon, use_previous_BRs)

        self.initial_kp = kp
        self.initial_ki = ki
        self.initial_kd = kd
        self.initial_alpha = alpha
        self.initial_beta = beta

    def get_gains(self, states, actions, replay_sample):
        # Replace all gains equal to -1 with the initial gains
        replay_sample.kp[replay_sample.kp == -1] = self.initial_kp
        replay_sample.ki[replay_sample.ki == -1] = self.initial_ki
        replay_sample.kd[replay_sample.kd == -1] = self.initial_kd
        return replay_sample.kp, replay_sample.ki, replay_sample.kd, \
            th.full((self.batch_size, 1), self.initial_alpha, device=self.device), \
            th.full((self.batch_size, 1), self.initial_beta, device=self.device)
    
    def adapt_gains(self, replay_sample):
        """Update the gains. Only works if get_gains was called before"""
        self.num_steps += 1
        if self.use_previous_BRs:
            BRs = replay_sample.BRs
        else:
            BRs = self.model.BRs
        scale = 1 / self.num_steps
        self.running_BRs = (1 - scale) * self.running_BRs + scale * BRs.T @ BRs
        learning_rate = self.meta_lr * self.model.learning_rate / self.batch_size

        new_kps = replay_sample.kp + learning_rate * (BRs.T @ self.model.p_update / (self.epsilon + self.running_BRs))
        new_kis = replay_sample.ki + learning_rate * (BRs.T @ self.model.i_update / (self.epsilon + self.running_BRs))
        new_kds = replay_sample.kd + learning_rate * (BRs.T @ self.model.d_update / (self.epsilon + self.running_BRs))

        self.model.replay_buffer.update(replay_sample.indices, kp=new_kps, ki=new_kis, kd=new_kds)


class GainAdaptingNetwork(nn.Module):
    def __init__(self, input_dim, range_min, range_max, device):
        super(GainAdaptingNetwork, self).__init__()
        self.linear = nn.Linear(input_dim, 1, device=device)
        self.range_min = range_min
        self.range_max = range_max
    
    def forward(self, x):
        logits = self.linear(x)
        output = (self.range_max - self.range_min) * th.sigmoid(logits) + self.range_min
        return output


class NetworkGainAdapter(GainAdapter):
    def __init__(self, kp, ki, kd, alpha, beta, meta_lr, epsilon, use_previous_BRs=True):
        super().__init__(meta_lr, epsilon, use_previous_BRs)

    def set_model(self, model):
        super().set_model(model)
        state_dim = self.model.env.observation_space.shape[0]

        # Create a logistic regression model for each gain that takes in the state and
        # action as input, and outputs the gains in some range (chosen by heuristic).

        self.kp_model = GainAdaptingNetwork(state_dim + 1, 0.5, 1.5, device=self.device)
        self.ki_model = GainAdaptingNetwork(state_dim + 1, -1, 1, device=self.device)
        self.kd_model = GainAdaptingNetwork(state_dim + 1, -1, 1, device=self.device)
        self.alpha_model = GainAdaptingNetwork(state_dim + 1, 0, 1, device=self.device)
        self.beta_model = GainAdaptingNetwork(state_dim + 1, 0, 1, device=self.device)

        self.kp_optimizer = th.optim.Adam(self.kp_model.parameters(), lr=0.001)
        self.ki_optimizer = th.optim.Adam(self.ki_model.parameters(), lr=0.001)
        self.kd_optimizer = th.optim.Adam(self.kd_model.parameters(), lr=0.001)
        self.alpha_optimizer = th.optim.Adam(self.alpha_model.parameters(), lr=0.001)
        self.beta_optimizer = th.optim.Adam(self.beta_model.parameters(), lr=0.001)

    def get_gains(self, states, actions, replay_sample):
        """Run a single gradient step to update the gains"""
        # Concatenate the states and actions and make them use the same device
        x = th.cat((states, actions), dim=1)
        kps = self.kp_model(x)
        kis = self.ki_model(x)
        kds = self.kd_model(x)
        alphas = self.alpha_model(x)
        betas = self.beta_model(x)

        self.kp_optimizer.zero_grad()
        self.ki_optimizer.zero_grad()
        self.kd_optimizer.zero_grad()
        self.alpha_optimizer.zero_grad()
        self.beta_optimizer.zero_grad()
        
        return kps, kis, kds, alphas, betas

    def adapt_gains(self, replay_sample):
        """
        Update the gains. Only works if get_gains was called before.
        Assumes that loss.backwards was called before.
        """
        # Clip norms
        th.nn.utils.clip_grad_norm_(self.kp_model.parameters(), 10)
        th.nn.utils.clip_grad_norm_(self.ki_model.parameters(), 10)
        th.nn.utils.clip_grad_norm_(self.kd_model.parameters(), 10)
        th.nn.utils.clip_grad_norm_(self.alpha_model.parameters(), 10)
        th.nn.utils.clip_grad_norm_(self.beta_model.parameters(), 10)

        # Update the gains
        self.kp_optimizer.step()
        self.ki_optimizer.step()
        self.kd_optimizer.step()
        self.alpha_optimizer.step()
        self.beta_optimizer.step()