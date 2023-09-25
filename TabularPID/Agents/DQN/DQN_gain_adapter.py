import torch as th
import torch.nn as nn

class GainAdapter():
    def __init__(self, meta_lr, epsilon, use_previous_BRs=True, meta_lr_p=-1, meta_lr_d=-1, meta_lr_i=-1):
        self.running_BRs = 0

        self.epsilon = epsilon
        self.meta_lr = meta_lr
        self.use_previous_BRs = use_previous_BRs

        if meta_lr_p == -1:
            self.meta_lr_p = meta_lr
        else:
            self.meta_lr_p = meta_lr_p
        if meta_lr_d == -1:
            self.meta_lr_d = meta_lr
        else:
            self.meta_lr_d = meta_lr_d
        if meta_lr_i == -1:
            self.meta_lr_i = meta_lr
        else:
            self.meta_lr_i = meta_lr_i
    
        self.adapts_single_gains = False
        self.num_steps = 0
    
    def set_model(self, model):
        self.model = model
        self.device = model.device
        self.batch_size = model.batch_size

    def get_gains(self, states, actions, replay_sample):
        raise NotImplementedError

    def BR(self, replay_sample, network):
        with th.no_grad():
            next_q_values = network(replay_sample.next_observations)
            next_q_values, _ = next_q_values.max(dim=1)
            next_q_values = next_q_values.reshape(-1, 1)
            target_q_values = replay_sample.rewards + (1 - replay_sample.dones) * self.model.gamma * next_q_values
            current_q_values = network(replay_sample.observations)
            current_q_values = th.gather(current_q_values, dim=1, index=replay_sample.actions.long())
        
            return target_q_values - current_q_values
    
    def adapt_gains(self, loss, replay_sample):
        """Update the gains"""
        raise NotImplementedError


class NoGainAdapter(GainAdapter):
    """Doesn't adapt the gains at all."""
    def __init__(self, kp, ki, kd, alpha, beta, meta_lr, epsilon, use_previous_BRs=True, meta_lr_p=-1, meta_lr_d=-1, meta_lr_i=-1):
        super().__init__(meta_lr, epsilon, use_previous_BRs, meta_lr_p, meta_lr_d, meta_lr_i)

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
    def __init__(self, kp, ki, kd, alpha, beta, meta_lr, epsilon, use_previous_BRs=True, meta_lr_p=-1, meta_lr_d=-1, meta_lr_i=-1):
        super().__init__(meta_lr, epsilon, use_previous_BRs, meta_lr_p, meta_lr_d, meta_lr_i)
        
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
        # The target network plays the role of the previous Q,
        # The current network plays the role of the next Q
        self.model.policy.set_training_mode(True)
        running_BRs = replay_sample.BRs
        with th.no_grad():
            next_BRs = self.BR(replay_sample, self.model.q_net)
            previous_BRs = self.BR(replay_sample, self.model.q_net_target)
            normalization = self.epsilon + (previous_BRs.T @ previous_BRs) / self.batch_size

            running_BRs = 0.25 * running_BRs + 0.75 * next_BRs

            zs = self.beta * replay_sample.zs + self.alpha * previous_BRs
            Q = self.model.q_net_target(replay_sample.observations)
            Q = th.gather(Q, dim=1, index=replay_sample.actions.long())
            Qp = self.model.d_net(replay_sample.observations)
            Qp = th.gather(Qp, dim=1, index=replay_sample.actions.long())

        self.kp += (self.meta_lr_p / self.batch_size) * (running_BRs.T @ previous_BRs / normalization).item()
        self.ki += (self.meta_lr_i / self.batch_size) * (running_BRs.T @ zs / normalization).item()
        self.kd += (self.meta_lr_d / self.batch_size) * (running_BRs.T @ (Q - Qp) / normalization).item()

        self.model.replay_buffer.update(replay_sample.indices, zs=zs, BRs=running_BRs)


class DiagonalGainAdapter(GainAdapter):
    """The diagonal gain adaptation algorithm for PID-DQN"""
    def __init__(self, kp, ki, kd, alpha, beta, meta_lr, epsilon, use_previous_BRs=True, meta_lr_p=-1, meta_lr_d=-1, meta_lr_i=-1):
        super().__init__(meta_lr, epsilon, use_previous_BRs, meta_lr_p, meta_lr_d, meta_lr_i)

        self.initial_kp = kp
        self.initial_ki = ki
        self.initial_kd = kd
        self.alpha = alpha
        self.beta = beta

    def get_gains(self, states, actions, replay_sample):
        # Replace all gains equal to -1000 with the initial gains
        replay_sample.kp[replay_sample.kp == -1000] = self.initial_kp
        replay_sample.ki[replay_sample.ki == -1000] = self.initial_ki
        replay_sample.kd[replay_sample.kd == -1000] = self.initial_kd
        return replay_sample.kp, replay_sample.ki, replay_sample.kd, \
            th.full((self.batch_size, 1), self.alpha, device=self.device), \
            th.full((self.batch_size, 1), self.beta, device=self.device)
    
    def adapt_gains(self, replay_sample):
        """Update the gains. Only works if get_gains was called before"""
        self.model.policy.set_training_mode(True)
        running_BRs = replay_sample.BRs
        with th.no_grad():
            next_BRs = self.BR(replay_sample, self.model.q_net)
            previous_BRs = self.BR(replay_sample, self.model.q_net_target)
            normalization = self.epsilon + (previous_BRs.T @ previous_BRs) / self.batch_size

            running_BRs = 0.25 * running_BRs + 0.75 * next_BRs

            zs = self.beta * replay_sample.zs + self.alpha * previous_BRs
            Q = self.model.q_net_target(replay_sample.observations)
            Q = th.gather(Q, dim=1, index=replay_sample.actions.long())
            Qp = self.model.d_net(replay_sample.observations)
            Qp = th.gather(Qp, dim=1, index=replay_sample.actions.long())

        new_kps = (self.meta_lr_p / self.batch_size) * (next_BRs * previous_BRs / normalization).reshape(-1, 1)
        new_kis = (self.meta_lr_i / self.batch_size) * (next_BRs * zs / normalization).reshape(-1, 1)
        new_kds = (self.meta_lr_d / self.batch_size) * (next_BRs * (Q - Qp) / normalization).reshape(-1, 1)

        self.model.replay_buffer.update(replay_sample.indices, zs=zs, kp=new_kps, ki=new_kis, kd=new_kds, BRs=running_BRs)


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
    def __init__(self, kp, ki, kd, alpha, beta, meta_lr, epsilon, use_previous_BRs=True, meta_lr_p=-1, meta_lr_d=-1, meta_lr_i=-1):
        super().__init__(meta_lr, epsilon, use_previous_BRs, meta_lr_p, meta_lr_d, meta_lr_i)

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