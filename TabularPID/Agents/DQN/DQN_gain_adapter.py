import torch as th
import torch.nn as nn

class GainAdapter():
    def __init__(self, meta_lr, epsilon, use_previous_BRs=True, meta_lr_p=-1, meta_lr_d=-1, meta_lr_i=-1, lambd=0):
        self.running_BRs = 0
        self.lambd = lambd

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
    
    def apply_weight_decay(self, replay_buffer):
        raise NotImplementedError

    def BR(self, replay_sample, network, action_network=None):
        with th.no_grad():
            if action_network is None:
                action_network = network
            next_q_values = network(replay_sample.next_observations)
            next_actions = next_q_values.argmax(dim=1, keepdim=True)

            next_q_values = action_network(replay_sample.next_observations)
            next_q_values = th.gather(next_q_values, dim=1, index=next_actions).reshape(-1, 1)
            target_q_values = replay_sample.rewards + (1 - replay_sample.dones) * self.model.gamma * next_q_values
            
            current_q_values = network(replay_sample.observations)
            current_q_values = th.gather(current_q_values, dim=1, index=replay_sample.actions.long())

            return target_q_values - current_q_values
    
    def adapt_gains(self, loss, replay_sample):
        """Update the gains"""
        raise NotImplementedError


class NoGainAdapter(GainAdapter):
    """Doesn't adapt the gains at all."""
    def __init__(self, kp, ki, kd, alpha, beta, meta_lr, epsilon, use_previous_BRs=True, meta_lr_p=-1, meta_lr_d=-1, meta_lr_i=-1, lambd=0):
        super().__init__(meta_lr, epsilon, use_previous_BRs, meta_lr_p, meta_lr_d, meta_lr_i, lambd)

        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.alpha = alpha
        self.beta = beta

        self.adapts_single_gains = True

    def apply_weight_decay(self, replay_buffer):
        return

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
    def __init__(self, kp, ki, kd, alpha, beta, meta_lr, epsilon, use_previous_BRs=True, meta_lr_p=-1, meta_lr_d=-1, meta_lr_i=-1, lambd=0):
        super().__init__(meta_lr, epsilon, use_previous_BRs, meta_lr_p, meta_lr_d, meta_lr_i, lambd)
        
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.alpha = alpha
        self.beta = beta

        self.adapts_single_gains = True

    def apply_weight_decay(self, replay_buffer):
        replay_buffer.apply_weight_decay(self.lambd)

    def get_gains(self, states, actions, replay_sample):
        return th.full((self.batch_size, 1), self.kp, device=self.device).float(), th.full((self.batch_size, 1), self.ki, device=self.device).float(), \
            th.full((self.batch_size, 1), self.kd, device=self.device).float(), th.full((self.batch_size, 1), self.alpha, device=self.device).float(), \
            th.full((self.batch_size, 1), self.beta, device=self.device).float()

    def adapt_gains(self, replay_sample):
        """Update the gains"""
        # The target network plays the role of the previous Q,
        # The current network plays the role of the next Q
        self.model.policy.set_training_mode(False)
        with th.no_grad():
            if self.model.is_double:
                next_BRs = self.BR(replay_sample, self.model.q_net, self.model.q_net_target)
                previous_BRs = self.BR(replay_sample, self.model.q_net_target, self.model.q_net)
            else:
                next_BRs = self.BR(replay_sample, self.model.q_net)
                previous_BRs = self.BR(replay_sample, self.model.q_net_target)

            zs = self.beta * replay_sample.zs + self.alpha * previous_BRs
            Q = self.model.q_net_target(replay_sample.observations)
            Q = th.gather(Q, dim=1, index=replay_sample.actions.long())
            Qp = self.model.d_net(replay_sample.observations)
            Qp = th.gather(Qp, dim=1, index=replay_sample.actions.long())

            normalization = self.epsilon + th.sqrt(next_BRs.T @ next_BRs)

        self.kp += (self.meta_lr_p / self.batch_size) * (next_BRs.T @ previous_BRs / normalization).item()
        self.ki += (self.meta_lr_i / self.batch_size) * (next_BRs.T @ zs / normalization).item()
        self.kd += (self.meta_lr_d / self.batch_size) * (next_BRs.T @ (Q - Qp) / normalization).item()

        self.model.replay_buffer.update(replay_sample.indices, zs=zs)


class DiagonalGainAdapter(GainAdapter):
    """The diagonal gain adaptation algorithm for PID-DQN"""
    def __init__(self, kp, ki, kd, alpha, beta, meta_lr, epsilon, use_previous_BRs=True, meta_lr_p=-1, meta_lr_d=-1, meta_lr_i=-1, lambd=0):
        super().__init__(meta_lr, epsilon, use_previous_BRs, meta_lr_p, meta_lr_d, meta_lr_i, lambd)

        self.initial_kp = kp
        self.initial_ki = ki
        self.initial_kd = kd
        self.alpha = alpha
        self.beta = beta

    # def apply_weight_decay(self, replay_buffer):
    #     replay_buffer.apply_weight_decay(self.lambd)

    def get_gains(self, states, actions, replay_sample):
        return replay_sample.kp, replay_sample.ki, replay_sample.kd, \
            th.full((self.batch_size, 1), self.alpha, device=self.device).float(), \
            th.full((self.batch_size, 1), self.beta, device=self.device).float()
    
    def adapt_gains(self, replay_sample):
        self.model.policy.set_training_mode(False)
        with th.no_grad():
            if self.model.is_double:
                next_BRs = self.BR(replay_sample, self.model.q_net, self.model.q_net_target)
                previous_BRs = self.BR(replay_sample, self.model.q_net_target, self.model.q_net)
            else:
                next_BRs = self.BR(replay_sample, self.model.q_net)
                previous_BRs = self.BR(replay_sample, self.model.q_net)

            # scale = 1
            # zero_mask = replay_sample.BRs == 0
            # previous_BRs_squared = previous_BRs * previous_BRs
            # running_BRs = th.zeros_like(previous_BRs_squared)
            # running_BRs[zero_mask] = previous_BRs_squared[zero_mask]
            # running_BRs[~zero_mask] = (1 - scale) * replay_sample.BRs[~zero_mask] + scale * previous_BRs_squared[~zero_mask]
            running_BRs = previous_BRs * previous_BRs

            normalization = self.epsilon + running_BRs

            zs = self.beta * replay_sample.zs + self.alpha * previous_BRs
            Q = self.model.q_net_target(replay_sample.observations)
            Q = th.gather(Q, dim=1, index=replay_sample.actions.long())

            if self.model.tabular_d:
                ds = replay_sample.ds
                new_ds = (1 - self.model.d_tau) * ds + self.model.d_tau * Q
            else:
                Qp = self.model.d_net(replay_sample.observations)
                Qp = th.gather(Qp, dim=1, index=replay_sample.actions.long())
                ds = Qp

        # # If any next_BR is bigger than 1:
        # if th.any(next_BRs > 1):
        #     # Find the index
        #     index = th.argmax(next_BRs > 1)
        #     breakpoint()

        new_kps = 1 + (replay_sample.kp - 1) * (1 - self.lambd) + self.meta_lr_p * (next_BRs * previous_BRs / normalization).reshape(-1, 1)
        new_kis = replay_sample.ki * (1 - self.lambd) + self.meta_lr_i * (next_BRs * zs / normalization).reshape(-1, 1)
        new_kds = replay_sample.kd * (1 - self.lambd) + self.meta_lr_d * (next_BRs * (Q - ds) / normalization).reshape(-1, 1)

        if self.model.tabular_d:
            self.model.replay_buffer.update(replay_sample.indices, zs=zs, kp=new_kps, ki=new_kis, kd=new_kds, ds=new_ds)
        else:
            self.model.replay_buffer.update(replay_sample.indices, zs=zs, kp=new_kps, ki=new_kis, kd=new_kds)


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
        Update the gains. Assumes get_gains was called before.
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
