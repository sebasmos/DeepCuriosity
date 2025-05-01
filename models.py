import torch
import torch.nn as nn
from torch.distributions import Normal


class MLP(nn.Module):
    def __init__(self, sizes, activation=nn.Tanh, output_activation=nn.Identity):
        super().__init__()
        layers = []
        for i in range(len(sizes) - 1):
            act = activation if i < len(sizes) - 2 else output_activation
            layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PPOAgent_Raw(nn.Module):
    def __init__(self, obs_dim, act_dim, actor_sizes, critic_sizes, act_limit=1.0):
        super().__init__()
        self.act_limit = act_limit

        self.actor_body = MLP([obs_dim] + list(actor_sizes), activation=nn.Tanh)
        self.actor_mean = MLP([actor_sizes[-1], act_dim], output_activation=nn.Tanh)
        self.actor_logstd = nn.Parameter(torch.zeros(act_dim))

        self.critic = MLP([obs_dim] + list(critic_sizes) + [1], activation=nn.Tanh)

    def forward(self, obs):
        pi_body_out = self.actor_body(obs)
        mean = self.actor_mean(pi_body_out) * self.act_limit
        log_std = self.actor_logstd.expand_as(mean)
        std = torch.exp(log_std)
        return Normal(mean, std)

    def get_value(self, obs):
        return self.critic(obs).squeeze(-1)

    @torch.no_grad()
    def act(self, obs_tensor):
        dist = self(obs_tensor)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(axis=-1)
        action = torch.clamp(action, -self.act_limit, self.act_limit)
        return action.cpu().numpy(), log_prob.cpu().numpy()


class ICM(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=256):
        super().__init__()
        self.obs_encoder = MLP([obs_dim, hidden_size, hidden_size], activation=nn.ReLU)
        self.forward_model = MLP([hidden_size + act_dim, hidden_size, hidden_size], activation=nn.ReLU)
        self.inverse_model = MLP([hidden_size * 2, hidden_size, act_dim], activation=nn.ReLU)

    def forward(self, obs, next_obs, action):
        obs_encoded = self.obs_encoder(obs)
        next_obs_encoded = self.obs_encoder(next_obs)

        inv_input = torch.cat([obs_encoded, next_obs_encoded], dim=-1)
        pred_action = self.inverse_model(inv_input)

        fwd_input = torch.cat([obs_encoded, action], dim=-1)
        pred_next_obs_encoded = self.forward_model(fwd_input)

        return pred_action, pred_next_obs_encoded, next_obs_encoded

    def compute_intrinsic_reward(self, obs, next_obs, action):
        _, pred_next, target_next = self.forward(obs, next_obs, action)
        intrinsic_reward = 0.5 * ((pred_next - target_next).pow(2)).sum(dim=-1)
        return intrinsic_reward.detach()


class PPOAgent_ICM(PPOAgent_Raw):
    def __init__(self, obs_dim, act_dim, actor_sizes, critic_sizes, act_limit=1.0):
        super().__init__(obs_dim, act_dim, actor_sizes, critic_sizes, act_limit)
        self.icm = ICM(obs_dim, act_dim).to(next(self.parameters()).device)