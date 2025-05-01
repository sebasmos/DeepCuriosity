import numpy as np
import torch

class PPOBuffer:
    """Stores trajectories and computes GAE advantages and returns."""

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95, device='cpu'):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.device = device

    def store(self, obs, act, rew, val, logp):
        """Store one timestep of agent-environment interaction."""
        assert self.ptr < self.max_size, "Buffer overflow!"
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """Compute GAE advantages and reward-to-go returns for path segment."""
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = self._discount_cumsum(deltas, self.gamma * self.lam)

        # Rewards-to-go for value target
        self.ret_buf[path_slice] = self._discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def _discount_cumsum(self, x, discount):
        """Efficient calculation of discounted cumulative sums."""
        res = np.zeros_like(x)
        acc = 0
        for t in reversed(range(len(x))):
            acc = x[t] + discount * acc
            res[t] = acc
        return res

    def get(self):
        """Normalize advantages and return data as torch tensors."""
        assert self.ptr == self.max_size, "Buffer not full yet!"
        self.ptr, self.path_start_idx = 0, 0

        # Normalize advantages
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-8)

        return {
            'obs': torch.as_tensor(self.obs_buf, dtype=torch.float32, device=self.device),
            'act': torch.as_tensor(self.act_buf, dtype=torch.float32, device=self.device),
            'ret': torch.as_tensor(self.ret_buf, dtype=torch.float32, device=self.device),
            'adv': torch.as_tensor(self.adv_buf, dtype=torch.float32, device=self.device),
            'logp': torch.as_tensor(self.logp_buf, dtype=torch.float32, device=self.device),
        }