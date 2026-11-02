"""Fixed-size circular experience replay with action masks for the next state."""
from __future__ import annotations

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, num_actions: int):
        self.capacity, self.obs_dim, self.num_actions = capacity, obs_dim, num_actions
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)   # only true terminations cut bootstrapping
        self.next_masks = np.ones((capacity, num_actions), dtype=bool)
        self.pos = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(self, obs, action, reward, next_obs, terminated, next_mask=None) -> None:
        i = self.pos
        self.obs[i] = obs
        self.next_obs[i] = next_obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.terminated[i] = float(terminated)
        self.next_masks[i] = True if next_mask is None else next_mask
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict:
        idx = rng.integers(0, self.size, size=batch_size)
        return {
            "obs": self.obs[idx], "actions": self.actions[idx], "rewards": self.rewards[idx],
            "next_obs": self.next_obs[idx], "terminated": self.terminated[idx], "next_masks": self.next_masks[idx],
        }

    # persistence (for resuming training)
    def save(self, path) -> None:
        n = self.size
        np.savez_compressed(path, obs=self.obs[:n], next_obs=self.next_obs[:n], actions=self.actions[:n],
                            rewards=self.rewards[:n], terminated=self.terminated[:n],
                            next_masks=self.next_masks[:n], pos=self.pos, capacity=self.capacity)

    def load(self, path) -> None:
        d = np.load(path)
        n = int(min(len(d["actions"]), self.capacity))
        if d["obs"].shape[1] != self.obs_dim or d["next_masks"].shape[1] != self.num_actions:
            raise ValueError("Replay buffer file does not match this environment's obs/action dimensions")
        self.obs[:n], self.next_obs[:n] = d["obs"][:n], d["next_obs"][:n]
        self.actions[:n], self.rewards[:n] = d["actions"][:n], d["rewards"][:n]
        self.terminated[:n], self.next_masks[:n] = d["terminated"][:n], d["next_masks"][:n]
        self.size = n
        self.pos = int(d["pos"]) % self.capacity if n == self.capacity else n % self.capacity
