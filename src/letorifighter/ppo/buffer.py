"""Rollout storage for a single environment, with generalised advantage estimation."""

import numpy as np


class RolloutBuffer:
    def __init__(self, steps: int, obs_dim: int, num_heads: int = 22) -> None:
        self.steps = steps
        self.obs = np.zeros((steps, obs_dim), np.float32)
        self.task = np.zeros(steps, np.int64)
        self.actions = np.zeros((steps, num_heads), np.int64)
        self.logp = np.zeros(steps, np.float32)
        self.values = np.zeros(steps, np.float32)
        self.rewards = np.zeros(steps, np.float32)
        self.dones = np.zeros(steps, np.float32)  # the episode ended after this step
        self.advantages = np.zeros(steps, np.float32)
        self.returns = np.zeros(steps, np.float32)
        self.pos = 0

    def add(self, obs, task, action, logp, value, reward, done) -> None:
        i = self.pos
        self.obs[i] = obs
        self.task[i] = task
        self.actions[i] = action
        self.logp[i] = logp
        self.values[i] = value
        self.rewards[i] = reward
        self.dones[i] = done
        self.pos += 1

    @property
    def full(self) -> bool:
        return self.pos == self.steps

    def compute_gae(self, last_value: float, gamma: float, lam: float) -> None:
        """
        Fill `advantages` and `returns`. `rewards` must already include the
        bootstrap of truncated episodes; `dones` cuts the recursion at every
        episode end, `last_value` bootstraps a rollout that ends mid-episode.
        """
        advantage = 0.0
        for t in reversed(range(self.steps)):
            next_value = self.values[t + 1] if t + 1 < self.steps else last_value
            alive = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * alive - self.values[t]
            advantage = delta + gamma * lam * alive * advantage
            self.advantages[t] = advantage
        self.returns[:] = self.advantages + self.values

    def reset(self) -> None:
        self.pos = 0
