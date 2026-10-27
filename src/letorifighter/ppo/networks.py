"""Policy and value networks: a factored categorical policy over 22 heads of 4."""

from itertools import pairwise

import numpy as np
import torch
from torch import nn

NUM_HEADS = 22
NUM_CHOICES = 4


class RunningNorm(nn.Module):
    """Running mean and variance of the observation vector (clipped z-score)."""

    mean: torch.Tensor
    var: torch.Tensor
    count: torch.Tensor

    def __init__(self, dim: int, clip: float = 10.0) -> None:
        super().__init__()
        self.clip = clip
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))

    def update(self, batch: np.ndarray) -> None:
        x = torch.as_tensor(batch, dtype=torch.float32, device=self.mean.device)
        n = x.shape[0]
        delta = x.mean(0) - self.mean
        total = self.count + n
        self.mean += delta * n / total
        m2 = self.var * self.count + x.var(0, unbiased=False) * n
        m2 += delta**2 * self.count * n / total
        self.var.copy_(m2 / total)
        self.count.copy_(total)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = (x - self.mean) / torch.sqrt(self.var + 1e-6)
        return z.clamp(-self.clip, self.clip)


def mlp(sizes: list[int], last_gain: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i, (a, b) in enumerate(pairwise(sizes)):
        linear = nn.Linear(a, b)
        last = i == len(sizes) - 2
        nn.init.orthogonal_(linear.weight, last_gain if last else np.sqrt(2))
        nn.init.zeros_(linear.bias)
        layers.append(linear)
        if not last:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """
    Separate actor and critic MLPs on `[normalised observation, task embedding]`.
    `logits(obs, task)` has shape (batch, 22, 4): one categorical per joint or grip.
    """

    def __init__(
        self,
        obs_dim: int,
        num_tasks: int,
        hidden: int = 256,
        task_dim: int = 16,
        num_heads: int = NUM_HEADS,
        num_choices: int = NUM_CHOICES,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_choices = num_choices
        self.norm = RunningNorm(obs_dim)
        self.task_embedding = nn.Embedding(num_tasks, task_dim)
        inputs = obs_dim + task_dim
        self.actor = mlp([inputs, hidden, hidden, num_heads * num_choices], 0.01)
        self.critic = mlp([inputs, hidden, hidden, 1], 1.0)

    def features(self, obs: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.norm(obs), self.task_embedding(task)], dim=-1)

    def logits(self, obs: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        x = self.features(obs, task)
        return self.actor(x).view(-1, self.num_heads, self.num_choices)

    def value(self, obs: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        return self.critic(self.features(obs, task)).squeeze(-1)
