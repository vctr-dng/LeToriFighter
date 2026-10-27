"""PPO (clipped objective, GAE, advantage normalisation, entropy bonus), single env.

The observation is the task environment's `Dict` (`LeToriTaskEnv`): the learner's
own state is flattened with `letorifighter.opponents.flatten_observation` and the
`task_id` goes through an embedding, so one network serves every task. The text
`instruction` is ignored here (task id conditioning, the "language vs task id"
control of the plan).

Episodes that are truncated (the horizon) bootstrap from the value of the final
observation; terminated episodes do not. The observation normaliser is frozen
while a rollout is used for the update and refreshed after it, so the stored
log-probabilities stay consistent with the update.
"""

import os
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from letorifighter.opponents import flatten_observation
from letorifighter.opponents.network import OBS_DIM

from .buffer import RolloutBuffer
from .networks import ActorCritic

CHECKPOINT_VERSION = 1


@dataclass
class PPOConfig:
    rollout_steps: int = 1024
    epochs: int = 10
    minibatch_size: int = 256
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01  # on the mean entropy per head
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.05  # stop the epochs early when exceeded
    hidden: int = 256
    task_dim: int = 16
    seed: int = 0
    device: str = "cpu"
    threads: int = 2  # torch CPU threads; small nets gain nothing from more
    extra: dict = field(default_factory=dict)


def featurize(obs: dict) -> tuple[np.ndarray, int]:
    """Task observation to `(flat float32 vector, task id)`."""
    return flatten_observation(obs), int(obs["task_id"])


class PPOAgent:
    def __init__(
        self,
        num_tasks: int,
        config: PPOConfig | None = None,
        obs_dim: int = OBS_DIM,
        featurizer: Callable[[dict], tuple[np.ndarray, int]] = featurize,
    ) -> None:
        self.config = config or PPOConfig()
        self.num_tasks = num_tasks
        self.obs_dim = obs_dim
        self.featurizer = featurizer
        self.device = torch.device(self.config.device)
        torch.set_num_threads(self.config.threads)
        torch.manual_seed(self.config.seed)
        self.rng = np.random.default_rng(self.config.seed)
        self.model = ActorCritic(
            obs_dim, num_tasks, self.config.hidden, self.config.task_dim
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate, eps=1e-5
        )
        self.turns = 0  # environment turns taken so far
        self.updates = 0
        self.meta: dict = {}
        self.recent = deque(maxlen=50)  # recent finished episodes
        self._obs: dict | None = None  # live env state, not checkpointed
        self._episode_return = 0.0

    # --------------------------------------------------------------- acting
    def _tensors(self, vec: np.ndarray, task: int):
        x = torch.as_tensor(vec, dtype=torch.float32, device=self.device)[None]
        t = torch.as_tensor([task], dtype=torch.int64, device=self.device)
        return x, t

    @torch.no_grad()
    def act(self, obs: dict, deterministic: bool = False, generator=None):
        """Returns `(action (22,), log-prob, value)` for one observation."""
        vec, task = self.featurizer(obs)
        x, t = self._tensors(vec, task)
        logits = self.model.logits(x, t)
        if deterministic:
            action = logits.argmax(-1)
        else:
            probs = torch.softmax(logits, -1).view(-1, logits.shape[-1])
            action = torch.multinomial(probs, 1, generator=generator).view(1, -1)
        logp = Categorical(logits=logits).log_prob(action).sum(-1)
        value = self.model.value(x, t)
        return action[0].cpu().numpy(), float(logp), float(value)

    @torch.no_grad()
    def value_of(self, obs: dict) -> float:
        vec, task = self.featurizer(obs)
        return float(self.model.value(*self._tensors(vec, task)))

    def policy(self, deterministic: bool = True, seed: int | None = None) -> PPOPolicy:
        return PPOPolicy(self, deterministic, seed)

    # -------------------------------------------------------------- rollout
    def collect(self, env, buffer: RolloutBuffer) -> float:
        """Fill `buffer` from `env`; returns the bootstrap value of the last observation."""
        cfg = self.config
        buffer.reset()
        if self._obs is None:
            self._obs, _ = env.reset(seed=int(self.rng.integers(2**31 - 1)))
            self._episode_return = 0.0
        while not buffer.full:
            obs = self._obs
            action, logp, value = self.act(obs)
            vec, task = self.featurizer(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            self.turns += 1
            self._episode_return += float(reward)
            done = terminated or truncated
            step_reward = float(reward)
            if truncated and not terminated:
                step_reward += cfg.gamma * self.value_of(next_obs)
            buffer.add(vec, task, action, logp, value, step_reward, done)
            if done:
                self.recent.append(
                    {
                        "return": self._episode_return,
                        "success": bool(info.get("success", False)),
                    }
                )
                self._obs, _ = env.reset()
                self._episode_return = 0.0
            else:
                self._obs = next_obs
        return 0.0 if self._obs is None else self.value_of(self._obs)

    # --------------------------------------------------------------- update
    def update(self, buffer: RolloutBuffer, progress: float = 0.0) -> dict:
        cfg = self.config
        if cfg.anneal_lr:
            for group in self.optimizer.param_groups:
                group["lr"] = cfg.learning_rate * max(1.0 - progress, 0.0)
        dev = self.device
        obs = torch.as_tensor(buffer.obs, device=dev)
        task = torch.as_tensor(buffer.task, device=dev)
        actions = torch.as_tensor(buffer.actions, device=dev)
        old_logp = torch.as_tensor(buffer.logp, device=dev)
        returns = torch.as_tensor(buffer.returns, device=dev)
        old_values = torch.as_tensor(buffer.values, device=dev)
        advantages = torch.as_tensor(buffer.advantages, device=dev)

        stats = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "kl": [],
            "clip": [],
        }
        stop = False
        n = buffer.steps
        for _ in range(cfg.epochs):
            order = torch.as_tensor(self.rng.permutation(n), device=dev)
            for start in range(0, n, cfg.minibatch_size):
                idx = order[start : start + cfg.minibatch_size]
                dist = Categorical(logits=self.model.logits(obs[idx], task[idx]))
                new_logp = dist.log_prob(actions[idx]).sum(-1)
                entropy = dist.entropy().mean()
                log_ratio = new_logp - old_logp[idx]
                ratio = log_ratio.exp()
                adv = advantages[idx]
                if len(idx) > 1:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                policy_loss = torch.max(
                    -adv * ratio,
                    -adv * ratio.clamp(1 - cfg.clip_range, 1 + cfg.clip_range),
                ).mean()
                value = self.model.value(obs[idx], task[idx])
                value_loss = 0.5 * (value - returns[idx]).pow(2).mean()
                loss = (
                    policy_loss
                    + cfg.value_coef * value_loss
                    - cfg.entropy_coef * entropy
                )
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), cfg.max_grad_norm
                )
                self.optimizer.step()
                with torch.no_grad():
                    kl = ((ratio - 1) - log_ratio).mean()
                    clip = ((ratio - 1).abs() > cfg.clip_range).float().mean()
                stats["policy_loss"].append(float(policy_loss.detach()))
                stats["value_loss"].append(float(value_loss.detach()))
                stats["entropy"].append(float(entropy.detach()))
                stats["kl"].append(float(kl))
                stats["clip"].append(float(clip))
                if cfg.target_kl is not None and float(kl) > 1.5 * cfg.target_kl:
                    stop = True
                    break
            if stop:
                break
        self.model.norm.update(buffer.obs)  # refreshed only after the update
        self.updates += 1
        var_y = float(returns.var())
        explained = (
            1.0 - float((returns - old_values).var()) / var_y if var_y > 1e-8 else 0.0
        )
        result = {k: float(np.mean(v)) for k, v in stats.items()}
        result["explained_variance"] = explained
        return result

    # ---------------------------------------------------------------- train
    def learn(
        self,
        env,
        total_turns: int,
        on_update: Callable[[PPOAgent, dict], None] | None = None,
    ) -> None:
        """Train until `self.turns >= total_turns` (continues after `load`)."""
        buffer = RolloutBuffer(self.config.rollout_steps, self.obs_dim)
        while self.turns < total_turns:
            started = time.perf_counter()
            turns_before = self.turns
            last_value = self.collect(env, buffer)
            collect_seconds = time.perf_counter() - started
            buffer.compute_gae(last_value, self.config.gamma, self.config.gae_lambda)
            metrics = self.update(buffer, self.turns / total_turns)
            recent = list(self.recent)
            metrics.update(
                turns=self.turns,
                updates=self.updates,
                turns_per_second=(self.turns - turns_before) / collect_seconds,
                episodes_recent=len(recent),
                return_mean=float(np.mean([e["return"] for e in recent]))
                if recent
                else float("nan"),
                success_rate=float(np.mean([e["success"] for e in recent]))
                if recent
                else float("nan"),
            )
            if on_update is not None:
                on_update(self, metrics)

    # ---------------------------------------------------------- checkpoints
    def save(self, path: str | Path, meta: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": CHECKPOINT_VERSION,
            "num_tasks": self.num_tasks,
            "obs_dim": self.obs_dim,
            "config": asdict(self.config),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "turns": self.turns,
            "updates": self.updates,
            "rng": self.rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(),
            "meta": meta or {},
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)

    @classmethod
    def load(
        cls, path: str | Path, device: str | None = None, featurizer=featurize
    ) -> PPOAgent:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        config = PPOConfig(**payload["config"])
        if device is not None:
            config.device = device
        agent = cls(payload["num_tasks"], config, payload["obs_dim"], featurizer)
        agent.model.load_state_dict(payload["model"])
        agent.optimizer.load_state_dict(payload["optimizer"])
        agent.turns = payload["turns"]
        agent.updates = payload["updates"]
        agent.rng.bit_generator.state = payload["rng"]
        torch.set_rng_state(payload["torch_rng"])
        agent.meta = payload["meta"]
        return agent


class PPOPolicy:
    """`policy(obs) -> action` for `letorifighter.eval.evaluate` (argmax per head by default)."""

    def __init__(
        self, agent: PPOAgent, deterministic: bool = True, seed: int | None = None
    ) -> None:
        self.agent = agent
        self.deterministic = deterministic
        self.generator = torch.Generator().manual_seed(0 if seed is None else seed)

    def reset(self, seed=None) -> None:
        if seed is not None:
            self.generator.manual_seed(int(seed))

    def __call__(self, obs: dict) -> np.ndarray:
        action, _, _ = self.agent.act(obs, self.deterministic, self.generator)
        return action
