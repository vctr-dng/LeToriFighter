"""Turn a trained VLA into a callable for `letorifighter.eval.evaluate`.

`VLAPolicy(obs) -> action` takes the task observation of `LeToriTaskEnv` (the
learner's `Dict` plus `task_id` and `instruction`, and an optional `"image"`
uint8 (H, W, 3) once RGB capture exists) and returns 22 integer states.

Chunk execution (receding horizon): one forward pass predicts `chunk_size`
future actions. The first `n_action_steps` of them are queued and executed, one
per turn, without looking at the new observations; when the queue is empty the
model is called again on the current observation. `n_action_steps=1` replans
every turn (closed loop, one forward per turn); `n_action_steps=chunk_size`
executes the whole chunk open loop (SmolVLA's default is n_action_steps =
chunk_size). Flow-matching noise (interface a) is drawn from a generator seeded
by `reset(seed)`, so an episode is reproducible.
"""
# pyrefly: ignore-errors
# (torch/lerobot/transformers are optional and loosely typed; checked by the tests)

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from letorifighter.data.lerobot_writer import ACTION_DIM, IMAGE_KEY, STATE_KEY
from letorifighter.opponents.network import flatten_observation

from .config import VLAConfig
from .data import ATTENTION_KEY, MASK_KEY, NO_IMAGE_SIZE, TOKENS_KEY
from .model import build_policy, predict_states
from .text import TextEncoder, task_id_text


class VLAPolicy:
    def __init__(
        self,
        policy: torch.nn.Module,
        cfg: VLAConfig,
        state_stats: dict[str, np.ndarray],
        text_encoder: TextEncoder | None = None,
        state_fn: Callable[[dict], np.ndarray] = flatten_observation,
        n_action_steps: int | None = None,
    ) -> None:
        self.policy = policy.eval()
        self.cfg = cfg
        self.mean = torch.as_tensor(state_stats["mean"], dtype=torch.float32)
        self.std = torch.as_tensor(state_stats["std"], dtype=torch.float32)
        self.text_encoder = text_encoder or TextEncoder(
            policy.config.vlm_model_name, cfg.tokenizer_max_length
        )
        self.state_fn = state_fn
        self.n_action_steps = n_action_steps or cfg.n_action_steps
        if not 1 <= self.n_action_steps <= cfg.chunk_size:
            raise ValueError("need 1 <= n_action_steps <= chunk_size")
        self.device = cfg.device
        self._queue: deque[np.ndarray] = deque()
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(0)
        self.forward_passes = 0

    def reset(self, seed: int | None = None) -> None:
        """Drop the queued actions and reseed the flow-matching noise."""
        self._queue.clear()
        self._generator.manual_seed(0 if seed is None else int(seed))

    def _batch(self, obs: dict) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        state = torch.as_tensor(self.state_fn(obs), dtype=torch.float32)
        if state.numel() != self.mean.numel():
            raise ValueError(
                f"state has {state.numel()} values, the model expects {self.mean.numel()}"
            )
        text = (
            obs["instruction"]
            if cfg.conditioning == "language"
            else task_id_text(obs["task_id"])
        )
        tokens, attention = self.text_encoder(text)
        if cfg.use_images:
            if "image" not in obs:
                raise ValueError("this model uses images but the observation has none")
            image = torch.from_numpy(np.asarray(obs["image"])).permute(2, 0, 1).float()
            image = image / 255.0
        else:
            image = torch.zeros(3, NO_IMAGE_SIZE, NO_IMAGE_SIZE)
        batch = {
            IMAGE_KEY: image,
            MASK_KEY: torch.tensor(cfg.use_images),
            STATE_KEY: (state - self.mean) / self.std,
            TOKENS_KEY: tokens,
            ATTENTION_KEY: attention,
        }
        return {k: v[None].to(self.device) for k, v in batch.items()}

    @torch.no_grad()
    def predict_chunk(self, obs: dict) -> np.ndarray:
        """One forward pass: states `(chunk_size, 22)` as int64."""
        batch = self._batch(obs)
        noise = None
        if self.cfg.head == "relaxed":
            shape = (1, self.cfg.chunk_size, self.policy.config.max_action_dim)
            noise = torch.randn(shape, generator=self._generator).to(self.device)
        use_amp = self.cfg.bf16 and str(self.device).startswith("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            states = predict_states(self.policy, batch, self.cfg.head, noise=noise)
        self.forward_passes += 1
        return states[0].cpu().numpy().astype(np.int64)

    def __call__(self, obs: dict) -> np.ndarray:
        if not self._queue:
            chunk = self.predict_chunk(obs)
            self._queue.extend(chunk[: self.n_action_steps])
        action = self._queue.popleft()
        assert action.shape == (ACTION_DIM,)
        return action


def load_policy(
    path: str | Path, device: str | None = None, n_action_steps: int | None = None
) -> VLAPolicy:
    """Load a checkpoint directory written by the trainer (`<out>/final` or
    `<out>/last`) as a `VLAPolicy`."""
    saved = torch.load(Path(path) / "model.pt", map_location="cpu", weights_only=False)
    cfg = VLAConfig(**saved["cfg"])
    if device is not None:
        cfg.device = device
    policy = build_policy(cfg, saved["state_dim"], pretrained_weights=False)
    policy.load_state_dict(saved["model"], assign=True)
    policy.to(cfg.device)
    return VLAPolicy(policy, cfg, saved["state_stats"], n_action_steps=n_action_steps)


__all__ = ["VLAPolicy", "load_policy"]
