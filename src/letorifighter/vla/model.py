"""Build SmolVLA policies for the three interfaces and run them on batches.

Loading the pretrained checkpoint uses the workaround from
`docs/m6-smolvla-feasibility.md`: `SmolVLAPolicy.from_pretrained` fails on
Python 3.14 (draccus cannot parse the config json), so the `SmolVLAConfig` is
built from the checkpoint's `config.json` and the weights are copied with
`load_state_dict(strict=False)` after dropping keys whose name or shape differs
(`state_proj` when `max_state_dim` changes, and the heads of interface b/c).
"""
# pyrefly: ignore-errors
# (torch/lerobot/transformers are optional and loosely typed; checked by the tests)

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from safetensors.torch import load_file
from torch import Tensor, nn

from letorifighter.data.lerobot_writer import (
    ACTION_DIM,
    ACTION_KEY,
    IMAGE_KEY,
    STATE_KEY,
)

from .config import VLAConfig
from .heads import DiscreteHeadPolicy
from .levels import levels_to_states

# Keys of the checkpoint config.json that are not SmolVLAConfig fields to copy.
_DROP = {
    "type",
    "input_features",
    "output_features",
    "device",
    "use_amp",
    "push_to_hub",
    "repo_id",
    "private",
    "tags",
    "license",
    "normalization_mapping",
    "rtc_config",
    "pretrained_path",
}


def base_config_kwargs(checkpoint: str) -> dict[str, Any]:
    """Architecture fields of the checkpoint's `config.json`."""
    path = hf_hub_download(checkpoint, "config.json")
    raw = json.loads(Path(path).read_text())
    fields = {f.name for f in dataclasses.fields(SmolVLAConfig)}
    kwargs = {k: v for k, v in raw.items() if k in fields and k not in _DROP}
    for key in ("resize_imgs_with_padding", "optimizer_betas"):
        if key in kwargs:
            kwargs[key] = tuple(kwargs[key])
    return kwargs


def smolvla_config(
    cfg: VLAConfig, state_dim: int, load_vlm_weights: bool
) -> SmolVLAConfig:
    kwargs = base_config_kwargs(cfg.checkpoint)
    kwargs.update(
        chunk_size=cfg.chunk_size,
        n_action_steps=cfg.n_action_steps,
        num_steps=cfg.num_flow_steps,
        max_state_dim=max(state_dim, 32),
        max_action_dim=max(ACTION_DIM, 32),
        resize_imgs_with_padding=(cfg.image_size, cfg.image_size),
        tokenizer_max_length=cfg.tokenizer_max_length,
        pad_language_to="max_length",
        train_expert_only=cfg.train_mode == "expert",
        load_vlm_weights=load_vlm_weights,
        device=cfg.device,
        input_features={
            IMAGE_KEY: PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, cfg.image_size, cfg.image_size)
            ),
            STATE_KEY: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
        },
        output_features={
            ACTION_KEY: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))
        },
    )
    return SmolVLAConfig(**kwargs)


def load_base_weights(policy: nn.Module, checkpoint: str) -> dict[str, list[str]]:
    """Copy `model.safetensors` of the checkpoint into `policy` where name and
    shape match. Returns the keys that were skipped or left at their init."""
    path = hf_hub_download(checkpoint, "model.safetensors")
    saved = load_file(path)
    own = policy.state_dict()
    matching = {k: v for k, v in saved.items() if k in own and own[k].shape == v.shape}
    policy.load_state_dict(matching, strict=False)
    return {
        "reinitialised": sorted(set(own) - set(matching)),
        "unused": sorted(set(saved) - set(matching)),
    }


def build_policy(
    cfg: VLAConfig, state_dim: int, *, pretrained_weights: bool = True
) -> SmolVLAPolicy:
    """Construct the policy of `cfg.head` with the initialisation of `cfg.init`.

    `pretrained_weights=False` builds only the architecture (random init, no
    large downloads except the VLM config and tokenizer); used before loading a
    trained checkpoint.
    """
    if cfg.head == "diffusion":
        from .diffusion import DiffusionHeadPolicy as policy_class
    elif cfg.head == "ce":
        policy_class = DiscreteHeadPolicy
    else:
        policy_class = SmolVLAPolicy
    load_vlm = pretrained_weights and cfg.init in ("pretrained", "vlm")
    scfg = smolvla_config(cfg, state_dim, load_vlm)
    policy = policy_class(scfg)
    if pretrained_weights and cfg.init == "pretrained":
        info = load_base_weights(policy, cfg.checkpoint)
        policy.load_report = info  # type: ignore[attr-defined]
    return policy.to(cfg.device)


def count_parameters(policy: nn.Module) -> tuple[int, int]:
    """(trainable, total) parameter counts."""
    total = sum(p.numel() for p in policy.parameters())
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    return trainable, total


def compute_loss(policy: nn.Module, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
    """Training loss of a batch from `VLADataset` (already on the device):
    flow-matching MSE on the levels (a), cross-entropy over 22x4 logits (b) or
    masked-token cross-entropy (c)."""
    return policy(batch)


@torch.no_grad()
def predict_states(
    policy: nn.Module, batch: dict[str, Tensor], head: str, noise: Tensor | None = None
) -> Tensor:
    """Predicted chunk as states 0..3, long `(B, k, 22)`.

    Interface (a) samples the flow-matching expert and rounds each component to
    the nearest of the four levels; (b) takes the argmax of the 22x4 logits.
    """
    if head == "relaxed":
        levels = policy.predict_action_chunk(dict(batch), noise=noise)
        return levels_to_states(levels.float()).long()
    return policy.predict_action_chunk(dict(batch)).long()


def batch_to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


__all__ = [
    "base_config_kwargs",
    "batch_to_device",
    "build_policy",
    "compute_loss",
    "count_parameters",
    "load_base_weights",
    "predict_states",
    "smolvla_config",
]
