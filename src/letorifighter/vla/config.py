"""Run configuration: one flat dataclass, loadable from YAML and the CLI."""

from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_type_hints

HEADS = ("relaxed", "ce", "diffusion")
INITS = ("pretrained", "vlm", "scratch")


@dataclass
class VLAConfig:
    # --- data
    dataset: str = ""  # LeRobot v3 dataset root written by LeRobotEpisodeWriter
    repo_id: str = "local/toribash"
    val_episodes: int = 4  # last N episodes held out for the offline eval
    preload_images: bool = True  # decode all frames once (RAM: N*3*H*W bytes)

    # --- model / ablation switches
    head: str = "relaxed"  # relaxed (a) | ce (b) | diffusion (c, branch B)
    init: str = (
        "pretrained"  # pretrained (smolvla_base) | vlm (SmolVLM2 only) | scratch
    )
    use_images: bool = True  # False: image masked out (state-only)
    conditioning: str = "language"  # language | task_id
    chunk_size: int = 10  # k: actions predicted per forward
    n_action_steps: int = 5  # actions executed per forward at rollout (<= chunk_size)
    train_mode: str = "expert"  # expert (VLM frozen) | full
    image_size: int = 256  # images are resized with padding to this square size
    num_flow_steps: int = 10  # flow-matching Euler steps at inference (head relaxed)
    tokenizer_max_length: int = 48
    checkpoint: str = "lerobot/smolvla_base"
    bf16: bool = True  # autocast bfloat16 on cuda

    # --- training
    steps: int = 300
    batch_size: int = 16
    lr: float = 1e-4
    warmup_steps: int = 20
    weight_decay: float = 1e-10
    grad_clip: float = 10.0
    num_workers: int = 0
    seed: int = 0
    device: str = "cuda"
    eval_every: int = 100
    eval_frames: int = 256  # val frames used by the offline eval
    save_every: int = 100
    log_every: int = 10
    out_dir: str = ""  # default <data>/runs/<run_name>
    run_name: str = ""
    experiment: str = "m6-vla"
    resume: bool = False
    # in-game eval through letorifighter.eval.evaluate (needs the game; 0 = off)
    eval_game_episodes: int = 0
    eval_split: str = "val"

    # --- discrete diffusion head (head=diffusion)
    diffusion_steps: int = 4  # unmasking rounds at inference

    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.head not in HEADS:
            raise ValueError(f"head must be one of {HEADS}: {self.head}")
        if self.init not in INITS:
            raise ValueError(f"init must be one of {INITS}: {self.init}")
        if self.conditioning not in ("language", "task_id"):
            raise ValueError(f"conditioning: {self.conditioning}")
        if self.train_mode not in ("expert", "full"):
            raise ValueError(f"train_mode: {self.train_mode}")
        if not 1 <= self.n_action_steps <= self.chunk_size:
            raise ValueError("need 1 <= n_action_steps <= chunk_size")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def load_config(path: str | Path | None, overrides: dict[str, Any]) -> VLAConfig:
    import yaml

    data: dict[str, Any] = {}
    if path:
        data = yaml.safe_load(Path(path).read_text()) or {}
    data.update({k: v for k, v in overrides.items() if v is not None})
    known = {f.name for f in dataclasses.fields(VLAConfig)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    cfg = VLAConfig(**data)
    cfg.validate()
    return cfg


def add_cli_arguments(parser: argparse.ArgumentParser) -> None:
    """One `--flag` per config field (default None = keep YAML/dataclass value)."""
    hints = get_type_hints(VLAConfig)
    for f in dataclasses.fields(VLAConfig):
        if f.name == "extra":
            continue
        flag = "--" + f.name.replace("_", "-")
        kind = hints[f.name]
        if kind is bool:
            parser.add_argument(
                flag, dest=f.name, action=argparse.BooleanOptionalAction, default=None
            )
        else:
            parser.add_argument(flag, dest=f.name, type=kind, default=None)


def overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    names = {f.name for f in dataclasses.fields(VLAConfig)} - {"extra"}
    return {k: v for k, v in vars(args).items() if k in names and v is not None}


def dumps(cfg: VLAConfig) -> str:
    return json.dumps(cfg.to_dict(), indent=1, sort_keys=True)
