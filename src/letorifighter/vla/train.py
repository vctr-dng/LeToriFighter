"""Trainer for the M6 VLA: `uv run --group vla python scripts/train_vla.py ...`.

One run = one `VLAConfig` (YAML file and/or `--flag value` overrides). It trains
for `steps` optimiser steps with AdamW, linear warmup and cosine decay, bf16
autocast, gradient clipping; logs to a local MLflow store (sqlite file
`<data>/mlruns/mlflow.db`, experiment `m6-vla`; MLflow 3 deprecates the plain file
store) and to `<out_dir>/metrics.jsonl`; evaluates
offline on held-out episodes every `eval_every` steps (rounded per-component
accuracy against the recorded actions); writes `<out_dir>/last` every `save_every`
steps (model + optimizer + scheduler; `--resume` continues from it; deleted when the
run completes) and `<out_dir>/final` (model, config and state statistics, about
0.9 GB; `adapter.load_policy` reads it).
With `eval_game_episodes > 0` it also plays the policy in the game through
`letorifighter.eval.evaluate` (not tested: it needs Toribash).
"""
# pyrefly: ignore-errors
# (torch/lerobot/transformers are optional and loosely typed; checked by the tests)

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

from . import data_dir
from .config import (
    VLAConfig,
    add_cli_arguments,
    dumps,
    load_config,
    overrides_from_args,
)
from .data import VLADataset
from .model import (
    batch_to_device,
    build_policy,
    compute_loss,
    count_parameters,
    predict_states,
)
from .text import TextEncoder, instruction_task_ids


def resolve_task_ids(root: Path) -> dict[str, int]:
    """instruction -> task id: from `synthetic.json` for synthetic data, else the
    frozen task definitions."""
    description = root / "synthetic.json"
    if description.exists():
        return json.loads(description.read_text())["task_ids"]
    return instruction_task_ids()


def make_datasets(cfg: VLAConfig) -> tuple[VLADataset, VLADataset | None]:
    root = Path(cfg.dataset)
    total = LeRobotDatasetMetadata(cfg.repo_id, root=root).total_episodes
    n_val = min(cfg.val_episodes, total - 1) if total > 1 else 0
    train_eps = list(range(total - n_val))
    val_eps = list(range(total - n_val, total))
    encoder = TextEncoder(_vlm_name(cfg), cfg.tokenizer_max_length)
    common = {
        "repo_id": cfg.repo_id,
        "conditioning": cfg.conditioning,
        "task_ids": resolve_task_ids(root) if cfg.conditioning == "task_id" else None,
        "use_images": cfg.use_images,
        "text_encoder": encoder,
        "preload_images": cfg.preload_images,
    }
    train = VLADataset(root, cfg.chunk_size, episodes=train_eps, **common)
    val = (
        VLADataset(
            root, cfg.chunk_size, episodes=val_eps, state_stats=train.stats, **common
        )
        if val_eps
        else None
    )
    return train, val


def _vlm_name(cfg: VLAConfig) -> str:
    from .model import base_config_kwargs

    return base_config_kwargs(cfg.checkpoint).get(
        "vlm_model_name", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    )


def infinite(loader):
    while True:
        yield from loader


def lr_factor(step: int, warmup: int, total: int, floor: float = 0.1) -> float:
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def autocast(cfg: VLAConfig):
    enabled = cfg.bf16 and cfg.device.startswith("cuda")
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=enabled)


@torch.no_grad()
def evaluate_offline(policy, val: VLADataset, cfg: VLAConfig) -> dict[str, float]:
    """Rounded accuracy of the predicted chunks against the recorded actions on
    `eval_frames` evenly spaced validation frames (chance is 0.25)."""
    n = min(cfg.eval_frames, len(val))
    indices = np.linspace(0, len(val) - 1, n).round().astype(int)
    subset = torch.utils.data.Subset(val, list(indices))
    loader = torch.utils.data.DataLoader(subset, batch_size=cfg.batch_size)
    policy.eval()
    correct = correct_first = exact_first = total = total_first = 0
    with torch.random.fork_rng():
        torch.manual_seed(1234)
        for batch in loader:
            batch = batch_to_device(batch, cfg.device)
            with autocast(cfg):
                pred = predict_states(policy, batch, cfg.head)
            target = batch["action_states"]
            valid = ~batch["action_is_pad"]  # (B, k)
            hit = pred == target  # (B, k, 22)
            correct += (hit & valid.unsqueeze(-1)).sum().item()
            total += valid.sum().item() * hit.shape[-1]
            correct_first += hit[:, 0].sum().item()
            exact_first += hit[:, 0].all(dim=-1).sum().item()
            total_first += hit.shape[0] * hit.shape[-1]
    policy.train()
    return {
        "val/acc": correct / max(total, 1),
        "val/acc_first": correct_first / max(total_first, 1),
        "val/exact_first": exact_first / max(total_first // 22, 1),
    }


def save_checkpoint(
    path: Path, policy, cfg, state_dim, stats, optimizer=None, scheduler=None, step=0
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "cfg": cfg.to_dict(),
            "state_dim": state_dim,
            "state_stats": stats,
            "model": policy.state_dict(),
            "step": step,
        },
        path / "model.pt",
    )
    if optimizer is not None:
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
            },
            path / "train.pt",
        )


def run_game_eval(policy, cfg: VLAConfig, stats, step: int) -> dict[str, Any]:
    """Play `eval_game_episodes` episodes of `eval_split` on a private Xvfb instance."""
    from letorifighter.eval import evaluate
    from letorifighter.tasks import LeToriTaskEnv

    from .adapter import VLAPolicy

    adapter = VLAPolicy(policy, cfg, stats)
    from letorifighter.direct import own_instance

    with own_instance() as kwargs:
        env = LeToriTaskEnv(split=cfg.eval_split, env_kwargs=kwargs)
        try:
            report = evaluate(
                adapter,
                env,
                cfg.eval_split,
                cfg.eval_game_episodes,
                out_path=Path(cfg.out_dir) / f"game_eval_{step}.json",
                policy_name=f"vla-{cfg.head}",
            )
        finally:
            env.close()
    return report["overall"]


def train(cfg: VLAConfig) -> dict[str, Any]:
    import mlflow

    cfg.validate()
    if not cfg.dataset:
        raise ValueError("config.dataset (dataset root) is required")
    run_name = cfg.run_name or f"{cfg.head}-{cfg.init}-{int(time.time())}"
    out = Path(cfg.out_dir or data_dir() / "runs" / run_name)
    cfg.out_dir = str(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    train_set, val_set = make_datasets(cfg)
    policy = build_policy(cfg, train_set.state_dim)
    policy.train()
    params = [p for p in policy.parameters() if p.requires_grad]
    trainable, total = count_parameters(policy)
    optimizer = torch.optim.AdamW(
        params,
        lr=cfg.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_factor(s, cfg.warmup_steps, cfg.steps)
    )
    step = 0
    last = out / "last"
    if cfg.resume and (last / "train.pt").exists():
        saved = torch.load(last / "model.pt", map_location="cpu", weights_only=False)
        policy.load_state_dict(saved["model"], strict=False)
        state = torch.load(last / "train.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step = state["step"]
        print(f"resumed from step {step}")

    loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=len(train_set) >= cfg.batch_size,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        generator=torch.Generator().manual_seed(cfg.seed + step),
    )
    batches = infinite(loader)

    (data_dir() / "mlruns").mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{data_dir() / 'mlruns' / 'mlflow.db'}")
    mlflow.set_experiment(cfg.experiment)
    on_cuda = cfg.device.startswith("cuda")
    if on_cuda:
        torch.cuda.reset_peak_memory_stats()
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    summary: dict[str, Any] = {}
    with mlflow.start_run(run_name=run_name), open(out / "metrics.jsonl", "a") as log:
        mlflow.log_params(
            {k: v for k, v in cfg.to_dict().items() if k != "extra"}
            | {"trainable_params": trainable, "total_params": total}
        )
        (out / "config.json").write_text(dumps(cfg))
        print(f"{run_name}: {trainable / 1e6:.1f}M trainable of {total / 1e6:.1f}M")

        def record(metrics: dict[str, float]) -> None:
            mlflow.log_metrics(metrics, step=step)
            row = {"step": step, **metrics}
            history.append(row)
            log.write(json.dumps(row) + "\n")
            log.flush()

        window_loss, window_n, window_time = 0.0, 0, 0.0
        while step < cfg.steps:
            t0 = time.perf_counter()
            batch = batch_to_device(next(batches), cfg.device)
            with autocast(cfg):
                loss, _ = compute_loss(policy, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            window_loss += loss.item()
            window_n += 1
            window_time += time.perf_counter() - t0
            if step % cfg.log_every == 0 or step == cfg.steps:
                metrics = {
                    "train/loss": window_loss / window_n,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/step_s": window_time / window_n,
                }
                if on_cuda:
                    metrics["gpu/peak_alloc_gb"] = (
                        torch.cuda.max_memory_allocated() / 1e9
                    )
                record(metrics)
                print(
                    f"step {step}/{cfg.steps} loss {metrics['train/loss']:.4f} "
                    f"{metrics['train/step_s']:.2f}s/step"
                )
                window_loss, window_n, window_time = 0.0, 0, 0.0
            if val_set is not None and (
                step % cfg.eval_every == 0 or step == cfg.steps
            ):
                result = evaluate_offline(policy, val_set, cfg)
                record(result)
                print(
                    "  val acc {val/acc:.3f} first {val/acc_first:.3f} "
                    "exact_first {val/exact_first:.3f}".format(**result)
                )
            if cfg.eval_game_episodes > 0 and (
                step % cfg.eval_every == 0 or step == cfg.steps
            ):
                overall = run_game_eval(policy, cfg, train_set.stats, step)
                record({"game/success_rate": overall["success_rate"]})
                policy.train()
            if step % cfg.save_every == 0 and step < cfg.steps:
                save_checkpoint(
                    last,
                    policy,
                    cfg,
                    train_set.state_dim,
                    train_set.stats,
                    optimizer,
                    scheduler,
                    step,
                )
        save_checkpoint(
            out / "final", policy, cfg, train_set.state_dim, train_set.stats, step=step
        )
        shutil.rmtree(last, ignore_errors=True)  # a finished run needs no resume state
        final_val = next((r for r in reversed(history) if "val/acc" in r), {})
        summary = {
            "run": run_name,
            "head": cfg.head,
            "init": cfg.init,
            "steps": step,
            "wall_seconds": time.perf_counter() - started,
            "trainable_params": trainable,
            "total_params": total,
            "peak_gpu_alloc_gb": (
                torch.cuda.max_memory_allocated() / 1e9 if on_cuda else None
            ),
            "final_train_loss": next(
                (r["train/loss"] for r in reversed(history) if "train/loss" in r), None
            ),
            "final_val": {k: v for k, v in final_val.items() if k != "step"},
            "out_dir": str(out),
        }
        mlflow.log_metrics({"wall_seconds": summary["wall_seconds"]}, step=step)
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train a SmolVLA policy on a LeRobot v3 Toribash dataset."
    )
    parser.add_argument("--config", help="YAML file with VLAConfig fields")
    add_cli_arguments(parser)
    args = parser.parse_args(argv)
    cfg = load_config(args.config, overrides_from_args(args))
    summary = train(cfg)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
