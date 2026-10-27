"""Roll out learner policies on train-split episodes and record a demonstration dataset (M5).

Per turn `t` the dataset holds the RGB frame of the observation before the
action, the flat state vector of that observation (`flatten_observation`:
`STATE_DIM` = 300 values, see below), the 22-component action taken at `t`
(joint and grip states 0..3), and the episode's language instruction (a TRAIN
paraphrase, stored as the LeRobot task string). The terminal observation has no
action and is not stored, so an episode of `horizon_turns` turns has that many
steps. Extra per-step columns: `info.success` (detector latched at this step's
observation), `info.stage` (index of the current stage), `info.has_action`, `info.task_id`.

State vector (`STATE_DIM` = 300, float32), the learner's own view:
for `self` then `opponent`: limb positions (21x3, centred on that fighter's groin,
in the learner's groin frame), limb velocities (21x3), groin height (1), joint
states (20, values 1..4 as the game reports them), grips (2); then `match_frame`
(counted from the start of the task) and `frames_next_turn`.

What is kept: every episode is written, successful or not. A record's `success`
and `label` (`"success"` / `"failure"`) say which (failure: the detector never
fired). Sources flagged `action_free` go to a separate `clips` dataset with zero
actions and `info.has_action = 0` (frames and state only, for video/world-model
co-training); everything else goes to `demos`.

Only the `train` split of the definitions is ever used: the plan builder and the
card both refuse any episode that is not a train episode (`data.leakage`).

Parallel workers: every worker computes the same plan and takes the items
`i % num_workers == worker_id`, writing its own shard. Per-worker environment
settings (a game directory, an X display, a capture display for the future
ToriLLE instance pool) go through `env_kwargs` / `worker_env_kwargs` of the config
and are passed to `TwoPlayerToriEnv` unchanged.
"""

from __future__ import annotations

import contextlib
import fcntl
import itertools
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import numpy as np

from letorifighter.opponents import flatten_observation
from letorifighter.opponents.base import check_action
from letorifighter.opponents.network import OBS_DIM
from letorifighter.tasks import Definitions, EpisodeSpec

from .leakage import assert_train_spec
from .policies import CollectPolicy, build_policy
from .shards import SIDECAR, ShardWriter, dir_size_bytes, finished_seeds

STATE_DIM = OBS_DIM
LOCK_PATH = "/tmp/toribash-game.lock"
GB = 1024**3


# ------------------------------------------------------------------------- lock
@contextlib.contextmanager
def game_lock(path: str = LOCK_PATH, log: Callable[[str], None] = print):
    """Hold the machine-wide game lock (one Toribash per machine in Steam-launch mode).

    Taken inside the process, as `scripts/train_ppo.py` does; never wrap a command
    that calls this in `flock` on the same file (it would deadlock). Direct-launch
    workers with their own instance do not need it.
    """
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o666)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log(f"waiting for the game lock {path} ...")
            fcntl.flock(fd, fcntl.LOCK_EX)
        log("game lock acquired")
        yield
    finally:
        os.close(fd)  # closing the descriptor releases the lock


# ----------------------------------------------------------------------- config
@dataclass
class SourceConfig:
    """One source of episodes: a policy, how many episodes per task, which opponents."""

    name: str
    policy: str | dict
    episodes: int | dict[str, int]  # per task (an int applies to every task)
    opponents: list[str] | None = None  # None: the train pool of each task
    action_free: bool = False  # write to the clips dataset without actions


@dataclass
class CollectConfig:
    name: str = "m5"
    out_dir: str = "data/m5"
    definitions: str = "v2"
    split: str = "train"  # only "train" is accepted
    tasks: list[str] | None = None  # None: every task of the train split
    capture_size: int = 224
    fps: int = 10  # nominal rate of the stored timeline (one step per turn)
    plan_seed: int = 0
    segment_episodes: int = 50
    max_disk_gb: float = 15.0  # stop when `out_dir` exceeds this (project cap: 50 GB)
    use_videos: bool = True
    horizon_override: int | None = (
        None  # cut every episode to this many turns (smoke tests only)
    )
    env_kwargs: dict = field(
        default_factory=dict
    )  # TwoPlayerToriEnv kwargs, every worker
    worker_env_kwargs: list[dict] = field(default_factory=list)  # per worker id, on top
    sources: list[SourceConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict) -> CollectConfig:
        raw = dict(raw)
        known = {f.name for f in fields(cls)}
        if set(raw) - known:
            raise ValueError(f"unknown config keys: {sorted(set(raw) - known)}")
        sources = []
        source_keys = {f.name for f in fields(SourceConfig)}
        for source in raw.pop("sources", []):
            if set(source) - source_keys:
                raise ValueError(
                    f"unknown source keys: {sorted(set(source) - source_keys)}"
                )
            sources.append(SourceConfig(**source))
        config = cls(**raw, sources=sources)
        if config.split != "train":
            raise ValueError("only the train split may be collected into the dataset")
        if len({s.name for s in sources}) != len(sources):
            raise ValueError("source names must be unique")
        return config

    def to_dict(self) -> dict:
        from dataclasses import asdict

        return asdict(self)

    @property
    def out(self) -> Path:
        return Path(self.out_dir)


# ------------------------------------------------------------------------- plan
@dataclass
class PlanItem:
    source: str
    spec: EpisodeSpec
    action_free: bool = False


def source_counts(source: SourceConfig, tasks: list[str]) -> dict[str, int]:
    if isinstance(source.episodes, int):
        return dict.fromkeys(tasks, source.episodes)
    unknown = set(source.episodes) - set(tasks)
    if unknown:
        raise ValueError(
            f"source {source.name}: tasks {sorted(unknown)} are not collectable"
        )
    return dict(source.episodes)


def build_plan(config: CollectConfig, definitions: Definitions) -> list[PlanItem]:
    """
    The deterministic list of episodes to collect: train-split specs only, unique
    episode seeds across the whole plan, interleaved over (source, task) so that
    stopping early (disk cap) leaves a balanced mix. Identical for every worker.
    """
    train_tasks = list(definitions.splits["train"]["tasks"])
    tasks = config.tasks or train_tasks
    outside = set(tasks) - set(train_tasks)
    if outside:
        raise ValueError(f"tasks {sorted(outside)} are not in the train split")
    seen: set[int] = set()
    groups: list[list[PlanItem]] = []
    for source_index, source in enumerate(config.sources):
        seed = config.plan_seed * 1000 + source_index
        for task, count in source_counts(source, tasks).items():
            items: list[PlanItem] = []
            index = 0
            while len(items) < count:
                spec = definitions.sample("train", index, seed)
                index += 1
                if index > 1000 * max(count, 1) + 1000:
                    raise ValueError(
                        f"cannot draw {count} episodes of {task} for {source.name}"
                    )
                if spec.task != task or spec.episode_seed in seen:
                    continue
                if source.opponents and spec.opponent not in source.opponents:
                    continue
                assert_train_spec(definitions, spec)
                if config.horizon_override:
                    spec = replace(spec, horizon_turns=config.horizon_override)
                seen.add(spec.episode_seed)
                items.append(PlanItem(source.name, spec, source.action_free))
            groups.append(items)
    plan = []
    for row in itertools.zip_longest(*groups):
        plan.extend(item for item in row if item is not None)
    return plan


# ------------------------------------------------------------------ one episode
@dataclass
class CollectedEpisode:
    frames: np.ndarray  # (T, H, W, 3) uint8
    states: np.ndarray  # (T, STATE_DIM) float32
    actions: np.ndarray  # (T, 22) int64
    stages: np.ndarray  # (T,) current stage index at each observation
    success_flags: np.ndarray  # (T,) detector latched at each observation
    instruction: str
    info: dict  # final env info

    @property
    def turns(self) -> int:
        return len(self.frames)


def collect_episode(env, spec: EpisodeSpec, policy: CollectPolicy) -> CollectedEpisode:
    """Play `spec` with `policy` on a `LeToriTaskEnv` that provides an `rgb` observation."""
    obs, info = env.reset(options={"episode": spec})
    policy.reset(spec.episode_seed)
    frames, states, actions, stages, flags = [], [], [], [], []
    stage, success = 0, False
    while True:
        if "rgb" not in obs:
            raise KeyError(
                "observation has no `rgb` key: create the env with capture_size"
            )
        frames.append(np.asarray(obs["rgb"], dtype=np.uint8))
        states.append(flatten_observation(obs))
        stages.append(stage)
        flags.append(success)
        action = check_action(policy(obs))
        actions.append(action)
        obs, _, terminated, truncated, info = env.step(action)
        stage, success = int(info["stage"]), bool(info["success"])
        if terminated or truncated:
            break
    return CollectedEpisode(
        np.stack(frames),
        np.stack(states),
        np.stack(actions),
        np.array(stages),
        np.array(flags),
        spec.instruction,
        info,
    )


def episode_record(
    item: PlanItem,
    episode: CollectedEpisode,
    policy: CollectPolicy,
    config: CollectConfig,
    definitions: Definitions,
    worker_id: int,
    wall_seconds: float,
) -> dict:
    info = episode.info
    return {
        "spec": item.spec.to_dict(),
        "task": item.spec.task,
        "opponent": item.spec.opponent,
        "pose": f"preroll{item.spec.preroll_turns}:{item.spec.pose_seed}",
        "seed": item.spec.episode_seed,
        "split": item.spec.split,
        "source": item.source,
        "policy_id": policy.policy_id,
        "policy_component": policy.component,
        "action_free": item.action_free,
        "success": bool(info["success"]),
        "success_turn": info["success_turn"],
        "label": "success" if info["success"] else "failure",
        "final_stage": int(info["stage"]),
        "trivial_start": bool(info["trivial_start"]),
        "winner": info.get("winner"),
        "injury_delta": float(info["injury_delta"]),
        "turns": episode.turns,
        "instruction": episode.instruction,
        "capture_size": config.capture_size,
        "definitions_version": definitions.version,
        "definitions_digest": definitions.digest,
        "worker": worker_id,
        "wall_seconds": wall_seconds,
    }


# ------------------------------------------------------------------ environment
def make_env(config: CollectConfig, definitions: Definitions, worker_id: int = 0):
    """
    A `LeToriTaskEnv` on a `TwoPlayerToriEnv` with RGB capture (needs the game).

    `config.env_kwargs` and then `config.worker_env_kwargs[worker_id]` are passed to
    `TwoPlayerToriEnv`: this is where `game_dir`, `display` and `capture_display` of a
    ToriLLE direct-launch instance go. `draw_game=True` is required by capture.
    """
    from torille.envs import TwoPlayerToriEnv

    from letorifighter.tasks import LeToriTaskEnv

    kwargs: dict[str, Any] = {
        "matchframes": definitions.max_episode_turns("train")
        * (definitions.turnframes + 1),
        "turnframes": definitions.turnframes,
        "capture_size": config.capture_size,
        "draw_game": True,
        **config.env_kwargs,
    }
    if worker_id < len(config.worker_env_kwargs):
        kwargs.update(config.worker_env_kwargs[worker_id])
    from letorifighter.direct import check_env_kwargs

    check_env_kwargs(kwargs)  # refuses Steam launch and desktop displays
    env = TwoPlayerToriEnv(**kwargs)
    return LeToriTaskEnv(env=env, split="train", definitions=definitions, reward="none")


# ---------------------------------------------------------------- direct workers
def run_pool_workers(
    n: int,
    build_command: Callable[[int, dict], list[str]],
    log: Callable[[str], None] = print,
) -> list[int]:
    """Start a ToriLLE `InstancePool(n)` (one game copy and Xvfb each) and n worker processes.

    `build_command(worker_id, instance_kwargs)` gives the argv of worker `worker_id`;
    `instance_kwargs` is `dict(launch_mode, game_dir, display, capture_display)`. The
    workers run without `DISPLAY`/`WAYLAND_DISPLAY`. Returns the exit codes. The pool
    (games, Xvfb, copies) is removed at the end, also on Ctrl-C (the workers started
    here are terminated first).
    """
    import subprocess

    from torille.pool import InstancePool

    from letorifighter.direct import check_env_kwargs

    env = {
        k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY")
    }
    procs: list[subprocess.Popen] = []
    with InstancePool(n, display="xvfb") as pool:
        log(f"pool of {len(pool)} instances, {pool.disk_bytes / GB:.2f} GB")
        try:
            for worker_id, kwargs in enumerate(pool):
                check_env_kwargs(kwargs)
                procs.append(
                    subprocess.Popen(build_command(worker_id, kwargs), env=env)
                )
            return [p.wait() for p in procs]
        finally:
            for p in procs:
                if p.poll() is None:
                    p.terminate()
            for p in procs:
                try:
                    p.wait(30)
                except subprocess.TimeoutExpired:
                    p.kill()


# ------------------------------------------------------------------------- loop
def write_config(config: CollectConfig) -> None:
    config.out.mkdir(parents=True, exist_ok=True)
    path = config.out / "config.json"
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(config.to_dict(), indent=1) + "\n")
    os.replace(tmp, path)


def run_collection(
    config: CollectConfig,
    definitions: Definitions,
    env,
    *,
    worker_id: int = 0,
    num_workers: int = 1,
    plan: list[PlanItem] | None = None,
    max_episodes: int | None = None,
    writer_factory: Callable[..., Any] | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """
    Collect this worker's share of the plan into `out_dir/shards/w<worker_id>`.

    Skips episodes whose seed any shard already holds (resume), stops when the
    output directory exceeds `max_disk_gb` or after `max_episodes` new episodes.
    Returns a summary with `stop` in `complete`, `disk_cap`, `max_episodes`.
    """
    if not 0 <= worker_id < num_workers:
        raise ValueError("worker_id must be in [0, num_workers)")
    plan = build_plan(config, definitions) if plan is None else plan
    mine = plan[worker_id::num_workers]
    done = finished_seeds(config.out)
    todo = [item for item in mine if item.spec.episode_seed not in done]
    write_config(config)
    log(
        f"[w{worker_id}] plan {len(plan)} episodes, {len(mine)} for this worker, "
        f"{len(mine) - len(todo)} already done, {len(todo)} to collect"
    )
    kwargs = {} if writer_factory is None else {"writer_factory": writer_factory}
    writer = ShardWriter(
        config.out / "shards" / f"w{worker_id}",
        name=config.name,
        fps=config.fps,
        image_shape=(config.capture_size, config.capture_size, 3),
        state_dim=STATE_DIM,
        segment_episodes=config.segment_episodes,
        use_videos=config.use_videos,
        **kwargs,
    )
    sources = {s.name: s for s in config.sources}
    policies: dict[str, CollectPolicy] = {}
    cap = config.max_disk_gb * GB
    started = time.monotonic()
    turns = episodes = successes = 0
    stop = "complete"
    try:
        for item in todo:
            if max_episodes is not None and episodes >= max_episodes:
                stop = "max_episodes"
                break
            size = dir_size_bytes(config.out)
            if size > cap:
                log(
                    f"[w{worker_id}] disk cap reached: {size / GB:.2f} GB > {config.max_disk_gb} GB"
                )
                stop = "disk_cap"
                break
            if item.source not in policies:
                policies[item.source] = build_policy(sources[item.source].policy)
            policy = policies[item.source]
            t0 = time.monotonic()
            episode = collect_episode(env, item.spec, policy)
            record = episode_record(
                item,
                episode,
                policy,
                config,
                definitions,
                worker_id,
                time.monotonic() - t0,
            )
            writer.add(episode, record)
            episodes += 1
            turns += episode.turns
            successes += record["success"]
            elapsed = time.monotonic() - started
            log(
                f"[w{worker_id}] {episodes}/{len(todo)} {item.spec.task} vs {item.spec.opponent} "
                f"{item.source} {record['label']} turns={episode.turns} | "
                f"{turns / elapsed:.1f} turns/s, {episodes / elapsed * 3600:.0f} ep/h, "
                f"disk {dir_size_bytes(config.out) / GB:.3f} GB"
            )
    finally:
        writer.close()
    elapsed = time.monotonic() - started
    summary = {
        "worker": worker_id,
        "episodes": episodes,
        "successes": successes,
        "turns": turns,
        "wall_seconds": elapsed,
        "turns_per_second": turns / elapsed if elapsed else 0.0,
        "skipped_done": len(mine) - len(todo),
        "remaining": len(todo) - episodes,
        "disk_gb": dir_size_bytes(config.out) / GB,
        "stop": stop,
    }
    log(f"[w{worker_id}] done: {json.dumps(summary)}")
    return summary


__all__ = [
    "SIDECAR",
    "STATE_DIM",
    "CollectConfig",
    "CollectedEpisode",
    "PlanItem",
    "SourceConfig",
    "build_plan",
    "collect_episode",
    "game_lock",
    "make_env",
    "run_collection",
    "run_pool_workers",
]
