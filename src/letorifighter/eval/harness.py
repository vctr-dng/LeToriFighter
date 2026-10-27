"""Evaluation harness: run a policy over a split and report the plan's metrics.

`evaluate(policy, env, split, episodes, seed)` plays `episodes` episodes of
`split` (sampled by `Definitions.episodes`, so the same arguments give the same
episodes) and returns a JSON-serialisable report:

* `success_rate`: fraction of episodes where the task detector latched success
  (with a 95% Wilson interval), and `success_rate_nontrivial` over episodes that
  did not already satisfy the first stage at the start;
* `win_rate` / `tie_rate`: win means the opponent took more damage than the
  learner during the task (damage before the task, i.e. in the pose pre-roll, is
  excluded);
* `injury_delta_mean`: mean of (damage to the opponent minus damage to the
  learner) per episode;
* `latency_ms`: wall time of the policy call per turn (mean, p50, p95, max over
  all turns), the "inference latency per turn" of the plan;
* the same metrics broken down by task, opponent and pose group, and every
  episode with its spec.

A policy is a callable `policy(obs) -> action` on the task observation (the
learner's `Dict` plus `task_id` and `instruction`); if it has `reset(seed)` it is
called at the start of every episode.
"""

import json
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import numpy as np

from letorifighter.tasks import Definitions, EpisodeSpec, LeToriTaskEnv

REPORT_SCHEMA = 1


def wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float]:
    if total == 0:
        return [0.0, 1.0]
    p = successes / total
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return [float(max(0.0, centre - half)), float(min(1.0, centre + half))]


def run_episode(policy: Callable, env: LeToriTaskEnv, spec: EpisodeSpec) -> dict:
    """Play one episode and return its record."""
    obs, info = env.reset(options={"episode": spec})
    if hasattr(policy, "reset"):
        policy.reset(spec.episode_seed)
    latencies = []
    reward_total = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        started = time.perf_counter()
        action = policy(obs)
        latencies.append(time.perf_counter() - started)
        obs, reward, terminated, truncated, info = env.step(action)
        reward_total += reward
    return {
        "spec": spec.to_dict(),
        "success": bool(info["success"]),
        "success_turn": info["success_turn"],
        "trivial_start": bool(info["trivial_start"]),
        "turns": info["turn"],
        "winner": info["winner"],
        "learner_injury": info["learner_injury"],
        "opponent_injury": info["opponent_injury"],
        "injury_delta": info["injury_delta"],
        "return": reward_total,
        "latency_ms": [1000.0 * x for x in latencies],
    }


def summarize(records: list[dict]) -> dict:
    n = len(records)
    if n == 0:
        return {"episodes": 0}
    successes = sum(r["success"] for r in records)
    nontrivial = [r for r in records if not r["trivial_start"]]
    latencies = np.array([x for r in records for x in r["latency_ms"]])
    wins = sum(r["winner"] == "learner" for r in records)
    ties = sum(r["winner"] is None for r in records)
    return {
        "episodes": n,
        "success_rate": successes / n,
        "success_ci95": wilson_interval(successes, n),
        "success_rate_nontrivial": (
            sum(r["success"] for r in nontrivial) / len(nontrivial)
            if nontrivial
            else None
        ),
        "trivial_start_rate": 1 - len(nontrivial) / n,
        "win_rate": wins / n,
        "tie_rate": ties / n,
        "injury_delta_mean": float(np.mean([r["injury_delta"] for r in records])),
        "mean_success_turn": (
            float(np.mean([r["success_turn"] for r in records if r["success"]]))
            if successes
            else None
        ),
        "latency_ms": {
            "mean": float(latencies.mean()),
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
            "max": float(latencies.max()),
            "turns": int(latencies.size),
        },
    }


def pose_group(spec: dict) -> str:
    return f"preroll{spec['preroll_turns']}"


def build_report(records: list[dict], meta: dict) -> dict:
    groups = {
        "by_task": lambda r: r["spec"]["task"],
        "by_opponent": lambda r: r["spec"]["opponent"],
        "by_pose": lambda r: pose_group(r["spec"]),
    }
    report = {"schema": REPORT_SCHEMA, **meta, "overall": summarize(records)}
    for key, group in groups.items():
        buckets = defaultdict(list)
        for record in records:
            buckets[group(record)].append(record)
        report[key] = {
            name: summarize(items) for name, items in sorted(buckets.items())
        }
    report["episodes"] = records
    return report


def evaluate(
    policy: Callable,
    env: LeToriTaskEnv,
    split: str,
    episodes: int,
    seed: int = 0,
    out_path: str | Path | None = None,
    policy_name: str = "policy",
    specs: list[EpisodeSpec] | None = None,
) -> dict:
    """
    Run `policy` for `episodes` episodes of `split`; write JSON if `out_path` is
    given. `specs` replaces the sampled episodes (e.g. a filtered subset of the split).
    """
    definitions: Definitions = env.definitions
    specs = specs if specs is not None else definitions.episodes(split, episodes, seed)
    episodes = len(specs)
    started = time.perf_counter()
    records = [run_episode(policy, env, spec) for spec in specs]
    meta = {
        "split": split,
        "definitions_version": definitions.version,
        "definitions_digest": definitions.digest,
        "n_episodes": episodes,
        "seed": seed,
        "policy": policy_name,
        "turnframes": definitions.turnframes,
        "wall_seconds": time.perf_counter() - started,
    }
    report = build_report(records, meta)
    if out_path is not None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=1) + "\n")
    return report
