"""Checks that a training episode belongs to the `train` split of the definitions.

The M5 dataset is training data: no episode of `val` or any `test_*` split may be
in it. `check_spec(definitions, spec)` returns the list of violations of one
episode (an `EpisodeSpec` or its dict); an empty list means the episode is a
train episode on every axis (split name, seed range, task, instruction,
opponent, pose). It is used when the collection plan is built and again by the
dataset card, on the sidecar records.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass

from letorifighter.tasks import Definitions

SPLIT = "train"


def check_spec(definitions: Definitions, spec) -> list[str]:
    """Violations of the train split by one episode spec (empty: it is clean)."""
    s: Mapping = asdict(spec) if is_dataclass(spec) else spec  # type: ignore[arg-type]
    train = definitions.splits[SPLIT]
    problems = []
    if s["split"] != SPLIT:
        problems.append(f"split is {s['split']!r}, not {SPLIT!r}")
    seed = int(s["episode_seed"])
    low, high = train["episode_seed_range"]
    if not low <= seed <= high:
        problems.append(f"episode seed {seed} outside the train range {low}-{high}")
    for name, split in definitions.splits.items():
        low, high = split["episode_seed_range"]
        if name != SPLIT and low <= seed <= high:
            problems.append(f"episode seed {seed} is in the {name} range")
    task = s["task"]
    if task not in train["tasks"]:
        problems.append(f"task {task!r} is not a train task")
        return problems
    spec_task = definitions.tasks[task]
    if spec_task.kind == "sequence":
        problems.append(f"task {task!r} is a composition")
    elif s["instruction"] not in spec_task.paraphrases[train["paraphrases"]]:
        problems.append(f"instruction {s['instruction']!r} is not a train paraphrase")
    if s["opponent"] not in definitions.opponent_pool(task, train["opponents"]):
        problems.append(f"opponent {s['opponent']!r} is not in the train pool")
    groups = definitions.document["poses"][train["poses"]]
    pose = (int(s["preroll_turns"]), int(s["pose_seed"]))
    if not any(
        g["preroll_turns"] == pose[0]
        and g["seed_range"][0] <= pose[1] <= g["seed_range"][1]
        for g in groups
    ):
        problems.append(
            f"pose (preroll {pose[0]}, seed {pose[1]}) is not in the train pool"
        )
    return problems


def assert_train_spec(definitions: Definitions, spec) -> None:
    problems = check_spec(definitions, spec)
    if problems:
        raise ValueError("not a train episode: " + "; ".join(problems))
