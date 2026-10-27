"""Loading, validating and sampling the frozen task and split definitions.

The definitions live in `splits/<version>.json` (see that file for the schema:
tasks with ids, horizons, language paraphrases, opponent pools, pose pools and
the named splits). A released version is never edited; a change makes a new
file, so results can name the version they used (`Definitions.version`,
`Definitions.digest`).

Episodes are sampled deterministically: `Definitions.episodes(split, n, seed)`
returns the same list for the same arguments. Tasks go round-robin so every task
is covered evenly; paraphrase, opponent, pose and episode seed are drawn per
episode from the split's pools.

Held-out axes:

* instructions: the `heldout` paraphrases of each task (never in `train`);
* compositions: sequence tasks (`reach_then_...`), never in the `train` split;
  their instruction is a composition template filled with the clauses of the
  steps, and the training instructions contain no composition template;
* opponents: the `heldout` opponent pool (disjoint from `train`);
* initial poses: `heldout` pose pool, a longer random pre-roll (see `poses.py`)
  with seeds that no other pool uses.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from pathlib import Path

import numpy as np

from .detectors import (
    ControlDetector,
    Detector,
    EscapeDetector,
    ReachDetector,
    SequenceDetector,
)

SPLITS_DIR = Path(__file__).parent / "splits"
DEFAULT_VERSION = "v2"


@dataclass(frozen=True)
class TaskSpec:
    name: str
    id: int
    kind: str  # reach, control, escape or sequence
    horizon_turns: int
    params: dict = field(default_factory=dict)
    steps: tuple[str, ...] = ()
    clauses: tuple[str, ...] = ()
    paraphrases: dict = field(
        default_factory=dict
    )  # {"train": [...], "heldout": [...]}
    opponents: dict | None = None  # per-task override of the opponent pools


@dataclass(frozen=True)
class EpisodeSpec:
    """Everything that defines one episode, so it can be replayed exactly."""

    split: str
    index: int
    task: str
    task_id: int
    instruction: str
    paraphrase: str  # e.g. "train:3", or "template:1" for compositions
    opponent: str
    preroll_turns: int
    pose_seed: int
    episode_seed: int
    horizon_turns: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> EpisodeSpec:
        return cls(**data)


def capitalize(text: str) -> str:
    return text[:1].upper() + text[1:]


class Definitions:
    """A loaded, validated definition file."""

    def __init__(self, document: dict) -> None:
        self.document = document
        self.version: str = document["version"]
        self.turnframes: int = document["turnframes"]
        self.digest = hashlib.sha256(
            json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.tasks: dict[str, TaskSpec] = {}
        for name, task in document["tasks"].items():
            self.tasks[name] = TaskSpec(
                name=name,
                id=task["id"],
                kind=task["kind"],
                horizon_turns=task["horizon_turns"],
                params=task.get("params", {}),
                steps=tuple(task.get("steps", ())),
                clauses=tuple(task.get("clauses", ())),
                paraphrases=task.get("paraphrases", {}),
                opponents=task.get("opponents"),
            )
        self.splits: dict = document["splits"]
        errors = self.validate()
        if errors:
            raise ValueError("invalid definitions: " + "; ".join(errors))

    # ------------------------------------------------------------ loading
    @classmethod
    def load(cls, version: str = DEFAULT_VERSION) -> Definitions:
        path = SPLITS_DIR / f"{version}.json"
        return cls(json.loads(path.read_text()))

    # --------------------------------------------------------- validation
    def validate(self) -> list[str]:
        doc = self.document
        errors = []
        ids = [t.id for t in self.tasks.values()]
        if len(set(ids)) != len(ids):
            errors.append("task ids are not unique")
        for task in self.tasks.values():
            if task.kind == "sequence":
                missing = [s for s in task.steps if s not in self.tasks]
                if missing or len(task.steps) < 2:
                    errors.append(f"{task.name}: bad steps {task.steps}")
                if any(
                    self.tasks[s].kind == "sequence"
                    for s in task.steps
                    if s in self.tasks
                ):
                    errors.append(f"{task.name}: nested sequences are not supported")
            else:
                train = task.paraphrases.get("train", [])
                heldout = task.paraphrases.get("heldout", [])
                if not train or not heldout:
                    errors.append(f"{task.name}: needs train and heldout paraphrases")
                if set(train) & set(heldout):
                    errors.append(f"{task.name}: train and heldout paraphrases overlap")
                if not task.clauses:
                    errors.append(f"{task.name}: needs clauses")
        opponents = doc["opponents"]
        if set(opponents["train"]) & set(opponents["heldout"]):
            errors.append("train and heldout opponents overlap")
        seen: dict[tuple[int, int], str] = {}
        for pool, groups in doc["poses"].items():
            for group in groups:
                lo, hi = group["seed_range"]
                if group["preroll_turns"] == 0:
                    continue
                for seed in (lo, hi):
                    key = (group["preroll_turns"], seed)
                    if key in seen and seen[key] != pool:
                        errors.append(f"pose seeds shared by {seen[key]} and {pool}")
                    seen[key] = pool
        for name, split in self.splits.items():
            for task in split["tasks"]:
                if task not in self.tasks:
                    errors.append(f"split {name}: unknown task {task}")
            for key, pools in (("opponents", opponents), ("poses", doc["poses"])):
                if split[key] not in pools:
                    errors.append(f"split {name}: unknown {key} pool {split[key]}")
            if split["paraphrases"] not in ("train", "heldout"):
                errors.append(f"split {name}: bad paraphrase set")
        train_tasks = set(self.splits["train"]["tasks"])
        for name, split in self.splits.items():
            if name.startswith("test_composition"):
                if train_tasks & set(split["tasks"]):
                    errors.append(f"split {name}: compositions seen in train")
                if any(self.tasks[t].kind != "sequence" for t in split["tasks"]):
                    errors.append(f"split {name}: not all tasks are compositions")
        ranges = sorted(tuple(s["episode_seed_range"]) for s in self.splits.values())
        for (_, hi), (lo, _) in pairwise(ranges):
            if lo <= hi:
                errors.append("episode seed ranges of splits overlap")
        return errors

    # ------------------------------------------------------------- lookup
    @property
    def task_ids(self) -> dict[str, int]:
        return {name: task.id for name, task in self.tasks.items()}

    def opponent_pool(self, task: str, pool: str) -> list[str]:
        override = self.tasks[task].opponents
        source = override if override else self.document["opponents"]
        return list(source[pool])

    def max_episode_turns(self, split: str | None = None) -> int:
        """Largest pre-roll plus horizon over a split (or all splits)."""
        names = [split] if split else list(self.splits)
        longest = 0
        for name in names:
            definition = self.splits[name]
            preroll = max(
                g["preroll_turns"] for g in self.document["poses"][definition["poses"]]
            )
            horizon = max(self.tasks[t].horizon_turns for t in definition["tasks"])
            longest = max(longest, preroll + horizon)
        return longest

    # ----------------------------------------------------------- detectors
    def detector(self, task: str) -> Detector:
        spec = self.tasks[task]
        if spec.kind == "sequence":
            return SequenceDetector([self.detector(step) for step in spec.steps])
        if spec.kind == "reach":
            return ReachDetector(**spec.params)
        if spec.kind == "control":
            return ControlDetector(**spec.params)
        if spec.kind == "escape":
            return EscapeDetector(**spec.params)
        raise ValueError(f"unknown task kind {spec.kind!r}")

    # ------------------------------------------------------------ sampling
    def instruction(self, task: str, paraphrase_set: str, rng) -> tuple[str, str]:
        """Draw an instruction; returns (text, id like "train:3" or "template:1")."""
        spec = self.tasks[task]
        if spec.kind != "sequence":
            options = spec.paraphrases[paraphrase_set]
            index = int(rng.integers(len(options)))
            return options[index], f"{paraphrase_set}:{index}"
        templates = self.document["composition_templates"]
        index = int(rng.integers(len(templates)))
        clauses = {}
        for letter, step in zip("ab", spec.steps, strict=True):
            options = self.tasks[step].clauses
            clauses[letter] = options[int(rng.integers(len(options)))]
        return capitalize(templates[index].format(**clauses)), f"template:{index}"

    def sample(self, split: str, index: int, seed: int) -> EpisodeSpec:
        definition = self.splits[split]
        rng = np.random.default_rng([seed, index, 7])
        tasks = definition["tasks"]
        task = tasks[(index + seed) % len(tasks)]
        spec = self.tasks[task]
        text, paraphrase = self.instruction(task, definition["paraphrases"], rng)
        opponents = self.opponent_pool(task, definition["opponents"])
        opponent = opponents[int(rng.integers(len(opponents)))]
        groups = self.document["poses"][definition["poses"]]
        group = groups[int(rng.integers(len(groups)))]
        lo, hi = group["seed_range"]
        lo_e, hi_e = definition["episode_seed_range"]
        return EpisodeSpec(
            split=split,
            index=index,
            task=task,
            task_id=spec.id,
            instruction=text,
            paraphrase=paraphrase,
            opponent=opponent,
            preroll_turns=group["preroll_turns"],
            pose_seed=int(rng.integers(lo, hi + 1)),
            episode_seed=int(rng.integers(lo_e, hi_e + 1)),
            horizon_turns=spec.horizon_turns,
        )

    def episodes(self, split: str, n: int, seed: int = 0) -> list[EpisodeSpec]:
        if split not in self.splits:
            raise KeyError(f"unknown split {split!r}, expected {sorted(self.splits)}")
        return [self.sample(split, i, seed) for i in range(n)]
