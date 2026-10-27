"""Frozen definitions: validation, determinism, held-out separation (no game)."""

import copy
import json

import pytest

from letorifighter.tasks import Definitions, EpisodeSpec
from letorifighter.tasks.definitions import SPLITS_DIR

# Pin of the frozen v1 file. If this changes the split definition changed: make
# a new version file instead of editing v1.
V1_DIGEST = "f8e7d001b8cedb9410c455f4433d718421831bf751c48c3e2b46a24445832ebc"

DEFS = Definitions.load("v1")


def test_v1_is_frozen():
    assert DEFS.version == "1.0.0"
    assert DEFS.digest == V1_DIGEST
    assert DEFS.validate() == []


def test_every_task_has_detector_and_language():
    for name, task in DEFS.tasks.items():
        DEFS.detector(name)
        if task.kind == "sequence":
            assert len(task.steps) == 2
        else:
            assert len(task.paraphrases["train"]) >= 4
            assert len(task.paraphrases["heldout"]) >= 2
            assert task.clauses


def test_task_ids_are_dense_and_unique():
    ids = sorted(DEFS.task_ids.values())
    assert ids == list(range(len(ids)))


def test_sampling_is_deterministic_and_seeded():
    a = DEFS.episodes("val", 12, seed=3)
    b = DEFS.episodes("val", 12, seed=3)
    c = DEFS.episodes("val", 12, seed=4)
    assert a == b
    assert a != c
    assert EpisodeSpec.from_dict(a[0].to_dict()) == a[0]
    json.dumps([e.to_dict() for e in a])


def test_tasks_are_covered_evenly():
    episodes = DEFS.episodes("train", 50, seed=0)
    counts = {
        t: sum(e.task == t for e in episodes) for t in DEFS.splits["train"]["tasks"]
    }
    assert max(counts.values()) - min(counts.values()) <= 1


def pools(split):
    episodes = DEFS.episodes(split, 400, seed=0)
    return {
        "tasks": {e.task for e in episodes},
        "opponents": {e.opponent for e in episodes},
        "poses": {(e.preroll_turns) for e in episodes},
        "paraphrases": {e.instruction for e in episodes},
        "pose_seeds": {
            (e.preroll_turns, e.pose_seed) for e in episodes if e.preroll_turns
        },
        "episode_seeds": {e.episode_seed for e in episodes},
    }


def test_heldout_instructions():
    train, heldout = pools("train"), pools("test_paraphrase")
    assert train["tasks"] == heldout["tasks"]
    assert not train["paraphrases"] & heldout["paraphrases"]


def test_heldout_compositions():
    train, comps = pools("train"), pools("test_composition")
    assert not train["tasks"] & comps["tasks"]
    assert all(DEFS.tasks[t].kind == "sequence" for t in comps["tasks"])
    # Compositions are built from the clauses of known tasks
    for episode in DEFS.episodes("test_composition", 20, seed=1):
        spec = DEFS.tasks[episode.task]
        text = episode.instruction.lower().rstrip(".")
        for step in spec.steps:
            assert any(c in text for c in DEFS.tasks[step].clauses)
    # and no training instruction is a composition
    assert all(e.paraphrase.startswith("train:") for e in DEFS.episodes("train", 50))


def test_heldout_opponents():
    train, heldout = pools("train"), pools("test_opponent")
    assert not train["opponents"] & heldout["opponents"]
    assert heldout["opponents"] <= {"scripted:flail", "scripted:guard", "scripted:rush"}
    assert len(heldout["opponents"]) >= 2


def test_escape_uses_chasers_only():
    for split in ("train", "test_opponent"):
        for episode in DEFS.episodes(split, 100, seed=0):
            if episode.task == "escape":
                assert episode.opponent in ("scripted:advance_grab", "scripted:rush")


def test_heldout_poses():
    train, val, heldout = pools("train"), pools("val"), pools("test_pose")
    assert heldout["poses"] == {6}
    assert max(train["poses"] | val["poses"]) < 6
    assert not heldout["pose_seeds"] & (train["pose_seeds"] | val["pose_seeds"])
    assert not val["pose_seeds"] & train["pose_seeds"]


def test_episode_seeds_are_disjoint_between_splits():
    names = list(DEFS.splits)
    seeds = {n: pools(n)["episode_seeds"] for n in names}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            assert not seeds[a] & seeds[b]


def test_max_episode_turns_covers_preroll_plus_horizon():
    assert DEFS.max_episode_turns("test_pose") == 6 + 50
    assert DEFS.max_episode_turns() >= 80


def test_unknown_split_raises():
    with pytest.raises(KeyError):
        DEFS.episodes("nope", 1)


def test_validation_catches_leaks():
    document = copy.deepcopy(DEFS.document)
    document["opponents"]["heldout"].append("passive")
    with pytest.raises(ValueError, match="opponents overlap"):
        Definitions(document)
    document = copy.deepcopy(DEFS.document)
    document["splits"]["test_composition"]["tasks"].append("reach")
    with pytest.raises(ValueError):
        Definitions(document)
    document = copy.deepcopy(DEFS.document)
    task = document["tasks"]["reach"]
    task["paraphrases"]["heldout"].append(task["paraphrases"]["train"][0])
    with pytest.raises(ValueError, match="paraphrases overlap"):
        Definitions(document)


def test_splits_dir_has_v1():
    assert (SPLITS_DIR / "v1.json").is_file()
