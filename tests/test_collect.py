"""Dataset collection on a fake env (no game, no torch, no lerobot): plan, policies,
episode recording, sharding, resume, disk guard, crash recovery, card, leakage check."""

import dataclasses
import json

import numpy as np
import pytest

from letorifighter.data import card as card_module
from letorifighter.data.collect import (
    STATE_DIM,
    CollectConfig,
    PlanItem,
    build_plan,
    collect_episode,
    run_collection,
)
from letorifighter.data.leakage import check_spec
from letorifighter.data.policies import build_policy
from letorifighter.data.shards import (
    CLIPS,
    DEMOS,
    SIDECAR,
    finished_seeds,
    merge_shards,
    read_jsonl,
)
from letorifighter.opponents import flatten_observation
from letorifighter.tasks import Definitions, LeToriTaskEnv
from letorifighter.testing import (
    FakeTwoPlayerEnv,
    make_observation,
    obs_with_hand_contact,
)

DEFS = Definitions.load("v2")
SIZE = 16
TURNS = 6


def config(tmp_path, **kwargs) -> CollectConfig:
    raw = {
        "name": "t",
        "out_dir": str(tmp_path / "run"),
        "capture_size": SIZE,
        "segment_episodes": 3,
        "sources": [
            {"name": "rand", "policy": "random", "episodes": {"reach": 4, "escape": 2}},
            {
                "name": "clip",
                "policy": {"policy": "hold", "epsilon": 0.5},
                "episodes": {"reach": 2},
                "action_free": True,
            },
        ],
    }
    raw.update(kwargs)
    return CollectConfig.from_dict(raw)


def short(plan: list[PlanItem]) -> list[PlanItem]:
    """The plan with every episode cut to TURNS turns (fake env, fast tests)."""
    return [
        dataclasses.replace(
            item, spec=dataclasses.replace(item.spec, horizon_turns=TURNS)
        )
        for item in plan
    ]


def contact_after(turn_start):
    def fn(turn, actions):
        if turn >= turn_start:
            obs = obs_with_hand_contact(match_frame=6.0 * turn)
        else:
            obs = make_observation(match_frame=6.0 * turn)
        return {"player_0": obs, "player_1": obs}

    return fn


def make_env(observation_fn=None):
    fake = FakeTwoPlayerEnv(capture_size=SIZE, observation_fn=observation_fn)
    return fake, LeToriTaskEnv(fake, split="train", definitions=DEFS)


class FakeWriter:
    """Stands in for `LeRobotEpisodeWriter`: stores arrays and the info.json counts."""

    def __init__(
        self, root, repo_id, *, fps, image_shape, state_dim, extra_features, use_videos
    ):
        from pathlib import Path

        self.root = Path(root)
        (self.root / "meta").mkdir(parents=True)
        self.state_dim, self.image_shape = state_dim, image_shape
        self.extra = set(extra_features)
        self.episodes = 0
        self.frames = 0

    def write_episode(self, frames, states, actions, language, extras=None):
        assert (
            frames.shape[1:] == self.image_shape and states.shape[1] == self.state_dim
        )
        assert set(extras) == self.extra
        np.savez(
            self.root / f"ep{self.episodes}.npz",
            frames=frames,
            states=states,
            actions=actions,
            language=language,
            **extras,
        )
        self.episodes += 1
        self.frames += len(frames)
        return self.episodes - 1

    def finalize(self):
        (self.root / "meta" / "info.json").write_text(
            json.dumps({"total_episodes": self.episodes, "total_frames": self.frames})
        )


def fake_aggregate(repo_ids, aggr_repo_id, roots, aggr_root):
    (aggr_root / "meta").mkdir(parents=True)
    infos = [json.loads((r / "meta" / "info.json").read_text()) for r in roots]
    (aggr_root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "total_episodes": sum(i["total_episodes"] for i in infos),
                "total_frames": sum(i["total_frames"] for i in infos),
            }
        )
    )


def collect(cfg, worker=0, workers=1, **kwargs):
    plan = short(build_plan(cfg, DEFS))
    _, env = make_env()
    return run_collection(
        cfg,
        DEFS,
        env,
        worker_id=worker,
        num_workers=workers,
        plan=plan,
        writer_factory=FakeWriter,
        log=lambda _: None,
        **kwargs,
    )


# ------------------------------------------------------------------------ plan
def test_plan_counts_train_only_and_unique(tmp_path):
    cfg = config(tmp_path)
    plan = build_plan(cfg, DEFS)
    assert len(plan) == 8
    counts = {}
    for item in plan:
        counts[(item.source, item.spec.task)] = (
            counts.get((item.source, item.spec.task), 0) + 1
        )
        assert check_spec(DEFS, item.spec) == []
        assert item.spec.split == "train" and item.spec.paraphrase.startswith("train:")
    assert counts == {("rand", "reach"): 4, ("rand", "escape"): 2, ("clip", "reach"): 2}
    assert len({i.spec.episode_seed for i in plan}) == 8
    assert [i.action_free for i in plan if i.source == "clip"] == [True, True]
    # deterministic, and interleaved: the first three items are one per (source, task) group
    assert [i.spec for i in build_plan(cfg, DEFS)] == [i.spec for i in plan]
    assert len({(i.source, i.spec.task) for i in plan[:3]}) == 3


def test_plan_rejects_other_splits_and_tasks(tmp_path):
    with pytest.raises(ValueError, match="train split"):
        config(tmp_path, split="val")
    cfg = config(
        tmp_path,
        sources=[
            {"name": "a", "policy": "random", "episodes": {"reach_then_escape": 1}}
        ],
    )
    with pytest.raises(ValueError, match="not in the train split|not collectable"):
        build_plan(cfg, DEFS)


def test_plan_opponent_filter_and_int_counts(tmp_path):
    cfg = config(
        tmp_path,
        tasks=["reach", "control_clinch"],
        sources=[
            {"name": "a", "policy": "random", "episodes": 3, "opponents": ["passive"]}
        ],
    )
    plan = build_plan(cfg, DEFS)
    assert len(plan) == 6 and {i.spec.opponent for i in plan} == {"passive"}


def test_leakage_check_flags_every_axis():
    good = DEFS.sample("train", 0, 0)
    assert check_spec(DEFS, good) == []
    val = DEFS.sample("val", 0, 0)
    assert any("split" in p for p in check_spec(DEFS, val))
    assert any("range" in p for p in check_spec(DEFS, val))
    heldout = dataclasses.replace(good, opponent="scripted:rush")
    assert any("opponent" in p for p in check_spec(DEFS, heldout))
    heldout_pose = dataclasses.replace(good, preroll_turns=6)
    assert any("pose" in p for p in check_spec(DEFS, heldout_pose))
    test_instr = dataclasses.replace(
        good, instruction=DEFS.tasks[good.task].paraphrases["heldout"][0]
    )
    assert any("paraphrase" in p for p in check_spec(DEFS, test_instr))


# -------------------------------------------------------------------- policies
def test_policies_ids_noise_and_mixture():
    obs = make_observation()
    base = build_policy("hold")
    noisy = build_policy({"policy": "hold", "epsilon": 0.0, "noise": 0.5})
    noisy.reset(3)
    first = [noisy(obs).copy() for _ in range(5)]
    noisy.reset(3)
    assert all((a == b).all() for a, b in zip(first, [noisy(obs) for _ in range(5)]))
    hold = base(obs)
    assert any((a != hold).any() for a in first)
    assert noisy.policy_id == "hold+eps0+noise0.5"
    mix = build_policy(
        {"mix": [{"policy": "hold", "weight": 1}, {"policy": "random", "weight": 1}]}
    )
    chosen = set()
    for seed in range(20):
        mix.reset(seed)
        chosen.add(mix.component)
    assert chosen == {"hold", "random"} and mix.policy_id.startswith("mix(")
    with pytest.raises(ValueError):
        build_policy({"policy": "hold", "mix": []})


# --------------------------------------------------------------------- episode
def test_collect_episode_alignment_and_success():
    fake, env = make_env(contact_after(3))
    spec = dataclasses.replace(
        DEFS.sample("train", 0, 0),
        task="reach",
        task_id=0,
        horizon_turns=TURNS,
        preroll_turns=0,
        opponent="passive",
    )
    policy = build_policy("random")
    episode = collect_episode(env, spec, policy)
    assert (
        episode.frames.shape == (TURNS, SIZE, SIZE, 3)
        and episode.frames.dtype == np.uint8
    )
    assert episode.states.shape == (TURNS, STATE_DIM) and episode.actions.shape == (
        TURNS,
        22,
    )
    # the action stored at turn t is the one the env received at turn t
    assert [h["player_0"].tolist() for h in fake.history] == episode.actions.tolist()
    # frames differ turn to turn; state t is the flattened observation before the action
    assert len({f.tobytes() for f in episode.frames}) == TURNS
    assert np.array_equal(episode.states[0], flatten_observation(make_observation()))
    # contact holds from turn 3 for 2 observations: success latches at turn 4
    assert episode.info["success"] and episode.info["success_turn"] == 4
    assert episode.success_flags.tolist() == [False] * 4 + [True] * 2
    assert episode.instruction == spec.instruction


# ------------------------------------------------------------------- the loop
def test_run_collection_records_and_sidecar(tmp_path):
    cfg = config(tmp_path)
    summary = collect(cfg)
    assert summary["episodes"] == 8 and summary["stop"] == "complete"
    records = read_jsonl(cfg.out / "shards" / "w0" / SIDECAR)
    assert len(records) == 8
    clips = [r for r in records if r["dataset"] == CLIPS]
    demos = [r for r in records if r["dataset"] == DEMOS]
    assert len(clips) == 2 and len(demos) == 6
    r = demos[0]
    for key in (
        "spec",
        "task",
        "opponent",
        "pose",
        "seed",
        "policy_id",
        "success",
        "success_turn",
        "label",
        "split",
        "source",
        "instruction",
        "turns",
    ):
        assert key in r
    assert r["split"] == "train" and r["label"] == "failure" and r["turns"] == TURNS
    assert r["spec"]["paraphrase"].startswith("train:")
    # segments of 3 episodes: demos 3 + 3, clips 2
    segments = sorted(
        p.name for p in (cfg.out / "shards" / "w0" / "segments").iterdir()
    )
    assert len(segments) == 3 and sum(s.startswith("demos") for s in segments) == 2
    # clips store zero actions and has_action 0
    clip_file = next((cfg.out / "shards" / "w0" / "segments").glob("clips-*/ep0.npz"))
    data = np.load(clip_file)
    assert (data["actions"] == 0).all() and (data["info.has_action"] == 0).all()
    demo_file = next((cfg.out / "shards" / "w0" / "segments").glob("demos-*/ep0.npz"))
    assert np.load(demo_file)["info.has_action"].min() == 1


def test_resume_skips_finished_seeds(tmp_path):
    cfg = config(tmp_path)
    first = collect(cfg, max_episodes=5)
    assert first["episodes"] == 5 and first["stop"] == "max_episodes"
    assert len(finished_seeds(cfg.out)) == 5
    second = collect(cfg)
    assert second["episodes"] == 3 and second["skipped_done"] == 5
    third = collect(cfg)
    assert third["episodes"] == 0
    records = read_jsonl(cfg.out / "shards" / "w0" / SIDECAR)
    assert len({r["seed"] for r in records}) == 8


def test_crash_leaves_no_partial_segment(tmp_path):
    cfg = config(tmp_path)

    class Boom(FakeWriter):
        def write_episode(self, *a, **k):
            if self.episodes == 1:
                raise RuntimeError("disk full")
            return super().write_episode(*a, **k)

    plan = short(build_plan(cfg, DEFS))
    _, env = make_env()
    with pytest.raises(RuntimeError):
        run_collection(
            cfg, DEFS, env, plan=plan, writer_factory=Boom, log=lambda _: None
        )
    # the finally clause finalised the open segment holding the one good episode...
    assert len(read_jsonl(cfg.out / "shards" / "w0" / SIDECAR)) >= 1
    # ...and a stray unfinished segment (killed process) is removed on restart
    stray = cfg.out / "shards" / "w0" / "segments" / "demos-0099"
    stray.mkdir()
    summary = collect(cfg)
    assert not stray.exists()
    assert len(finished_seeds(cfg.out)) == 8 and summary["stop"] == "complete"


def test_disk_guard_stops_collection(tmp_path):
    cfg = config(
        tmp_path, max_disk_gb=1e-5
    )  # about 10 KB; one 6-turn episode is larger
    summary = collect(cfg)
    assert summary["stop"] == "disk_cap" and 1 <= summary["episodes"] < 8
    assert summary["remaining"] == 8 - summary["episodes"]


def test_workers_split_the_plan_and_merge(tmp_path):
    cfg = config(tmp_path)
    for worker in range(3):
        collect(cfg, worker=worker, workers=3)
    seeds = [
        r["seed"]
        for w in range(3)
        for r in read_jsonl(cfg.out / "shards" / f"w{w}" / SIDECAR)
    ]
    assert sorted(seeds) == sorted(i.spec.episode_seed for i in build_plan(cfg, DEFS))
    manifest = merge_shards(
        cfg.out, name="t", aggregate=fake_aggregate, log=lambda _: None
    )
    assert manifest["datasets"][DEMOS]["episodes"] == 6
    assert manifest["datasets"][CLIPS]["episodes"] == 2
    merged = read_jsonl(cfg.out / "merged" / SIDECAR)
    for kind in (DEMOS, CLIPS):
        indices = [r["episode_index"] for r in merged if r["dataset"] == kind]
        assert indices == list(range(len(indices)))
    with pytest.raises(FileExistsError):
        merge_shards(cfg.out, name="t", aggregate=fake_aggregate)
    merge_shards(
        cfg.out,
        name="t",
        aggregate=fake_aggregate,
        force=True,
        delete_shards=True,
        log=lambda _: None,
    )
    assert not (cfg.out / "shards").exists()


def test_merge_rejects_duplicate_seeds(tmp_path):
    cfg = config(tmp_path)
    collect(cfg)
    src = cfg.out / "shards" / "w0"
    import shutil

    shutil.copytree(src, cfg.out / "shards" / "w1")
    with pytest.raises(ValueError, match="twice"):
        merge_shards(cfg.out, name="t", aggregate=fake_aggregate, log=lambda _: None)


# ------------------------------------------------------------------------ card
def test_card_contents_and_leakage(tmp_path):
    cfg = config(tmp_path)
    collect(cfg)
    merge_shards(cfg.out, name="t", aggregate=fake_aggregate, log=lambda _: None)
    card = card_module.build_card(cfg.out)
    assert card["episodes"] == 8 and card["frames"] == 8 * TURNS
    assert card["demo_episodes"] == 6 and card["clip_episodes"] == 2
    assert (
        card["by_task"]["reach"]["episodes"] == 6
        and card["by_task"]["escape"]["episodes"] == 2
    )
    assert card["by_source"]["rand"]["episodes"] == 6
    assert card["by_kind"]["clips"]["episodes"] == 2
    assert sum(g["episodes"] for g in card["by_opponent"].values()) == 8
    assert set(card["by_pose"]) <= {"preroll0", "preroll2"}
    assert card["capture_size"] == SIZE and card["size_bytes"]["total"] > 0
    assert card["integrity"]["ok"], card["integrity"]
    assert card["integrity"]["episodes_in_other_split_seed_ranges"]["val"] == 0
    assert card["split_policy"]["definitions_version"] == "2.0.0"
    assert "letorifighter" in card["versions"] and "torille" in card["versions"]
    text = card_module.render_card(card)
    for needle in (
        "## Integrity check",
        "PASS",
        "By task",
        "reach",
        "Toribash build id",
        "LeRobot",
    ):
        assert needle in text
    path, _ = card_module.write_card(cfg.out)
    assert path.exists() and (cfg.out / "card.json").exists()


def test_card_integrity_fails_on_val_episode(tmp_path):
    cfg = config(tmp_path)
    collect(cfg)
    sidecar = cfg.out / "shards" / "w0" / SIDECAR
    records = read_jsonl(sidecar)
    val = DEFS.sample("val", 3, 0).to_dict()
    records[0]["spec"] = val
    sidecar.write_text("".join(json.dumps(r) + "\n" for r in records))
    card = card_module.build_card(cfg.out)
    assert not card["integrity"]["ok"]
    assert card["integrity"]["episodes_in_other_split_seed_ranges"]["val"] == 1
    assert any("val" in v for v in card["integrity"]["violations"])
    assert "FAIL" in card_module.render_card(card)
