"""Collect with the real LeRobot writer (mp4 video), merge shards with LeRobot's aggregate,
and read the result back through `VLADataset`. Needs `--group vla`; skipped otherwise."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lerobot")

from test_collect import DEFS, TURNS, config, make_env, short

from letorifighter.data.card import build_card
from letorifighter.data.collect import build_plan, run_collection
from letorifighter.data.shards import CLIPS, DEMOS, SIDECAR, merge_shards, read_jsonl
from letorifighter.vla.data import VLADataset

SIZE = 32  # config() uses 16; AV1 needs a little more


class FakeText:
    def __init__(self):
        self.seen = []

    def __call__(self, text):
        self.seen.append(text)
        return torch.arange(5) + len(text), torch.ones(5, dtype=torch.bool)


@pytest.fixture(scope="module")
def collected(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("roundtrip")
    cfg = config(tmp, capture_size=SIZE, segment_episodes=2)
    fake, env = make_env()
    fake.capture_size = SIZE
    plan = short(build_plan(cfg, DEFS))
    run_collection(cfg, DEFS, env, plan=plan, log=lambda _: None)
    manifest = merge_shards(cfg.out, name=cfg.name, log=lambda _: None)
    return cfg, manifest, fake


def test_merged_dataset_matches_sidecar(collected):
    cfg, manifest, _ = collected
    assert manifest["datasets"][DEMOS]["episodes"] == 6
    assert manifest["datasets"][CLIPS]["episodes"] == 2
    assert manifest["datasets"][DEMOS]["frames"] == 6 * TURNS
    records = read_jsonl(cfg.out / "merged" / SIDECAR)
    assert [r["episode_index"] for r in records if r["dataset"] == DEMOS] == list(
        range(6)
    )
    card = build_card(cfg.out)
    assert card["integrity"]["ok"], card["integrity"]
    assert card["size_bytes"]["merged"] > 0
    mp4s = list((cfg.out / "merged" / DEMOS / "videos").rglob("*.mp4"))
    assert mp4s  # video storage, not PNG


def test_vla_dataset_reads_what_we_wrote(collected):
    cfg, _, _ = collected
    records = [
        r for r in read_jsonl(cfg.out / "merged" / SIDECAR) if r["dataset"] == DEMOS
    ]
    text = FakeText()
    ds = VLADataset(
        cfg.out / "merged" / DEMOS, 4, repo_id="local/t-demos", text_encoder=text
    )
    assert len(ds) == 6 * TURNS and ds.state_dim == 300
    item = ds[0]
    assert item["observation.images.front"].shape == (3, SIZE, SIZE)
    assert item["action_states"].shape == (4, 22) and item["action_states"].max() <= 3
    assert item["action_is_pad"].tolist() == [False] * 4
    frames = torch.stack([ds[i]["observation.images.front"] for i in range(TURNS)])
    assert (
        frames.std(dim=0).mean() > 0.01
    )  # decoded frames are not constant over the episode
    # the language is the instruction of the episode, a train paraphrase
    train_texts = {
        p for t in DEFS.tasks.values() for p in t.paraphrases.get("train", [])
    }
    assert {r["instruction"] for r in records} <= train_texts
    for i in range(0, len(ds), TURNS):
        ds[i]
    assert set(text.seen) <= train_texts and set(text.seen)
    # last frame of episode 0: chunk padding starts after it
    assert ds[TURNS - 1]["action_is_pad"].tolist() == [False, True, True, True]


def test_recorded_actions_equal_played_actions(collected):
    cfg, _, fake = collected
    ds = VLADataset(
        cfg.out / "merged" / DEMOS, 1, repo_id="local/t-demos", text_encoder=FakeText()
    )
    records = [
        r for r in read_jsonl(cfg.out / "merged" / SIDECAR) if r["dataset"] == DEMOS
    ]
    # find a demo episode in the env history by matching the action sequence of episode 0
    played = np.array([h["player_0"] for h in fake.history])
    stored = np.stack([ds[i]["action_states"][0].numpy() for i in range(len(ds))])
    first = stored[:TURNS]
    starts = [
        s
        for s in range(len(played) - TURNS + 1)
        if np.array_equal(played[s : s + TURNS], first)
    ]
    assert starts, "stored actions of episode 0 were never played"
    assert len(records) == 6


def test_clips_have_no_actions(collected):
    cfg, _, _ = collected
    ds = VLADataset(
        cfg.out / "merged" / CLIPS, 2, repo_id="local/t-clips", text_encoder=FakeText()
    )
    assert len(ds) == 2 * TURNS
    assert int(ds.all_actions.max()) == 0
    table = ds.lerobot.hf_dataset.with_format("numpy")
    assert float(np.asarray(table["info.has_action"]).max()) == 0.0
