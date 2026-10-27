"""Writes synthetic episodes, reloads them with LeRobot, checks the content."""

import numpy as np
import pytest

pytest.importorskip("lerobot")

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from letorifighter.data.lerobot_writer import LeRobotEpisodeWriter

H = W = 32
STATE_DIM = 5
LENGTHS = [4, 6]
TASKS = ["reach the head", "stay on your feet"]


def make_episode(rng, length, language):
    return {
        "frames": rng.integers(0, 256, (length, H, W, 3), dtype=np.uint8),
        "states": rng.normal(size=(length, STATE_DIM)).astype(np.float32),
        "actions": rng.integers(0, 4, (length, 22)),
        "language": language,
        "extras": {"progress": rng.normal(size=(length, 2)).astype(np.float32)},
    }


@pytest.mark.parametrize("use_videos", [False, True])
def test_roundtrip(tmp_path, use_videos):
    rng = np.random.default_rng(0)
    episodes = [make_episode(rng, n, t) for n, t in zip(LENGTHS, TASKS)]
    root = tmp_path / "ds"
    with LeRobotEpisodeWriter(
        root,
        "local/test",
        fps=10,
        image_shape=(H, W, 3),
        state_dim=STATE_DIM,
        extra_features={"progress": ("float32", (2,))},
        use_videos=use_videos,
    ) as writer:
        writer.write_episodes(episodes)

    ds = LeRobotDataset("local/test", root=root)
    assert ds.meta.info["codebase_version"] == "v3.0"
    assert ds.num_episodes == 2
    assert ds.num_frames == sum(LENGTHS)
    assert sorted(ds.meta.tasks.index) == sorted(TASKS)

    starts = np.cumsum([0, *LENGTHS])
    for ep, (episode, start) in enumerate(zip(episodes, starts)):
        meta = ds.meta.episodes[ep]
        assert meta["length"] == LENGTHS[ep]
        assert meta["tasks"] == [TASKS[ep]]
        for t in range(LENGTHS[ep]):
            item = ds[int(start) + t]
            assert item["episode_index"].item() == ep
            assert item["frame_index"].item() == t
            assert item["task"] == TASKS[ep]
            img = item["observation.images.front"]
            assert tuple(img.shape) == (3, H, W)
            assert str(img.dtype) == "torch.float32"
            assert item["observation.state"].shape == (STATE_DIM,)
            np.testing.assert_allclose(
                item["observation.state"].numpy(), episode["states"][t], rtol=1e-6
            )
            action = item["action"].numpy()
            assert action.shape == (22,)
            np.testing.assert_array_equal(action, episode["actions"][t])
            np.testing.assert_allclose(
                item["progress"].numpy(), episode["extras"]["progress"][t]
            )
            if not use_videos:
                # PNG is lossless
                expected = episode["frames"][t].transpose(2, 0, 1) / 255
                np.testing.assert_allclose(img.numpy(), expected, atol=1e-6)


def test_rejects_bad_actions(tmp_path):
    rng = np.random.default_rng(0)
    bad = make_episode(rng, 3, "x")
    bad["actions"] = bad["actions"] + 4
    writer = LeRobotEpisodeWriter(
        tmp_path / "ds",
        "local/test",
        fps=10,
        image_shape=(H, W, 3),
        state_dim=STATE_DIM,
        extra_features={"progress": ("float32", (2,))},
        use_videos=False,
    )
    with pytest.raises(ValueError, match="0..3"):
        writer.write_episode(**bad)
