"""VLADataset on a small synthetic LeRobot v3 dataset (no model, no downloads)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lerobot")

from letorifighter.vla.data import VLADataset
from letorifighter.vla.levels import states_to_levels
from letorifighter.vla.synthetic import make_synthetic_dataset

LENGTH = 6
K = 4


class FakeText:
    """Stands in for the tokenizer: records the text it is asked to encode."""

    def __init__(self):
        self.seen = []

    def __call__(self, text):
        self.seen.append(text)
        tokens = torch.arange(5) + len(text)
        return tokens, torch.ones(5, dtype=torch.bool)


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    root = tmp_path_factory.mktemp("syn") / "ds"
    description = make_synthetic_dataset(
        root, episodes=4, length=LENGTH, image_size=32, state_dim=12, n_scenes=4
    )
    return root, description


def make_dataset(root, **kwargs):
    kwargs.setdefault("text_encoder", FakeText())
    return VLADataset(root, K, repo_id="local/synthetic", **kwargs)


def test_item_keys_shapes_and_levels(synthetic):
    root, _ = synthetic
    ds = make_dataset(root)
    assert len(ds) == 4 * LENGTH and ds.state_dim == 12
    item = ds[0]
    assert item["observation.images.front"].shape == (3, 32, 32)
    assert item["observation.images.front_padding_mask"].item() is True
    assert item["observation.state"].shape == (12,)
    assert item["observation.language.tokens"].shape == (5,)
    assert item["action"].shape == (K, 22) and item["action"].dtype == torch.float32
    assert item["action_states"].shape == (K, 22)
    torch.testing.assert_close(
        item["action"],
        torch.from_numpy(states_to_levels(item["action_states"].numpy())).float(),
    )
    assert not item["action_is_pad"].any()


def test_chunk_padding_at_episode_end(synthetic):
    root, _ = synthetic
    ds = make_dataset(root)
    last = ds[LENGTH - 1]  # last frame of episode 0
    assert last["action_is_pad"].tolist() == [False, True, True, True]
    assert (last["action_states"] == last["action_states"][0]).all()
    third = ds[LENGTH - 3]
    assert third["action_is_pad"].tolist() == [False, False, False, True]
    # the chunk continues with the following frames of the same episode
    nxt = ds[LENGTH - 2]
    assert torch.equal(third["action_states"][1], nxt["action_states"][0])


def test_actions_follow_the_synthetic_tables(synthetic):
    root, description = synthetic
    ds = make_dataset(root)
    tables = np.array(description["tables"])
    for i in (0, 7, 13):
        item = ds[i]
        assert item["action_states"][0].tolist() in tables.reshape(-1, 22).tolist()


def test_episode_selection_and_state_stats(synthetic):
    root, _ = synthetic
    train = make_dataset(root, episodes=[0, 1, 2])
    val = make_dataset(root, episodes=[3], state_stats=train.stats)
    assert len(train) == 3 * LENGTH and len(val) == LENGTH
    assert {int(train[i]["episode_index"]) for i in range(len(train))} == {0, 1, 2}
    assert all(int(val[i]["episode_index"]) == 3 for i in range(len(val)))
    np.testing.assert_array_equal(val.stats["mean"], train.stats["mean"])
    states = torch.stack([train[i]["observation.state"] for i in range(len(train))])
    assert states.mean().abs() < 0.2  # standardised


def test_state_only_masks_the_image(synthetic):
    root, _ = synthetic
    item = make_dataset(root, use_images=False)[0]
    assert item["observation.images.front_padding_mask"].item() is False
    assert item["observation.images.front"].abs().sum() == 0


def test_task_id_conditioning_replaces_the_instruction(synthetic):
    root, description = synthetic
    fake = FakeText()
    ds = make_dataset(
        root,
        conditioning="task_id",
        task_ids=description["task_ids"],
        text_encoder=fake,
    )
    item = ds[0]
    assert fake.seen == [f"task {int(item['task_id'])}"]
    with pytest.raises(ValueError, match="task_ids"):
        make_dataset(root, conditioning="task_id")
    fake = FakeText()
    make_dataset(root, text_encoder=fake)[0]
    assert fake.seen[0] in description["task_ids"]


def test_synthetic_is_deterministic(tmp_path):
    a = make_synthetic_dataset(
        tmp_path / "a",
        episodes=2,
        length=3,
        image_size=32,
        state_dim=4,
        n_scenes=4,
        seed=3,
    )
    b = make_synthetic_dataset(
        tmp_path / "b",
        episodes=2,
        length=3,
        image_size=32,
        state_dim=4,
        n_scenes=4,
        seed=3,
    )
    assert a["tables"] == b["tables"]
