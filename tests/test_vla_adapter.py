"""Receding-horizon execution of the policy adapter (tiny model, CPU)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lerobot")
pytest.importorskip("transformers")

from letorifighter.vla.adapter import VLAPolicy
from letorifighter.vla.config import VLAConfig
from letorifighter.vla.heads import DiscreteHeadPolicy

K = 4
STATE_DIM = 8


class FakeText:
    def __call__(self, text):
        return torch.arange(6), torch.ones(6, dtype=torch.bool)


def make_policy(tiny_smolvla_config, n_action_steps, head="ce"):
    cfg = VLAConfig(
        head=head,
        chunk_size=K,
        n_action_steps=n_action_steps,
        use_images=False,
        device="cpu",
        bf16=False,
    )
    model = DiscreteHeadPolicy(tiny_smolvla_config(K, STATE_DIM))
    stats = {"mean": np.zeros(STATE_DIM), "std": np.ones(STATE_DIM)}
    return VLAPolicy(
        model,
        cfg,
        stats,
        text_encoder=FakeText(),
        state_fn=lambda obs: obs["state"],
    )


def obs(seed=0):
    rng = np.random.default_rng(seed)
    return {"state": rng.normal(size=STATE_DIM), "instruction": "reach", "task_id": 0}


@pytest.mark.parametrize(
    ("n_action_steps", "expected_passes"), [(1, 5), (2, 3), (4, 2)]
)
def test_forward_passes_follow_n_action_steps(
    tiny_smolvla_config, n_action_steps, expected_passes
):
    policy = make_policy(tiny_smolvla_config, n_action_steps)
    for i in range(5):
        action = policy(obs(i))
        assert action.shape == (22,) and action.dtype == np.int64
        assert 0 <= action.min() and action.max() <= 3
    assert policy.forward_passes == expected_passes


def test_queued_actions_are_the_chunk_prefix(tiny_smolvla_config):
    policy = make_policy(tiny_smolvla_config, 3)
    first = obs(0)
    chunk = policy.predict_chunk(first)
    assert chunk.shape == (K, 22)
    policy.reset()
    taken = [policy(first), policy(obs(1)), policy(obs(2))]  # later obs are ignored
    np.testing.assert_array_equal(np.stack(taken), chunk[:3])


def test_reset_clears_the_queue(tiny_smolvla_config):
    policy = make_policy(tiny_smolvla_config, 4)
    policy(obs(0))
    policy.reset(seed=1)
    policy(obs(1))
    assert policy.forward_passes == 2


def test_rejects_wrong_state_size_and_missing_image(tiny_smolvla_config):
    policy = make_policy(tiny_smolvla_config, 1)
    with pytest.raises(ValueError, match="state has"):
        policy({"state": np.zeros(3), "instruction": "x", "task_id": 0})
    policy.cfg.use_images = True
    with pytest.raises(ValueError, match="no"):
        policy(obs())


def test_invalid_n_action_steps(tiny_smolvla_config):
    with pytest.raises(ValueError, match="n_action_steps"):
        make_policy(tiny_smolvla_config, K + 1)
