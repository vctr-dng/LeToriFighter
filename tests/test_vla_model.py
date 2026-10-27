"""Policy classes on a tiny random SmolVLA (CPU, no downloads)."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lerobot")
pytest.importorskip("transformers")

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from letorifighter.vla.heads import DiscreteHeadPolicy
from letorifighter.vla.levels import states_to_levels
from letorifighter.vla.model import predict_states

K = 4


def test_relaxed_policy_shapes(tiny_smolvla_config, tiny_batch):
    policy = SmolVLAPolicy(tiny_smolvla_config(K))
    batch = tiny_batch(K)
    loss, _ = policy(batch)
    assert loss.ndim == 0 and torch.isfinite(loss)
    states = predict_states(policy, batch, "relaxed")
    assert states.shape == (2, K, 22)
    assert states.dtype == torch.long and states.min() >= 0 and states.max() <= 3


def test_discrete_head_replaces_flow_projections(tiny_smolvla_config, tiny_batch):
    policy = DiscreteHeadPolicy(tiny_smolvla_config(K))
    model = policy.model
    for gone in ("action_in_proj", "action_out_proj", "action_time_mlp_in"):
        assert not hasattr(model, gone)
    assert model.logit_head.out_features == 22 * 4
    assert model.queries.shape[0] == K
    logits = policy._logits(tiny_batch(K))
    assert logits.shape == (2, K, 22, 4)
    states = predict_states(policy, tiny_batch(K), "ce")
    assert states.shape == (2, K, 22) and states.dtype == torch.long


def test_discrete_loss_starts_near_chance_and_ignores_padding(
    tiny_smolvla_config, tiny_batch
):
    torch.manual_seed(0)
    policy = DiscreteHeadPolicy(tiny_smolvla_config(K))
    batch = tiny_batch(K)
    loss, _ = policy(batch)
    assert abs(loss.item() - 1.386) < 0.5  # ln 4 for uniform logits
    batch["action_is_pad"][:] = True
    loss, _ = policy(batch)
    assert loss.item() == 0.0
    # padded positions never contribute: changing their targets changes nothing
    batch = tiny_batch(K)
    batch["action_is_pad"][:, 2:] = True
    reference, _ = policy(batch)
    batch["action_states"][:, 2:] = (batch["action_states"][:, 2:] + 1) % 4
    changed, _ = policy(batch)
    assert torch.isclose(reference, changed)


def test_discrete_head_fits_a_fixed_batch(tiny_smolvla_config, tiny_batch):
    torch.manual_seed(0)
    policy = DiscreteHeadPolicy(tiny_smolvla_config(K))
    batch = tiny_batch(K)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-3)
    first = None
    for _ in range(80):
        loss, _ = policy(batch)
        first = first if first is not None else loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    assert loss.item() < 0.3 < first
    accuracy = (
        (predict_states(policy, batch, "ce") == batch["action_states"]).float().mean()
    )
    assert accuracy > 0.9


def test_relaxed_targets_are_levels(tiny_batch):
    batch = tiny_batch(K)
    torch.testing.assert_close(
        batch["action"], states_to_levels(batch["action_states"]).float()
    )
