"""Discrete diffusion head on a tiny random SmolVLA (CPU, no downloads)."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lerobot")
pytest.importorskip("transformers")

from letorifighter.vla.diffusion import (
    MASK_ID,
    DiffusionHeadPolicy,
    cosine_keep_masked,
    sample_mask,
)
from letorifighter.vla.model import predict_states

K = 4


def test_cosine_schedule_is_monotone_and_ends_at_zero():
    remaining = [cosine_keep_masked(88, t, 4) for t in range(1, 5)]
    assert remaining == sorted(remaining, reverse=True)
    assert remaining[0] < 88 and remaining[-1] == 0
    assert cosine_keep_masked(88, 12, 12) == 0


def test_sample_mask_respects_validity_and_minimum():
    torch.manual_seed(0)
    valid = torch.ones(64, K, 22, dtype=torch.bool)
    valid[:, 2:] = False  # padded steps never masked
    mask = sample_mask(valid)
    assert not (mask & ~valid).any()
    counts = mask.flatten(1).sum(1)
    assert counts.min() >= 1 and counts.max() <= 44
    assert counts.unique().numel() > 5  # ratios vary across samples
    none_valid = torch.zeros(2, K, 22, dtype=torch.bool)
    assert not sample_mask(none_valid).any()


def test_forward_and_decode_shapes(tiny_smolvla_config, tiny_batch):
    policy = DiffusionHeadPolicy(tiny_smolvla_config(K))
    policy.decode_steps = 3
    batch = tiny_batch(K)
    loss, _ = policy(batch)
    assert loss.ndim == 0 and torch.isfinite(loss)
    states = predict_states(policy, batch, "diffusion")
    assert states.shape == (2, K, 22) and states.dtype == torch.long
    assert states.max() < MASK_ID  # nothing left masked


def test_decode_matches_single_pass_logits_with_one_step(
    tiny_smolvla_config, tiny_batch
):
    """With one round everything is decoded from the all-mask input, i.e. the
    argmax of the logits of a fully masked chunk."""
    torch.manual_seed(0)
    policy = DiffusionHeadPolicy(tiny_smolvla_config(K)).eval()
    policy.decode_steps = 1
    batch = tiny_batch(K)
    decoded = policy.predict_action_chunk(dict(batch))
    masked = torch.full((2, K, 22), MASK_ID)
    with torch.no_grad():
        logits = policy.model.compute_logits(*policy._inputs(dict(batch)), masked)
    assert torch.equal(decoded, logits.argmax(-1))


def test_cached_decode_equals_uncached_logits(tiny_smolvla_config, tiny_batch):
    """The key/value cache path (used by decode) agrees with a full forward."""
    torch.manual_seed(1)
    policy = DiffusionHeadPolicy(tiny_smolvla_config(K)).eval()
    batch = tiny_batch(K)
    one = policy.model.decode(*policy._inputs(dict(batch)), steps=1)
    masked = torch.full((2, K, 22), MASK_ID)
    with torch.no_grad():
        full = policy.model.compute_logits(*policy._inputs(dict(batch)), masked)
    assert (one == full.argmax(-1)).float().mean() > 0.99


def test_fits_a_fixed_batch(tiny_smolvla_config, tiny_batch):
    torch.manual_seed(0)
    policy = DiffusionHeadPolicy(tiny_smolvla_config(K))
    policy.decode_steps = 3
    batch = tiny_batch(K)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-3)
    for _ in range(150):
        loss, _ = policy(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    accuracy = (
        (predict_states(policy, batch, "diffusion") == batch["action_states"])
        .float()
        .mean()
    )
    assert accuracy > 0.9
