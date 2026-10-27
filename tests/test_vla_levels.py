"""Levels, rounding and chunk indexing: numpy only, run without torch."""

import numpy as np
import pytest

from letorifighter.vla.chunking import chunk_indices
from letorifighter.vla.levels import (
    LEVELS,
    levels_to_states,
    round_to_levels,
    states_to_levels,
)


def test_levels_values():
    np.testing.assert_allclose(states_to_levels(np.arange(4)), LEVELS)
    assert LEVELS == (-1.0, -1 / 3, 1 / 3, 1.0)


def test_roundtrip_all_states():
    states = np.arange(4)
    np.testing.assert_array_equal(levels_to_states(states_to_levels(states)), states)


def test_rounding_nearest_and_clipping():
    levels = np.array([-5.0, -0.9, -0.5, -0.1, 0.0, 0.1, 0.5, 0.9, 7.0])
    expected = [0, 0, 1, 1, 1, 2, 2, 3, 3]
    # 0.0 is the midpoint of the two inner levels; either neighbour is acceptable
    got = levels_to_states(levels)
    assert list(got[:4]) == expected[:4]
    assert got[4] in (1, 2)
    assert list(got[5:]) == expected[5:]
    np.testing.assert_allclose(
        round_to_levels(np.array([-0.2, 0.34, 0.95])), [-1 / 3, 1 / 3, 1.0]
    )


def test_rounding_noise_robustness():
    rng = np.random.default_rng(0)
    states = rng.integers(0, 4, (1000, 22))
    noisy = states_to_levels(states) + rng.uniform(-0.3, 0.3, states.shape)
    np.testing.assert_array_equal(levels_to_states(noisy), states)  # half gap is 1/3


def test_list_input():
    assert list(levels_to_states([-1, 1])) == [0, 3]
    assert list(states_to_levels([0, 3])) == [-1, 1]


def test_torch_tensors_if_available():
    torch = pytest.importorskip("torch")
    x = torch.tensor([-1.0, -0.3, 0.4, 0.8])
    assert levels_to_states(x).long().tolist() == [0, 1, 2, 3]
    assert torch.allclose(
        states_to_levels(torch.arange(4)), torch.tensor(LEVELS), atol=1e-6
    )


def test_chunk_indices_pads_at_episode_end():
    episode = np.array([0, 0, 0, 1, 1, 2])
    index, pad = chunk_indices(episode, 3)
    assert index.tolist() == [
        [0, 1, 2],
        [1, 2, 2],
        [2, 2, 2],
        [3, 4, 4],
        [4, 4, 4],
        [5, 5, 5],
    ]
    assert pad.tolist() == [
        [False, False, False],
        [False, False, True],
        [False, True, True],
        [False, False, True],
        [False, True, True],
        [False, True, True],
    ]


def test_chunk_size_one_and_empty():
    index, pad = chunk_indices(np.array([0, 0, 1]), 1)
    assert index.tolist() == [[0], [1], [2]]
    assert not pad.any()
    index, pad = chunk_indices(np.array([], dtype=int), 4)
    assert index.shape == (0, 4)


def test_chunk_indices_rejects_interleaved_episodes():
    with pytest.raises(ValueError, match="contiguous"):
        chunk_indices(np.array([0, 1, 0]), 2)
