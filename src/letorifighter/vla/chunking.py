"""Action chunk indexing over a flat frame table (numpy only)."""

import numpy as np


def chunk_indices(episode_index: np.ndarray, chunk_size: int):
    """Indices of the next `chunk_size` frames of every frame.

    `episode_index` is the per-frame episode id of a table whose episodes are
    contiguous and ordered by frame. Returns `(indices, is_pad)`, both `(N, k)`:
    row i holds i, i+1, ..., i+k-1 clamped to the last frame of i's episode, and
    `is_pad` is True where the position lies beyond the episode end (the clamped
    index repeats the last action, the loss ignores it).
    """
    episode_index = np.asarray(episode_index)
    n = len(episode_index)
    if n == 0:
        return np.zeros((0, chunk_size), np.int64), np.zeros((0, chunk_size), bool)
    boundaries = np.flatnonzero(np.diff(episode_index) != 0)
    if len(np.unique(episode_index)) != len(boundaries) + 1:
        raise ValueError("episodes must be contiguous in the frame table")
    last_of_episode = np.empty(n, np.int64)
    start = 0
    for end in [*(boundaries + 1), n]:
        last_of_episode[start:end] = end - 1
        start = end
    raw = np.arange(n)[:, None] + np.arange(chunk_size)[None, :]
    return np.minimum(raw, last_of_episode[:, None]), raw > last_of_episode[:, None]
