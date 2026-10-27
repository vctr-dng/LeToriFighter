"""The opponent interface.

An opponent fills one slot of `torille.envs.TwoPlayerToriEnv`. It sees only
its own observation (the per-agent `Dict` of the environment, in the frame of
that agent) and returns an action in `MultiDiscrete([4] * 22)`.
"""

from typing import Protocol, runtime_checkable

import numpy as np

from letorifighter.bodies import NUM_CONTROLLABLES, NUM_JOINT_STATES


@runtime_checkable
class Opponent(Protocol):
    """Maps the opponent's own observation to an action, one turn at a time."""

    def reset(self, seed: int | None = None) -> None:
        """Start a new episode. All randomness of the opponent comes from `seed`."""
        ...

    def act(self, obs: dict) -> np.ndarray:
        """Return an int64 array of shape (22,) with values in 0..3."""
        ...


def check_action(action) -> np.ndarray:
    """Validate and return `action` as an int64 array in `MultiDiscrete([4] * 22)`."""
    array = np.asarray(action)
    if array.shape != (NUM_CONTROLLABLES,) or not np.issubdtype(
        array.dtype, np.integer
    ):
        raise ValueError(
            f"action must be an integer array of shape ({NUM_CONTROLLABLES},), "
            f"got shape {array.shape}, dtype {array.dtype}"
        )
    if array.min() < 0 or array.max() >= NUM_JOINT_STATES:
        raise ValueError(f"action values must be in 0..3, got {array.tolist()}")
    return array.astype(np.int64)
