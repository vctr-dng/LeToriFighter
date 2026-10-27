"""Passive and random opponents."""

import numpy as np

from letorifighter.bodies import (
    GRIP_OFF,
    HOLD,
    NUM_CONTROLLABLES,
    NUM_JOINT_STATES,
    NUM_JOINTS,
)


class PassiveOpponent:
    """Holds every joint and keeps both hands open, every turn."""

    def __init__(self) -> None:
        self._action = np.full(NUM_CONTROLLABLES, HOLD, dtype=np.int64)
        self._action[NUM_JOINTS:] = GRIP_OFF

    def reset(self, seed: int | None = None) -> None:
        del seed

    def act(self, obs: dict) -> np.ndarray:
        del obs
        return self._action.copy()


class RandomOpponent:
    """Draws every joint and grip uniformly, from a generator seeded in `reset`."""

    def __init__(self, seed: int | None = None) -> None:
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is None:
            seed = self._seed
        self._rng = np.random.default_rng(seed)

    def act(self, obs: dict) -> np.ndarray:
        del obs
        return self._rng.integers(0, NUM_JOINT_STATES, NUM_CONTROLLABLES).astype(
            np.int64
        )
