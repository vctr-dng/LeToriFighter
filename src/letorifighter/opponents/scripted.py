"""Simple deterministic scripted opponents.

Each behaviour is a pure function of the turn counter (turns since `reset`)
and the opponent's own observation, so a scripted opponent has no randomness.
The joint directions were measured in the native game, see the notes on each
behaviour (the own frame has x lateral, y forward-negative and z up: at the
start the other fighter is at y = -1.3 m).

* `advance_grab`: pitch forward and reach for the other fighter, then clamp.
* `flail`: arms swing in a fixed travelling pattern, grips open and close.
* `guard`: crouch with the arms raised in front, hold and keep the hands open.
"""

from collections.abc import Callable

import numpy as np

from letorifighter.bodies import (
    CONTRACT,
    EXTEND,
    GRIP_OFF,
    GRIP_ON,
    HOLD,
    LIMB,
    NUM_CONTROLLABLES,
    NUM_JOINTS,
    SLOT,
)

# A hand closer than this (metres, in game units) to any limb of the other
# fighter counts as "reached": `advance_grab` then stops moving and clamps.
GRAB_DISTANCE = 0.35

_HANDS = [LIMB["r_hand"], LIMB["l_hand"]]


def _hold(grips: int = GRIP_OFF) -> np.ndarray:
    action = np.full(NUM_CONTROLLABLES, HOLD, dtype=np.int64)
    action[NUM_JOINTS:] = grips
    return action


def _set(action: np.ndarray, **joints: int) -> np.ndarray:
    for name, value in joints.items():
        action[SLOT[name]] = value
    return action


def hand_gap(obs: dict) -> float:
    """Smallest distance from either own hand to any limb of the other fighter."""
    hands = obs["self"]["positions"][_HANDS]
    others = obs["opponent"]["positions"]
    return float(np.linalg.norm(hands[:, None, :] - others[None, :, :], axis=2).min())


def advance_grab(turn: int, obs: dict) -> np.ndarray:
    """
    Measured in the game: abs extend leans forward, pecs contract swings the
    arms forward, glutes and hips contract swing the legs back, so the body
    pitches toward the other fighter with the hands in front. Once a hand is
    within `GRAB_DISTANCE` of the other fighter everything is held with the
    grips closed.
    """
    del turn
    if hand_gap(obs) < GRAB_DISTANCE:
        return _hold(GRIP_ON)
    return _set(
        _hold(GRIP_ON),
        r_pec=CONTRACT,
        l_pec=CONTRACT,
        abs=EXTEND,
        r_glute=CONTRACT,
        l_glute=CONTRACT,
        r_hip=CONTRACT,
        l_hip=CONTRACT,
    )


def flail(turn: int, obs: dict) -> np.ndarray:
    """
    Arms and torso swing in a fixed pattern: a 4-turn cycle of the pecs,
    shoulders and elbows with a phase offset between the two sides, the
    chest twisting every second turn, and the grips toggling every turn.
    """
    del obs
    action = _hold()
    phase = turn % 4
    right = EXTEND if phase < 2 else CONTRACT
    left = CONTRACT if phase < 2 else EXTEND
    _set(
        action,
        r_pec=right,
        l_pec=left,
        r_shoulder=left,
        l_shoulder=right,
        r_elbow=right,
        l_elbow=left,
        chest=EXTEND if turn % 2 == 0 else CONTRACT,
    )
    action[SLOT["l_grip"]] = GRIP_ON if turn % 2 == 0 else GRIP_OFF
    action[SLOT["r_grip"]] = GRIP_OFF if turn % 2 == 0 else GRIP_ON
    return action


def guard(turn: int, obs: dict) -> np.ndarray:
    """Pecs, shoulders and elbows contract (arms up and bent in front), knees contract."""
    del turn, obs
    return _set(
        _hold(),
        r_pec=CONTRACT,
        l_pec=CONTRACT,
        r_shoulder=CONTRACT,
        l_shoulder=CONTRACT,
        r_elbow=CONTRACT,
        l_elbow=CONTRACT,
        r_knee=CONTRACT,
        l_knee=CONTRACT,
    )


BEHAVIOURS: dict[str, Callable[[int, dict], np.ndarray]] = {
    "advance_grab": advance_grab,
    "flail": flail,
    "guard": guard,
}


class ScriptedOpponent:
    """Plays one of `BEHAVIOURS` by name. Deterministic: `seed` is ignored."""

    def __init__(self, behaviour: str = "advance_grab") -> None:
        if behaviour not in BEHAVIOURS:
            raise ValueError(
                f"unknown behaviour {behaviour!r}, expected one of {sorted(BEHAVIOURS)}"
            )
        self.behaviour = behaviour
        self._function = BEHAVIOURS[behaviour]
        self._turn = 0

    def reset(self, seed: int | None = None) -> None:
        del seed
        self._turn = 0

    def act(self, obs: dict) -> np.ndarray:
        action = self._function(self._turn, obs)
        self._turn += 1
        return action
