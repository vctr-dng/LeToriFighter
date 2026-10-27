"""Distances between named body parts, and the definition of "contact".

Toribash exposes limb centre positions and velocities, not contact flags, so
contact is defined by a distance threshold between the centres of named limbs.
All functions take the learner's own observation (`TwoPlayerToriEnv` per-agent
`Dict`) and use only distances, which do not depend on the frame the positions
are expressed in.

Calibration (`scripts/calibrate_contact.py`, output in
`results/contact_calibration.json`; 36 episodes of random and scripted
fighters, 1800 turn boundaries): bodies do not interpenetrate, so the distance
between two limb centres has a floor. For a hand against the other fighter's
head the floor is 0.213 (1st percentile 0.214, pile-up of grabs between 0.21 and
0.25); against breast, chest, stomach and groin the smallest distance seen was
0.35, 0.33, 0.40 and 0.39 (those limbs are bigger boxes, and hands touched them
rarely, so these minima are upper bounds on the true floors). The thresholds
are the floor plus a margin of about 0.1 for the box orientation and the
sampling at turn boundaries only: 0.30 for the head and 0.45 for the four trunk
parts. Damage is not a usable contact signal: in the same data a hand within
0.25 of the opponent coincided with opponent damage in 0 of 222 turns (a grab
does not hurt), against a 5.5% base rate per turn.

Limits: positions are sampled once per turn (`turnframes` frames), so a hand
that touches and leaves inside one turn is missed; the thresholds were fitted
with `turnframes=5` on the classic mod.
"""

import numpy as np

from letorifighter.bodies import LIMB

HANDS = ("r_hand", "l_hand")
TORSO = ("head", "breast", "chest", "stomach", "groin")

# Hand-to-limb contact threshold per target limb, in game units (metres).
CONTACT_THRESHOLD = {
    "head": 0.30,
    "breast": 0.45,
    "chest": 0.45,
    "stomach": 0.45,
    "groin": 0.45,
}


def distance(obs: dict, own_part: str, other_part: str) -> float:
    """Distance between a limb of the learner and a limb of the other fighter."""
    a = obs["self"]["positions"][LIMB[own_part]]
    b = obs["opponent"]["positions"][LIMB[other_part]]
    return float(np.linalg.norm(a - b))


def groin_distance(obs: dict) -> float:
    """Distance between the two groins (start of a game: 1.3)."""
    return distance(obs, "groin", "groin")


def hand_contacts(obs: dict, targets=TORSO) -> dict[str, bool]:
    """For each hand (`r_hand`, `l_hand`): is it in contact with any of `targets`?"""
    result = {}
    for hand in HANDS:
        result[hand] = any(
            distance(obs, hand, target) < CONTACT_THRESHOLD[target]
            for target in targets
        )
    return result


def reach_margin(obs: dict, targets=TORSO) -> float:
    """
    Smallest `distance - threshold` over hands and target limbs. Negative means
    contact; the magnitude says how close to touching (useful for logging).
    """
    return min(
        distance(obs, hand, target) - CONTACT_THRESHOLD[target]
        for hand in HANDS
        for target in targets
    )
