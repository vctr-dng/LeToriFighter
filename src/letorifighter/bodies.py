"""Names for the indices of Toribash limbs, joints and action slots.

The orders are those of ToriLLE's state and action structure
(`ToriLLE/docs/protocol/state_structure.md`, `action_structure.md`) and of the
per-agent observation of `torille.envs.TwoPlayerToriEnv`.
"""

NUM_LIMBS = 21
NUM_JOINTS = 20
NUM_CONTROLLABLES = 22
NUM_JOINT_STATES = 4

LIMBS = (
    "head",
    "breast",
    "chest",
    "stomach",
    "groin",
    "r_pecs",
    "r_biceps",
    "r_triceps",
    "l_pecs",
    "l_biceps",
    "l_triceps",
    "r_hand",
    "l_hand",
    "r_butt",
    "l_butt",
    "r_thigh",
    "l_thigh",
    "l_leg",
    "r_leg",
    "r_foot",
    "l_foot",
)
LIMB = {name: index for index, name in enumerate(LIMBS)}

JOINTS = (
    "neck",
    "chest",
    "lumbar",
    "abs",
    "r_pec",
    "r_shoulder",
    "r_elbow",
    "l_pec",
    "l_shoulder",
    "l_elbow",
    "r_wrist",
    "l_wrist",
    "r_glute",
    "l_glute",
    "r_hip",
    "l_hip",
    "r_knee",
    "l_knee",
    "r_ankle",
    "l_ankle",
)
# Action slots: the 20 joints, then the left and the right grip.
SLOTS = (*JOINTS, "l_grip", "r_grip")
SLOT = {name: index for index, name in enumerate(SLOTS)}

# Action values of `TwoPlayerToriEnv` (the game's states 1..4 minus 1).
EXTEND, CONTRACT, HOLD, RELAX = 0, 1, 2, 3
# Grip values 0 and 1 release, 2 and 3 grip.
GRIP_OFF, GRIP_ON = 0, 2
