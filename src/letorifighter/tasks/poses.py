"""Initial poses.

In Toribash the start of a game is deterministic (both fighters stand 1.3 apart,
measured; see `ToriLLE/docs/envs.md`), so "an unseen initial pose" has to be
produced. The mechanism here is a random pre-roll: after `reset`, both fighters
play `preroll_turns` turns of uniformly random actions drawn from
`np.random.default_rng(pose_seed)` (player_0's action first, then player_1's,
each turn). The game is a deterministic function of its actions, so the pose
reached after the pre-roll is a deterministic function of
`(preroll_turns, pose_seed)`. Pose pools differ in the pre-roll length and use
disjoint seeds, so the held-out poses (longer pre-roll, unseen seeds) are
further from the canonical start than anything trained on.

Whatever happens during the pre-roll does not count for the task: detectors
start with the observation after it, injuries are measured from that point, and
the `match_frame` the policy sees starts at zero there.
"""

import numpy as np

from letorifighter.bodies import NUM_CONTROLLABLES, NUM_JOINT_STATES

AGENTS = ("player_0", "player_1")


def random_actions(rng: np.random.Generator) -> dict:
    return {
        agent: rng.integers(0, NUM_JOINT_STATES, NUM_CONTROLLABLES).astype(np.int64)
        for agent in AGENTS
    }


def preroll(env, observations: dict, turns: int, pose_seed: int):
    """
    Play `turns` random turns on an env that was just reset.

    Returns `(observations, infos, ended)`; `infos` is `None` for zero turns and
    `ended` is True if the game finished during the pre-roll.
    """
    rng = np.random.default_rng(pose_seed)
    infos = None
    for _ in range(turns):
        observations, _, _, _, infos = env.step(random_actions(rng))
        if not env.agents:
            return observations, infos, True
    return observations, infos, False
