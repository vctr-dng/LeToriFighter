"""Measure limb-to-limb distances between the two fighters to justify the contact threshold.

Toribash exposes limb centre positions only (no contact flags). Bodies cannot
interpenetrate, so the distance between two limb centres has a hard floor (the
sum of the half extents along the line joining them). This script plays random
and scripted fighters against each other, records the distance of every
cross-fighter limb pair at every turn boundary, and writes the per-pair minimum
and low percentiles, plus how often the opponent took damage in the same turn
as a hand came within a given distance of the opponent's torso.

Run: flock /tmp/toribash-game.lock uv run python scripts/calibrate_contact.py
"""

import json
from pathlib import Path

import numpy as np
from torille.envs import TwoPlayerToriEnv

from letorifighter.bodies import LIMB, LIMBS
from letorifighter.opponents import make_opponent
from letorifighter.rollout import play_episode

OUT = Path(__file__).resolve().parents[1] / "results" / "contact_calibration.json"
LEARNERS = ["random", "scripted:advance_grab", "scripted:rush"]
OPPONENTS = ["passive", "random", "scripted:advance_grab", "scripted:flail"]
SEEDS = [0, 1, 2]
TURNS = 50
TORSO = [LIMB[n] for n in ("head", "breast", "chest", "stomach", "groin")]
HANDS = [LIMB["r_hand"], LIMB["l_hand"]]


def main() -> None:
    env = TwoPlayerToriEnv(matchframes=TURNS * 6, turnframes=5)
    env.set_draw_game(False)
    matrices_list, hand_torso_list, damaged_list = [], [], []
    hand_part_list = []  # per turn: distance from the nearer hand to each torso part
    previous = {"injury": 0.0}
    try:
        for learner_name in LEARNERS:
            for opponent_name in OPPONENTS:
                for seed in SEEDS:
                    learner = (
                        make_opponent(learner_name, seed=seed)
                        if learner_name == "random"
                        else make_opponent(learner_name)
                    )
                    previous["injury"] = 0.0

                    def on_turn(observations, actions, infos):
                        obs = observations["player_0"]
                        own, other = (
                            obs["self"]["positions"],
                            obs["opponent"]["positions"],
                        )
                        matrices_list.append(
                            np.linalg.norm(own[:, None] - other[None], axis=2)
                        )
                        hand_part_list.append(
                            np.linalg.norm(
                                own[HANDS][:, None] - other[TORSO][None], axis=2
                            ).min(axis=0)
                        )
                        hand_torso_list.append(
                            np.linalg.norm(
                                own[HANDS][:, None] - other[TORSO][None], axis=2
                            ).min()
                        )
                        injury = infos["player_0"]["opponent_injury"]
                        damaged_list.append(injury > previous["injury"])
                        previous["injury"] = injury

                    learner.reset(seed)
                    play_episode(
                        env,
                        lambda o, p=learner: p.act(o),
                        make_opponent(opponent_name, seed=seed)
                        if opponent_name == "random"
                        else make_opponent(opponent_name),
                        seed=seed,
                        on_turn=on_turn,
                    )
    finally:
        env.close()
    matrices = np.array(matrices_list)  # (turns, 21, 21)
    hand_torso = np.array(hand_torso_list)
    hand_part = np.array(hand_part_list)
    damaged = np.array(damaged_list)
    result = {
        "episodes": len(LEARNERS) * len(OPPONENTS) * len(SEEDS),
        "turn_boundaries": len(matrices),
        "learners": LEARNERS,
        "opponents": OPPONENTS,
        "pair_min_distance": {
            f"{LIMBS[i]}-{LIMBS[j]}": round(float(matrices[:, i, j].min()), 3)
            for i in HANDS
            for j in range(21)
        },
        "nearer_hand_to_torso_part": {
            LIMBS[part]: {
                "min": round(float(hand_part[:, k].min()), 3),
                "p1": round(float(np.percentile(hand_part[:, k], 1)), 3),
                "p5": round(float(np.percentile(hand_part[:, k], 5)), 3),
            }
            for k, part in enumerate(TORSO)
        },
        "hand_to_torso_percentiles": {
            str(q): round(float(np.percentile(hand_torso, q)), 3)
            for q in (0, 1, 5, 10, 25, 50)
        },
        "hand_to_torso_threshold_stats": {
            str(d): {
                "turns_within": int((hand_torso < d).sum()),
                "fraction_with_opponent_damage": round(
                    float(damaged[hand_torso < d].mean()), 3
                )
                if (hand_torso < d).any()
                else None,
            }
            for d in (0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0)
        },
        "base_rate_opponent_damage": round(float(damaged.mean()), 3),
        "any_pair_min_distance": round(float(matrices.min()), 3),
        "any_pair_percentiles": {
            str(q): round(float(np.percentile(matrices.min(axis=(1, 2)), q)), 3)
            for q in (0, 1, 5, 25, 50)
        },
    }
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
