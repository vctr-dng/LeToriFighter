"""`uv run python -m letorifighter.eval --split val --episodes 20 --policy random`.

Launches the game: wrap in `flock /tmp/toribash-game.lock`.
"""

import argparse
import json

import numpy as np

from letorifighter.eval.harness import evaluate
from letorifighter.tasks import LeToriTaskEnv


class RandomPolicy:
    def __init__(self) -> None:
        self.rng = np.random.default_rng(0)

    def reset(self, seed=None) -> None:
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs):
        return self.rng.integers(0, 4, 22)


class HoldPolicy:
    def __call__(self, obs):
        action = np.full(22, 2)
        action[20:] = 0
        return action


POLICIES = {"random": RandomPolicy, "hold": HoldPolicy}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy", choices=sorted(POLICIES), default="random")
    parser.add_argument("--out", default=None, help="JSON report path")
    parser.add_argument("--draw", action="store_true", help="draw the game")
    args = parser.parse_args()
    env = LeToriTaskEnv(split=args.split, draw_game=args.draw)
    try:
        report = evaluate(
            POLICIES[args.policy](),
            env,
            args.split,
            args.episodes,
            seed=args.seed,
            out_path=args.out,
            policy_name=args.policy,
        )
    finally:
        env.close()
    print(json.dumps(report["overall"], indent=1))


if __name__ == "__main__":
    main()
