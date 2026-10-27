"""`uv run python -m letorifighter.eval --split val --episodes 20 --policy random`.

Launches the game: wrap in `flock /tmp/toribash-game.lock`.
"""

import argparse
import json

from letorifighter.eval.harness import evaluate
from letorifighter.eval.policies import make_policy
from letorifighter.tasks import LeToriTaskEnv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--policy",
        default="random",
        help="random, hold, scripted:<behaviour> or vla:<checkpoint dir>",
    )
    parser.add_argument("--out", default=None, help="JSON report path")
    parser.add_argument("--draw", action="store_true", help="draw the game")
    args = parser.parse_args()
    env = LeToriTaskEnv(split=args.split, draw_game=args.draw)
    try:
        report = evaluate(
            make_policy(args.policy),
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
