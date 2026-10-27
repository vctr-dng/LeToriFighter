"""Trivial-baseline success rates on the task splits (documented in docs/m3-tasks-eval.md).

Run: flock /tmp/toribash-game.lock uv run python scripts/m3_baselines.py
"""

import json
from pathlib import Path

from letorifighter.eval import evaluate, make_policy
from letorifighter.tasks import LeToriTaskEnv

OUT = Path(__file__).resolve().parents[1] / "results" / "m3_baselines.json"
RUNS = [
    ("val", "hold", 50),
    ("val", "random", 50),
    ("val", "scripted:advance_grab", 50),
    ("val", "scripted:rush", 50),
    ("test_paraphrase", "random", 25),
    ("test_opponent", "random", 25),
    ("test_pose", "random", 25),
    ("test_composition", "random", 24),
]


def main() -> None:
    results = []
    env = LeToriTaskEnv(split="val")
    try:
        for split, policy, episodes in RUNS:
            report = evaluate(
                make_policy(policy), env, split, episodes, seed=0, policy_name=policy
            )
            report.pop("episodes")
            results.append(report)
            print(
                split,
                policy,
                round(report["overall"]["success_rate"], 2),
                {k: round(v["success_rate"], 2) for k, v in report["by_task"].items()},
                flush=True,
            )
    finally:
        env.close()
    OUT.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
