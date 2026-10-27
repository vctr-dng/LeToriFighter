"""Trivial-baseline success rates on the task splits (documented in docs/m3-tasks-eval.md).

Run: env -u DISPLAY -u WAYLAND_DISPLAY uv run python scripts/m3_baselines.py (own Xvfb instance)
"""

import json
from pathlib import Path

from letorifighter.direct import own_instance
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
    with own_instance() as kwargs:
        env = LeToriTaskEnv(split="val", env_kwargs=kwargs)
        _run(env, results)
    OUT.write_text(json.dumps(results, indent=1) + "\n")


def _run(env, results) -> None:
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


if __name__ == "__main__":
    main()
