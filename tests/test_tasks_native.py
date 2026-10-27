"""Task wrapper and evaluation harness on the real game. Run with `pytest -m native`.

The game is launched in a subprocess wrapped in `flock /tmp/toribash-game.lock`
(one Toribash instance per machine), so do not wrap pytest itself in `flock`.
"""

import json
import shutil
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.native

LOCK_FILE = "/tmp/toribash-game.lock"

POSE_SCRIPT = textwrap.dedent(
    """
    import dataclasses

    import numpy as np

    from letorifighter.tasks import LeToriTaskEnv

    env = LeToriTaskEnv(split="val")
    base = env.definitions.sample("val", 0, 0)
    base = dataclasses.replace(base, task="reach", task_id=0, horizon_turns=2)

    def start(preroll, pose_seed):
        spec = dataclasses.replace(base, preroll_turns=preroll, pose_seed=pose_seed)
        obs, info = env.reset(options={"episode": spec})
        assert env.observation_space.contains(obs)
        assert obs["match_frame"][0] == 0.0
        positions = obs["self"]["positions"].copy()
        for _ in range(2):
            env.step(env.action_space.sample())
        return positions

    try:
        canonical = start(0, 0)
        a = start(4, 11)
        b = start(4, 11)
        c = start(4, 12)
        assert np.abs(a - b).max() == 0.0, "pre-roll is not deterministic"
        assert np.abs(a - c).max() > 0.01, "pose seeds give the same pose"
        assert np.abs(a - canonical).max() > 0.01, "pre-roll did not change the pose"
        print("POSES OK")
    finally:
        env.close()
    """
)

EVAL_SCRIPT = textwrap.dedent(
    """
    import json
    import sys

    import numpy as np

    from letorifighter.eval import evaluate
    from letorifighter.tasks import LeToriTaskEnv

    rng = np.random.default_rng(0)
    env = LeToriTaskEnv(split="val")
    try:
        report = evaluate(
            lambda obs: rng.integers(0, 4, 22),
            env,
            "val",
            episodes=3,
            seed=0,
            out_path=sys.argv[1],
            policy_name="random",
        )
    finally:
        env.close()
    print("EVAL OK", json.dumps(report["overall"]["success_rate"]))
    """
)


def run_game_script(script, *args):
    if shutil.which("flock") is None:
        pytest.skip("flock is not available")
    return subprocess.run(
        ["flock", LOCK_FILE, sys.executable, "-I", "-c", script, *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=900,
    )


def test_preroll_poses_are_deterministic_and_differ():
    result = run_game_script(POSE_SCRIPT)
    assert result.returncode == 0, result.stderr
    assert "POSES OK" in result.stdout


def test_short_evaluation_with_random_policy(tmp_path):
    out = tmp_path / "report.json"
    result = run_game_script(EVAL_SCRIPT, str(out))
    assert result.returncode == 0, result.stderr
    assert "EVAL OK" in result.stdout
    report = json.loads(out.read_text())
    assert report["overall"]["episodes"] == 3
    assert report["overall"]["latency_ms"]["turns"] > 100
    assert {"success_rate", "win_rate", "injury_delta_mean", "latency_ms"} <= set(
        report["overall"]
    )
    assert len(report["episodes"]) == 3
