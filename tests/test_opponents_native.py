"""Plays real episodes against the opponents. Run with `pytest -m native`.

The game is launched in a subprocess wrapped in `flock /tmp/toribash-game.lock`
(one Toribash instance per machine).
"""

import shutil
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.native

LOCK_FILE = "/tmp/toribash-game.lock"

SCRIPT = textwrap.dedent(
    """
    import numpy as np

    from letorifighter.opponents import make_opponent
    from letorifighter.rollout import play_episode
    from torille.envs import TwoPlayerToriEnv

    env = TwoPlayerToriEnv(matchframes=100, turnframes=5)
    env.set_draw_game(False)
    rng = np.random.default_rng(0)
    try:
        for name in ("passive", "random", "scripted:advance_grab", "scripted:flail"):
            result = play_episode(
                env,
                lambda obs: rng.integers(0, 4, 22),
                make_opponent(name),
                seed=0,
            )
            assert result.turns > 0 and result.truncated, (name, result)
            assert len(result.learner_latency) == result.turns
            print("EPISODE", name, result.turns, result.winner)
    finally:
        env.close()
    """
)


def test_full_episodes_against_each_opponent():
    if shutil.which("flock") is None:
        pytest.skip("flock is not available")
    result = subprocess.run(
        ["flock", LOCK_FILE, sys.executable, "-I", "-c", SCRIPT],
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("EPISODE ") == 4
