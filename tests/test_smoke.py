"""Steps a ToriLLE env from LeToriFighter. Run with `pytest -m native`.

Only one Toribash instance can run on the machine, so the game is launched in a
subprocess wrapped in `flock /tmp/toribash-game.lock`.
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
    import gymnasium as gym

    import torille.envs  # noqa: F401

    env = gym.make("Toribash-DestroyUke-v0", matchframes=100, turnframes=5)
    try:
        observation, _ = env.reset()
        assert env.observation_space.contains(observation)
        truncated = False
        steps = 0
        while not truncated:
            observation, _, terminated, truncated, info = env.step(
                env.action_space.sample()
            )
            steps += 1
            assert not terminated
        assert info["match_frame"] == 100
        print("STEPS", steps)
    finally:
        env.close()
    """
)


def test_letorifighter_steps_torille_env():
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
    assert "STEPS " in result.stdout
