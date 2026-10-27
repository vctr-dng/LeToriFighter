"""Steps a ToriLLE env from LeToriFighter. Run with `pytest -m native`.

The game runs in direct-launch mode on an Xvfb `InstancePool` instance (see the
`xvfb_instance` fixture), in a subprocess without `DISPLAY`; nothing appears on the desktop.
"""

import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.native

SCRIPT = textwrap.dedent(
    """
    import json
    import os

    import gymnasium as gym

    from letorifighter.direct import check_env_kwargs

    import torille.envs  # noqa: F401

    kwargs = check_env_kwargs(json.loads(os.environ["LETORI_ENV_KWARGS"]))
    env = gym.make("Toribash-DestroyUke-v0", matchframes=100, turnframes=5, **kwargs)
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


def test_letorifighter_steps_torille_env(game_env):
    result = subprocess.run(
        [sys.executable, "-I", "-c", SCRIPT],
        env=game_env,
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
    assert "STEPS " in result.stdout
