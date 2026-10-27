"""Plays real episodes against the opponents. Run with `pytest -m native`.

The game runs in direct-launch mode on an Xvfb `InstancePool` instance (see the
`xvfb_instance` fixture), in a subprocess without `DISPLAY`.
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

    import numpy as np

    from letorifighter.direct import check_env_kwargs
    from letorifighter.opponents import make_opponent
    from letorifighter.rollout import play_episode
    from torille.envs import TwoPlayerToriEnv

    kwargs = check_env_kwargs(json.loads(os.environ["LETORI_ENV_KWARGS"]))
    env = TwoPlayerToriEnv(matchframes=100, turnframes=5, **kwargs)
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


def test_full_episodes_against_each_opponent(game_env):
    result = subprocess.run(
        [sys.executable, "-I", "-c", SCRIPT],
        env=game_env,
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("EPISODE ") == 4
