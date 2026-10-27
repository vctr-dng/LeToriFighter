"""The display guard refuses anything that could open a window on the desktop."""

import pytest

from letorifighter import direct
from letorifighter.direct import UnsafeDisplayError, check_env_kwargs

GOOD = {
    "launch_mode": "direct",
    "game_dir": "/x",
    "display": ":101",
    "capture_display": ":101",
}


def test_pool_instance_is_accepted():
    assert check_env_kwargs(dict(GOOD)) == GOOD


@pytest.mark.parametrize(
    "change",
    [
        {"launch_mode": "steam"},
        {"display": ":0"},
        {"display": ":0.0"},
        {"display": None},
        {"capture_display": ":0"},
        {"game_dir": None},
    ],
)
def test_unsafe_kwargs_are_refused(change, monkeypatch):
    monkeypatch.delenv(direct.ALLOW_STEAM_ENV, raising=False)
    with pytest.raises(UnsafeDisplayError):
        check_env_kwargs({**GOOD, **change})


def test_default_launch_is_refused(monkeypatch):
    monkeypatch.delenv(direct.ALLOW_STEAM_ENV, raising=False)
    with pytest.raises(UnsafeDisplayError):
        check_env_kwargs({})


def test_original_display_is_refused(monkeypatch):
    monkeypatch.setattr(direct, "_ORIGINAL_DISPLAY", ":7")
    with pytest.raises(UnsafeDisplayError):
        check_env_kwargs({**GOOD, "display": ":7"})


def test_steam_needs_explicit_opt_in(monkeypatch):
    monkeypatch.setenv(direct.ALLOW_STEAM_ENV, "1")
    check_env_kwargs({"launch_mode": "steam"})
