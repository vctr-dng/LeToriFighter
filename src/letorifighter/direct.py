"""Direct-launch (Xvfb) helpers: a guard against opening game windows on a desktop.

The game must only run on a ToriLLE `InstancePool` display (`launch_mode="direct"`).
`check_env_kwargs` refuses anything that could put a window on the user's desktop:
Steam launch (it starts on the desktop display), a missing display (falls back to
`$DISPLAY`), `:0`, or the display `$DISPLAY` had when this process started.
`LETORIFIGHTER_ALLOW_STEAM=1` lifts the Steam-launch refusal (only for deliberate
use on a machine without a desktop concern).
"""

import contextlib
import os

DESKTOP_DISPLAYS = (":0", ":0.0")
_ORIGINAL_DISPLAY = os.environ.get("DISPLAY")  # at import time
ALLOW_STEAM_ENV = "LETORIFIGHTER_ALLOW_STEAM"


class UnsafeDisplayError(RuntimeError):
    pass


def _norm(display: str) -> str:
    return display.split(".")[0] if display.count(".") == 1 else display


def check_env_kwargs(kwargs: dict) -> dict:
    """Return `kwargs` if starting a game with them is safe, else raise."""
    mode = kwargs.get("launch_mode", "steam")
    if mode != "direct":
        if os.environ.get(ALLOW_STEAM_ENV) == "1":
            return kwargs
        raise UnsafeDisplayError(
            f"launch_mode={mode!r} starts the game on the desktop display; "
            "use launch_mode='direct' on an InstancePool display "
            f"(or set {ALLOW_STEAM_ENV}=1 on purpose)"
        )
    unsafe = {_norm(d) for d in DESKTOP_DISPLAYS}
    if _ORIGINAL_DISPLAY:
        unsafe.add(_norm(_ORIGINAL_DISPLAY))
    for key in ("display", "capture_display"):
        value = kwargs.get(key)
        if key == "display" and not value:
            raise UnsafeDisplayError(
                "direct launch without an explicit display would use $DISPLAY"
            )
        if value and _norm(str(value)) in unsafe:
            raise UnsafeDisplayError(
                f"{key}={value!r} is the desktop display; refusing to start a game"
            )
    if not kwargs.get("game_dir"):
        raise UnsafeDisplayError(
            "direct launch needs a game_dir (an InstancePool copy)"
        )
    return kwargs


@contextlib.contextmanager
def own_instance():
    """A private `InstancePool(1)` (game copy and Xvfb); yields guarded env kwargs."""
    from torille.pool import InstancePool

    with InstancePool(1, display="xvfb") as pool:
        yield check_env_kwargs(next(iter(pool)))
