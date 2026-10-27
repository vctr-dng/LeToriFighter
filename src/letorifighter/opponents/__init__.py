"""Opponent policies for `torille.envs.TwoPlayerToriEnv` (see `base.Opponent`)."""

from .base import Opponent, check_action
from .network import FrozenNetworkOpponent, flatten_observation
from .scripted import BEHAVIOURS, ScriptedOpponent
from .simple import PassiveOpponent, RandomOpponent

__all__ = [
    "BEHAVIOURS",
    "FrozenNetworkOpponent",
    "Opponent",
    "PassiveOpponent",
    "RandomOpponent",
    "ScriptedOpponent",
    "check_action",
    "flatten_observation",
    "make_opponent",
]


def make_opponent(name: str, **kwargs) -> Opponent:
    """Build an opponent by name: `passive`, `random`, or `scripted:<behaviour>`."""
    if name == "passive":
        return PassiveOpponent()
    if name == "random":
        return RandomOpponent(**kwargs)
    if name.startswith("scripted:"):
        return ScriptedOpponent(name.split(":", 1)[1])
    raise ValueError(
        f"unknown opponent {name!r}: use passive, random or "
        f"scripted:<{'|'.join(sorted(BEHAVIOURS))}>"
    )
