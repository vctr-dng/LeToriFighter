"""Reference policies for the harness: baselines that need no training."""

import numpy as np

from letorifighter.opponents import make_opponent


class RandomPolicy:
    """Uniform random actions, seeded per episode by the harness."""

    def __init__(self) -> None:
        self.rng = np.random.default_rng(0)

    def reset(self, seed=None) -> None:
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs):
        return self.rng.integers(0, 4, 22)


class HoldPolicy:
    """Holds every joint with open hands."""

    def __call__(self, obs):
        action = np.full(22, 2)
        action[20:] = 0
        return action


class ScriptedPolicy:
    """An opponent behaviour (`passive`, `random`, `scripted:<name>`) used as the learner."""

    def __init__(self, name: str) -> None:
        self._policy = make_opponent(name)

    def reset(self, seed=None) -> None:
        self._policy.reset(seed)

    def __call__(self, obs):
        return self._policy.act(obs)


def make_policy(name: str):
    if name == "random":
        return RandomPolicy()
    if name == "hold":
        return HoldPolicy()
    if name.startswith("scripted:"):
        return ScriptedPolicy(name)
    raise ValueError(f"unknown policy {name!r}: random, hold or scripted:<behaviour>")
