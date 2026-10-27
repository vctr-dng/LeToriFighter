"""Training-time wrappers for `LeToriTaskEnv`: task mix and potential-based shaping.

Success is always decided by the detector (`info["success"]`); shaping only adds
to the reward used for training. Evaluation (`letorifighter.eval`) never uses it.
"""

import gymnasium as gym
import numpy as np

from letorifighter.tasks import geometry

# task name -> (kind, target) of the potential. "contact": minimise the hand-to-torso
# margin; "distance": be near groin distance `target`; "far": maximise distance.
POTENTIALS: dict[str, tuple[str, float]] = {
    "reach": ("contact", 0.0),
    "control_clinch": ("contact", 0.0),
    "control_keep_close": ("distance", 0.7),
    "control_keep_distance": ("distance", 1.8),
    "escape": ("far", 0.0),
}


def potential(task: str, obs: dict) -> float:
    if task not in POTENTIALS:
        return 0.0
    kind, target = POTENTIALS[task]
    if kind == "contact":
        return -max(geometry.reach_margin(obs), 0.0)
    if kind == "far":
        return min(geometry.groin_distance(obs), 3.0)
    return -abs(geometry.groin_distance(obs) - target)


class TaskMixWrapper(gym.Wrapper):
    """
    Restricts the episodes of a `LeToriTaskEnv` split to some tasks (and
    opponents) by rejection sampling `Definitions.sample`. `tasks` and
    `opponents` are names; None keeps everything the split offers.
    """

    def __init__(self, env, tasks=None, opponents=None) -> None:
        super().__init__(env)
        self.definitions = env.unwrapped.definitions
        self.split = env.unwrapped.split
        offered = self.definitions.splits[self.split]["tasks"]
        self.tasks = list(tasks) if tasks else list(offered)
        unknown = set(self.tasks) - set(offered)
        if unknown:
            raise ValueError(f"tasks {sorted(unknown)} are not in split {self.split}")
        self.opponents = set(opponents) if opponents else None
        self._rng = np.random.default_rng()
        # fail early if the filter excludes every episode
        self._draw(np.random.default_rng(0), limit=2000)

    def _draw(self, rng, limit: int = 100_000):
        for _ in range(limit):
            spec = self.definitions.sample(
                self.split, int(rng.integers(0, 2**31 - 1)), 0
            )
            if spec.task in self.tasks and (
                self.opponents is None or spec.opponent in self.opponents
            ):
                return spec
        raise ValueError(
            f"no episode with tasks {self.tasks}, opponents {self.opponents}"
        )

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        options = dict(options or {})
        options.setdefault("episode", self._draw(self._rng))
        return self.env.reset(options=options)


class ShapedReward(gym.Wrapper):
    """Adds `scale * (gamma * phi(s') - phi(s))` to the reward (potential-based)."""

    def __init__(self, env, scale: float = 1.0, gamma: float = 0.99) -> None:
        super().__init__(env)
        self.scale = scale
        self.gamma = gamma
        self._task = ""
        self._phi = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._task = info["task"]
        self._phi = potential(self._task, obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        phi = potential(self._task, obs)
        bonus = self.scale * (self.gamma * phi - self._phi)
        self._phi = phi
        info["shaping"] = bonus
        return obs, float(reward) + bonus, terminated, truncated, info
