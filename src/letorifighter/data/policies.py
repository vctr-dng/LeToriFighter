"""Learner policies for data collection: scripted, random, PPO, mixtures, noise injection.

A collection policy maps the task observation to a 22-component action, has
`reset(seed)` (all randomness comes from the episode seed, so an episode can be
replayed), a stable `policy_id` string and `component`, the id of the base
behaviour that acted in the current episode (differs from `policy_id` for
mixtures).

Spec forms accepted by `build_policy`:

* `"random"`, `"hold"`, `"scripted:<behaviour>"`, `"passive"`;
* `"ppo:<checkpoint>"` (argmax per head) or `"ppo-sample:<checkpoint>"` (samples
  the policy distribution), checkpoints of `letorifighter.ppo` (needs torch);
* `{"mix": [{"policy": <spec>, "weight": 3}, ...]}`: one component per episode,
  drawn with the episode seed;
* any of the above wrapped as `{"policy": <spec>, "epsilon": 0.1, "noise": 0.05}`:
  with probability `epsilon` per turn the action is uniformly random, otherwise
  each component is resampled uniformly with probability `noise`. Noise gives
  diversity (and recovery behaviour in the data) at the cost of cleaner demos.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from letorifighter.eval.policies import make_policy

NUM_COMPONENTS = 22
NUM_STATES = 4


class CollectPolicy:
    """Common behaviour: `policy_id`, `component`, `reset(seed)`, `__call__(obs)`."""

    policy_id: str = "policy"
    component: str = "policy"

    def reset(self, seed: int | None = None) -> None:
        raise NotImplementedError

    def __call__(self, obs: dict) -> np.ndarray:
        raise NotImplementedError


class BasePolicy(CollectPolicy):
    """Wraps a callable policy (`random`, `hold`, scripted, PPO)."""

    def __init__(self, policy_id: str, inner) -> None:
        self.policy_id = self.component = policy_id
        self.inner = inner

    def reset(self, seed: int | None = None) -> None:
        if hasattr(self.inner, "reset"):
            self.inner.reset(seed)

    def __call__(self, obs: dict) -> np.ndarray:
        return np.asarray(self.inner(obs))


class NoisyPolicy(CollectPolicy):
    """Epsilon-random turns and per-component noise on top of another policy."""

    def __init__(self, inner: CollectPolicy, epsilon: float = 0.0, noise: float = 0.0):
        if not (0.0 <= epsilon <= 1.0 and 0.0 <= noise <= 1.0):
            raise ValueError("epsilon and noise must be in [0, 1]")
        self.inner, self.epsilon, self.noise = inner, epsilon, noise
        self.rng = np.random.default_rng(0)
        self.policy_id = f"{inner.policy_id}+eps{epsilon:g}+noise{noise:g}"

    @property
    def component(self) -> str:  # type: ignore[override]
        return self.inner.component

    def reset(self, seed: int | None = None) -> None:
        self.inner.reset(seed)
        self.rng = np.random.default_rng([0 if seed is None else int(seed), 11])

    def __call__(self, obs: dict) -> np.ndarray:
        action = np.array(self.inner(obs), dtype=np.int64)
        if self.rng.random() < self.epsilon:
            return self.rng.integers(0, NUM_STATES, NUM_COMPONENTS)
        if self.noise > 0:
            mask = self.rng.random(NUM_COMPONENTS) < self.noise
            action[mask] = self.rng.integers(0, NUM_STATES, int(mask.sum()))
        return action


class MixturePolicy(CollectPolicy):
    """One component policy per episode, drawn by weight from the episode seed."""

    def __init__(self, components: list[CollectPolicy], weights: list[float]) -> None:
        if not components or len(components) != len(weights) or min(weights) <= 0:
            raise ValueError("a mixture needs components with positive weights")
        self.components = components
        self.weights = np.asarray(weights, dtype=np.float64) / sum(weights)
        self.current = components[0]
        total = float(sum(weights))
        parts = ",".join(
            f"{c.policy_id}:{w / total:.3g}" for c, w in zip(components, weights)
        )
        self.policy_id = f"mix({parts})"

    @property
    def component(self) -> str:  # type: ignore[override]
        return self.current.component

    def reset(self, seed: int | None = None) -> None:
        rng = np.random.default_rng([0 if seed is None else int(seed), 13])
        self.current = self.components[
            int(rng.choice(len(self.components), p=self.weights))
        ]
        self.current.reset(seed)

    def __call__(self, obs: dict) -> np.ndarray:
        return self.current(obs)


_PPO_CACHE: dict[tuple[str, bool], object] = {}


def _ppo_policy(path: str, deterministic: bool) -> BasePolicy:
    try:
        from letorifighter.ppo import PPOAgent
    except ImportError as error:
        raise RuntimeError(
            "PPO policies need torch: run with `uv run --group torch ...`"
        ) from error
    key = (str(Path(path).resolve()), deterministic)
    if key not in _PPO_CACHE:
        _PPO_CACHE[key] = PPOAgent.load(path).policy(deterministic=deterministic)
    kind = "ppo" if deterministic else "ppo-sample"
    return BasePolicy(
        f"{kind}:{Path(path).parent.name}/{Path(path).name}", _PPO_CACHE[key]
    )


def build_policy(spec: str | dict) -> CollectPolicy:
    """Build a collection policy from a spec (see the module docstring)."""
    if isinstance(spec, str):
        for prefix, deterministic in (("ppo:", True), ("ppo-sample:", False)):
            if spec.startswith(prefix):
                return _ppo_policy(spec.removeprefix(prefix), deterministic)
        if spec == "passive":
            return _passive()
        return BasePolicy(spec, make_policy(spec))
    if not isinstance(spec, dict):
        raise TypeError(f"policy spec must be a string or a dict, got {spec!r}")
    unknown = set(spec) - {"policy", "mix", "epsilon", "noise"}
    if unknown:
        raise ValueError(f"unknown policy keys {sorted(unknown)}")
    if ("policy" in spec) == ("mix" in spec):
        raise ValueError("a policy dict needs exactly one of `policy` and `mix`")
    if "mix" in spec:
        items = spec["mix"]
        base: CollectPolicy = MixturePolicy(
            [build_policy(item["policy"]) for item in items],
            [float(item.get("weight", 1.0)) for item in items],
        )
    else:
        base = build_policy(spec["policy"])
    epsilon, noise = float(spec.get("epsilon", 0.0)), float(spec.get("noise", 0.0))
    return NoisyPolicy(base, epsilon, noise) if epsilon or noise else base


def _passive() -> BasePolicy:
    from letorifighter.eval.policies import ScriptedPolicy

    return BasePolicy("passive", ScriptedPolicy("passive"))
