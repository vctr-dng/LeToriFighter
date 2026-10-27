"""Success detectors: computed from the learner's observation, never from reward.

A detector is fed the learner's observation once per turn boundary (not the
observation of `reset`) and latches success the first time its condition has
held for `hold_turns` consecutive observations. Detectors:

* `ReachDetector`: a hand of the learner is in contact with the other fighter's
  head or trunk (`geometry.CONTACT_THRESHOLD`) for `hold_turns` (default 2)
  consecutive turns. Two turns, not one, so a single glancing pass does not count.
* `ControlDetector`: the learner holds a target relative configuration for
  `hold_turns` (default 5) consecutive turns. Configurations (`CONFIGS`):
  `keep_close` (groin distance in [0.4, 1.0]), `keep_distance` (groin distance
  in [1.5, 2.2]), `clinch` (a hand in contact with the other fighter's head or
  trunk and that same hand's grip closed). The start distance is 1.3, so every
  configuration but `keep_distance`'s lower edge needs the learner to move.
* `EscapeDetector`: groin distance at least 2.5 for `hold_turns` (default 3)
  consecutive turns, and the learner's groin still at least `min_groin_height`
  above the floor (default 0.0, off). Meant against an opponent that chases.
* `SequenceDetector`: stages in order; a stage starts counting with the
  observation after the previous stage succeeded.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from . import geometry


@dataclass(frozen=True)
class Status:
    success: bool  # latched: stays True once reached
    stage: int  # index of the current stage (0 for single-stage tasks)
    streak: int  # consecutive observations the condition has held
    metrics: dict = field(default_factory=dict)


class Detector(ABC):
    hold_turns: int = 1

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._streak = 0
        self._success = False

    @abstractmethod
    def condition(self, obs: dict) -> bool:
        """Does the instantaneous condition hold in this observation?"""

    def metrics(self, obs: dict) -> dict:
        """Continuous quantities behind the condition, for logging."""
        return {}

    def update(self, obs: dict) -> Status:
        if self._success:
            return Status(True, 0, self._streak, self.metrics(obs))
        if self.condition(obs):
            self._streak += 1
        else:
            self._streak = 0
        self._success = self._streak >= self.hold_turns
        return Status(self._success, 0, self._streak, self.metrics(obs))


class ReachDetector(Detector):
    def __init__(self, hold_turns: int = 2, targets=geometry.TORSO) -> None:
        self.hold_turns = hold_turns
        self.targets = tuple(targets)
        super().__init__()

    def condition(self, obs: dict) -> bool:
        return any(geometry.hand_contacts(obs, self.targets).values())

    def metrics(self, obs: dict) -> dict:
        return {"contact_margin": geometry.reach_margin(obs, self.targets)}


# name -> (kind, parameters)
CONFIGS = {
    "keep_close": ("distance", {"low": 0.4, "high": 1.0}),
    "keep_distance": ("distance", {"low": 1.5, "high": 2.2}),
    "clinch": ("clinch", {}),
}


class ControlDetector(Detector):
    def __init__(self, config: str, hold_turns: int = 5) -> None:
        if config not in CONFIGS:
            raise ValueError(f"unknown config {config!r}, expected {sorted(CONFIGS)}")
        self.config = config
        self.hold_turns = hold_turns
        super().__init__()

    def condition(self, obs: dict) -> bool:
        kind, params = CONFIGS[self.config]
        if kind == "distance":
            return params["low"] <= geometry.groin_distance(obs) <= params["high"]
        contacts = geometry.hand_contacts(obs)
        grips = obs["self"]["grips"]  # [left, right]
        return (contacts["l_hand"] and grips[0] >= 0.5) or (
            contacts["r_hand"] and grips[1] >= 0.5
        )

    def metrics(self, obs: dict) -> dict:
        return {"groin_distance": geometry.groin_distance(obs)}


class EscapeDetector(Detector):
    def __init__(
        self,
        distance: float = 2.5,
        hold_turns: int = 3,
        min_groin_height: float = 0.0,
    ) -> None:
        self.distance = distance
        self.hold_turns = hold_turns
        self.min_groin_height = min_groin_height
        super().__init__()

    def condition(self, obs: dict) -> bool:
        return (
            geometry.groin_distance(obs) >= self.distance
            and float(obs["self"]["groin_height"][0]) >= self.min_groin_height
        )

    def metrics(self, obs: dict) -> dict:
        return {"groin_distance": geometry.groin_distance(obs)}


class SequenceDetector(Detector):
    def __init__(self, stages: list[Detector]) -> None:
        if not stages:
            raise ValueError("a sequence needs at least one stage")
        self.stages = stages
        super().__init__()

    def reset(self) -> None:
        super().reset()
        self._stage = 0
        for stage in self.stages:
            stage.reset()

    def condition(self, obs: dict) -> bool:  # not used, update is overridden
        return self.stages[self._stage].condition(obs)

    def metrics(self, obs: dict) -> dict:
        return self.stages[min(self._stage, len(self.stages) - 1)].metrics(obs)

    def update(self, obs: dict) -> Status:
        if self._success:
            return Status(True, len(self.stages) - 1, 0, self.metrics(obs))
        status = self.stages[self._stage].update(obs)
        shown_stage = self._stage
        if status.success:
            if self._stage == len(self.stages) - 1:
                self._success = True
            else:
                self._stage += 1
        return Status(self._success, shown_stage, status.streak, status.metrics)
