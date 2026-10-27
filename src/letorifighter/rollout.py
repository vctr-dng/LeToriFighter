"""Play one `TwoPlayerToriEnv` episode: a learner callable against an `Opponent`."""

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from letorifighter.opponents.base import Opponent, check_action

AGENTS = ("player_0", "player_1")

Learner = Callable[[dict], np.ndarray]


@dataclass
class EpisodeResult:
    """Outcome of one episode, from the learner's point of view."""

    turns: int
    learner_return: float
    winner: str | None  # "learner", "opponent" or None for a tie
    learner_injury: float  # damage the learner took
    opponent_injury: float  # damage the opponent took
    terminated: bool
    truncated: bool
    learner_latency: list[float] = field(default_factory=list)  # seconds per turn


def play_episode(
    env,
    learner: Learner,
    opponent: Opponent,
    seed: int | None = None,
    learner_agent: str = "player_0",
    on_turn: Callable[[dict, dict, dict], None] | None = None,
) -> EpisodeResult:
    """
    Reset `env` and play until the episode ends.

    `learner` maps the learner's own observation to an action; if it has a
    `reset(seed)` method it is called too. `opponent.reset(seed)` is called
    with the same seed. `on_turn(observations, actions, infos)` is called after
    every turn if given. The env is not closed.
    """
    if learner_agent not in AGENTS:
        raise ValueError(f"learner_agent must be one of {AGENTS}")
    other = AGENTS[1 - AGENTS.index(learner_agent)]
    observations, _ = env.reset(seed=seed)
    opponent.reset(seed)
    if hasattr(learner, "reset"):
        learner.reset(seed)

    result = EpisodeResult(0, 0.0, None, 0.0, 0.0, False, False)
    infos: dict = {}
    while env.agents:
        started = time.perf_counter()
        learner_action = check_action(learner(observations[learner_agent]))
        result.learner_latency.append(time.perf_counter() - started)
        actions = {
            learner_agent: learner_action,
            other: check_action(opponent.act(observations[other])),
        }
        observations, rewards, terminations, truncations, infos = env.step(actions)
        result.turns += 1
        result.learner_return += float(rewards[learner_agent])
        result.terminated = bool(terminations[learner_agent])
        result.truncated = bool(truncations[learner_agent])
        if on_turn is not None:
            on_turn(observations, actions, infos)
    if infos:
        info = infos[learner_agent]
        result.learner_injury = float(info["injury"])
        result.opponent_injury = float(info["opponent_injury"])
        winner_agent = info["winner_agent"]
        if winner_agent is not None:
            result.winner = "learner" if winner_agent == learner_agent else "opponent"
    return result
