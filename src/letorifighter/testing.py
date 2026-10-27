"""Synthetic observations and a fake two-player env for tests without the game."""

from collections.abc import Callable

import numpy as np

from letorifighter.bodies import LIMB, NUM_JOINTS, NUM_LIMBS

AGENTS = ("player_0", "player_1")

# Standing pose at the start of a game, in the fighter's own frame (groin at
# the origin, x lateral, y negative toward the other fighter, z up). Values as
# observed in the native game at reset.
_STANDING = {
    "head": (0.0, -0.15, 1.10),
    "breast": (0.0, -0.10, 0.65),
    "chest": (0.0, -0.10, 0.40),
    "stomach": (0.0, -0.05, 0.20),
    "groin": (0.0, 0.0, 0.0),
    "r_pecs": (-0.25, -0.10, 0.60),
    "r_biceps": (-0.55, -0.10, 0.75),
    "r_triceps": (-0.95, -0.10, 0.75),
    "l_pecs": (0.25, -0.10, 0.60),
    "l_biceps": (0.55, -0.10, 0.75),
    "l_triceps": (0.95, -0.10, 0.75),
    "r_hand": (-1.35, -0.15, 0.75),
    "l_hand": (1.35, -0.15, 0.75),
    "r_butt": (-0.20, 0.0, -0.10),
    "l_butt": (0.20, 0.0, -0.10),
    "r_thigh": (-0.20, 0.0, -0.45),
    "l_thigh": (0.20, 0.0, -0.45),
    "l_leg": (0.20, 0.0, -1.05),
    "r_leg": (-0.20, 0.0, -1.05),
    "r_foot": (-0.20, -0.10, -1.45),
    "l_foot": (0.20, -0.10, -1.45),
}
STANDING_POSE = np.array([_STANDING[name] for name in LIMB], dtype=np.float32)
START_DISTANCE = 1.3  # distance between the two groins at reset


def facing_pose(distance: float = START_DISTANCE) -> np.ndarray:
    """The standing pose of the fighter that faces the first one, in the first one's frame."""
    pose = STANDING_POSE.copy()
    pose[:, 0] *= -1
    pose[:, 1] = -distance - pose[:, 1]
    return pose


def make_observation(
    self_positions=None,
    opponent_positions=None,
    match_frame: float = 0.0,
    frames_next_turn: float = 5.0,
) -> dict:
    """
    A per-agent observation in the `TwoPlayerToriEnv` format. By default both
    fighters stand at the start-of-game positions, `START_DISTANCE` apart.
    """

    def body(positions):
        positions = np.asarray(positions, dtype=np.float32).copy()
        return {
            "positions": positions,
            "velocities": np.zeros((NUM_LIMBS, 3), np.float32),
            "groin_height": np.array([1.49], np.float32),
            "joint_states": np.full(NUM_JOINTS, 3.0, np.float32),
            "grips": np.zeros(2, np.float32),
        }

    if self_positions is None:
        self_positions = STANDING_POSE
    if opponent_positions is None:
        opponent_positions = facing_pose()
    return {
        "self": body(self_positions),
        "opponent": body(opponent_positions),
        "match_frame": np.array([match_frame], np.float32),
        "frames_next_turn": np.array([frames_next_turn], np.float32),
    }


class FakeSettings:
    """Minimal stand-in for `ToribashSettings` (`set` and `get`)."""

    def __init__(self, **values) -> None:
        self.values = dict(values)

    def set(self, key, value) -> None:
        self.values[key] = value

    def get(self, key):
        return self.values[key]


class FakeTwoPlayerEnv:
    """
    Stand-in for `TwoPlayerToriEnv` that needs no game.

    `observation_fn(turn, actions)` returns the pair of observations after
    `turn` turns (turn 0 is the reset), default: the start pose every turn.
    The episode is truncated after `turns` turns. `injury_fn(turn)` returns
    `(injury of player_0, injury of player_1)`, default zeros. The winner is the
    fighter with less damage. If `matchframes` is set in `env.settings` (as the
    task wrapper does before each reset) the episode lasts
    `matchframes // (turnframes + 1)` turns, as in the real game.
    """

    possible_agents = AGENTS

    def __init__(
        self,
        turns: int = 5,
        turnframes: int = 5,
        observation_fn: Callable[[int, dict | None], dict] | None = None,
        injury_fn: Callable[[int], tuple[float, float]] | None = None,
    ) -> None:
        self.turns = turns
        self.turnframes = turnframes
        self.settings = FakeSettings(matchframes=turns * (turnframes + 1))
        self.observation_fn = observation_fn
        self.injury_fn = injury_fn or (lambda turn: (0.0, 0.0))
        self.agents: list[str] = []
        self.turn = 0
        self.seeds: list[int | None] = []
        self.history: list[dict] = []  # actions of every step ever taken
        self.closed = False

    def observation_space(self, agent):
        from torille.envs.two_player import observation_space

        return observation_space(10_000)

    def set_draw_game(self, visibility):
        pass

    def _observations(self, actions=None):
        if self.observation_fn is not None:
            return self.observation_fn(self.turn, actions)
        frame = float(self.turn * (self.turnframes + 1))
        return dict.fromkeys(AGENTS, make_observation(match_frame=frame))

    def reset(self, seed=None, options=None):
        self.seeds.append(seed)
        self.turns = self.settings.get("matchframes") // (self.turnframes + 1)
        self.turn = 0
        self.agents = list(AGENTS)
        return self._observations(), {agent: {} for agent in AGENTS}

    def step(self, actions):
        if not self.agents:
            raise RuntimeError("The episode has ended, call `reset`.")
        self.history.append(
            {agent: np.asarray(a).copy() for agent, a in actions.items()}
        )
        self.turn += 1
        done = self.turn >= self.turns
        injuries = self.injury_fn(self.turn)
        winner = 0
        if done and injuries[0] != injuries[1]:
            winner = 1 if injuries[0] < injuries[1] else 2
        winner_agent = {1: AGENTS[0], 2: AGENTS[1]}.get(winner) if done else None
        infos = {
            agent: {
                "winner": winner if done else None,
                "winner_agent": winner_agent,
                "injury": float(injuries[player]),
                "opponent_injury": float(injuries[1 - player]),
                "match_frame": self.turn * (self.turnframes + 1),
            }
            for player, agent in enumerate(AGENTS)
        }
        reward = (injuries[1] - injuries[0]) / 5000
        if done:
            self.agents = []
        return (
            self._observations(actions),
            {AGENTS[0]: reward, AGENTS[1]: -reward},
            dict.fromkeys(AGENTS, False),
            dict.fromkeys(AGENTS, done),
            infos,
        )

    def close(self):
        self.closed = True


def obs_at_distance(distance: float, **kwargs) -> dict:
    """Observation with the other fighter's groin `distance` away (standing poses)."""
    return make_observation(opponent_positions=facing_pose(distance), **kwargs)


def obs_with_hand_contact(
    hand: str = "r_hand",
    target: str = "head",
    gap: float = 0.1,
    distance: float = START_DISTANCE,
    grips=(0.0, 0.0),
    **kwargs,
) -> dict:
    """Observation where the learner's `hand` is `gap` above the other fighter's `target` limb."""
    opponent = facing_pose(distance)
    own = STANDING_POSE.copy()
    own[LIMB[hand]] = opponent[LIMB[target]] + np.array([0.0, 0.0, gap], np.float32)
    obs = make_observation(own, opponent, **kwargs)
    obs["self"]["grips"] = np.asarray(grips, np.float32)
    return obs
