"""Single-agent Gymnasium wrapper: one learner against an `Opponent`, with a task.

`LeToriTaskEnv` wraps a `torille.envs.TwoPlayerToriEnv` (the learner is
`player_0` by default) and an opponent policy chosen per episode from the task
definition. It exposes

* observation: the learner's own `Dict` observation of the two-player env
  (`self`, `opponent`, `match_frame`, `frames_next_turn`) plus `task_id`
  (`Discrete`) and `instruction` (`Text`). `match_frame` counts from the start of
  the task (the pose pre-roll is subtracted), so the policy cannot read the
  pre-roll length from it;
* action: `MultiDiscrete([4] * 22)`;
* reward: `"success"` (default: 1.0 on the turn the task is first completed),
  `"injury"` (the two-player env's injury reward) or `"none"` (0). Success is
  decided by the detector from the observation, never by reward;
* `terminated` / `truncated`: the episode is truncated after the task's
  `horizon_turns`; it is terminated when the game ends early, or on success if
  `terminate_on_success=True`. ToriLLE only allows `reset()` after the game has
  ended, so ending early costs a fast-forward of the remaining turns (holding
  every joint) inside the next `reset()`; the default is therefore to play every
  episode to its horizon and read the latched `info["success"]`;
* `info`: `task`, `task_id`, `instruction`, `opponent`, `turn`, `success`
  (latched), `success_turn`, `stage`, `streak`, detector metrics, `trivial_start`
  (the first stage's condition already held when the task started),
  `learner_injury` and `opponent_injury` (damage taken since the task started,
  so the pre-roll is excluded), `injury_delta` (opponent minus learner) and, when
  the episode ends, `winner` (`"learner"`, `"opponent"` or `None` for a tie,
  decided on `injury_delta`).

`reset(options={"episode": EpisodeSpec})` plays exactly that episode; without it
an episode is sampled from `split` using the env's `np_random`.
"""

import string

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from letorifighter.bodies import NUM_CONTROLLABLES, NUM_JOINT_STATES
from letorifighter.opponents import PassiveOpponent, make_opponent

from .definitions import DEFAULT_VERSION, Definitions, EpisodeSpec
from .detectors import Status
from .poses import preroll

AGENTS = ("player_0", "player_1")
INSTRUCTION_CHARSET = string.ascii_letters + string.digits + " .,'-"
MAX_INSTRUCTION_LENGTH = 256


def task_observation_space(base: spaces.Dict, num_tasks: int) -> spaces.Dict:
    return spaces.Dict(
        {
            **base.spaces,
            "task_id": spaces.Discrete(num_tasks),
            "instruction": spaces.Text(
                min_length=1,
                max_length=MAX_INSTRUCTION_LENGTH,
                charset=INSTRUCTION_CHARSET,
            ),
        }
    )


class LeToriTaskEnv(gym.Env):
    metadata = {"render_modes": []}  # noqa: RUF012

    def __init__(
        self,
        env=None,
        split: str = "train",
        definitions: Definitions | None = None,
        learner_agent: str = "player_0",
        reward: str = "success",
        terminate_on_success: bool = False,
        opponent_factory=make_opponent,
        draw_game: bool = False,
        env_kwargs: dict | None = None,
    ) -> None:
        if learner_agent not in AGENTS:
            raise ValueError(f"learner_agent must be one of {AGENTS}")
        if reward not in ("success", "injury", "none"):
            raise ValueError("reward must be 'success', 'injury' or 'none'")
        self.definitions = definitions or Definitions.load(DEFAULT_VERSION)
        if split not in self.definitions.splits:
            raise KeyError(f"unknown split {split!r}")
        self.split = split
        self.learner = learner_agent
        self.other = AGENTS[1 - AGENTS.index(learner_agent)]
        self.reward_mode = reward
        self.terminate_on_success = terminate_on_success
        self.opponent_factory = opponent_factory
        self.turnframes = self.definitions.turnframes
        self.frames_per_turn = self.turnframes + 1  # `match_frame` step per turn
        max_frames = self.definitions.max_episode_turns() * self.frames_per_turn
        if env is None:
            from torille.envs import TwoPlayerToriEnv

            from letorifighter.direct import check_env_kwargs

            # `env_kwargs`: launch_mode/game_dir/display of a ToriLLE InstancePool
            # instance; the guard refuses anything that could open a desktop window
            kwargs = check_env_kwargs(dict(env_kwargs or {}))
            env = TwoPlayerToriEnv(
                matchframes=max_frames, turnframes=self.turnframes, **kwargs
            )
            env.set_draw_game(draw_game)
        self.env = env
        base = env.observation_space(self.learner)
        self.observation_space = task_observation_space(
            spaces.Dict(
                {
                    **base.spaces,
                    "match_frame": spaces.Box(0.0, float(max_frames), (1,), np.float32),
                    "frames_next_turn": spaces.Box(
                        0.0, float(max_frames), (1,), np.float32
                    ),
                }
            ),
            max(self.definitions.task_ids.values()) + 1,
        )
        self.action_space = spaces.MultiDiscrete([NUM_JOINT_STATES] * NUM_CONTROLLABLES)
        self._spec: EpisodeSpec | None = None
        self._detector = None
        self._opponent = None
        self._last_obs: dict | None = None

    # ------------------------------------------------------------- helpers
    def _drain(self) -> None:
        """Finish a game that was ended early (ToriLLE only resets finished games)."""
        hold = PassiveOpponent().act({})
        while self.env.agents:
            self.env.step({agent: hold for agent in AGENTS})

    def _task_obs(self, obs: dict) -> dict:
        assert self._spec is not None
        result = dict(obs)
        result["match_frame"] = obs["match_frame"] - self._frame_offset
        result["task_id"] = np.int64(self._spec.task_id)
        result["instruction"] = self._spec.instruction
        return result

    def _injuries(self, infos: dict | None) -> tuple[float, float]:
        if infos is None:
            return 0.0, 0.0
        info = infos[self.learner]
        return float(info["injury"]), float(info["opponent_injury"])

    def _info(self, status, infos: dict | None) -> dict:
        assert self._spec is not None
        learner_injury, opponent_injury = self._injuries(infos)
        learner_injury -= self._baseline[0]
        opponent_injury -= self._baseline[1]
        return {
            "task": self._spec.task,
            "task_id": self._spec.task_id,
            "instruction": self._spec.instruction,
            "opponent": self._spec.opponent,
            "turn": self._turn,
            "success": self._succeeded,
            "success_turn": self._success_turn,
            "stage": status.stage,
            "streak": status.streak,
            "detector": status.metrics,
            "trivial_start": self._trivial_start,
            "learner_injury": learner_injury,
            "opponent_injury": opponent_injury,
            "injury_delta": opponent_injury - learner_injury,
            "episode": self._spec.to_dict(),
        }

    # ------------------------------------------------------------ Gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        spec = options.get("episode")
        if spec is None:
            index = int(self.np_random.integers(0, 2**31 - 1))
            spec = self.definitions.sample(self.split, index, 0)
        elif isinstance(spec, dict):
            spec = EpisodeSpec.from_dict(spec)
        self._spec = spec

        self._drain()
        frames = (spec.preroll_turns + spec.horizon_turns) * self.frames_per_turn
        self.env.settings.set("matchframes", frames)
        observations, _ = self.env.reset(seed=spec.episode_seed)
        observations, infos, ended = preroll(
            self.env, observations, spec.preroll_turns, spec.pose_seed
        )
        if ended:
            raise RuntimeError(
                f"the game ended during the {spec.preroll_turns}-turn pre-roll"
            )
        self._baseline = self._injuries(infos)
        self._frame_offset = observations[self.learner]["match_frame"].copy()
        self._opponent = self.opponent_factory(spec.opponent)
        self._opponent.reset(spec.episode_seed)
        self._detector = self.definitions.detector(spec.task)
        self._detector.reset()
        self._turn = 0
        self._succeeded = False
        self._success_turn = None
        self._last_obs = observations
        self._trivial_start = bool(self._detector.condition(observations[self.learner]))
        info = self._info(Status(False, 0, 0, {}), None)
        return self._task_obs(observations[self.learner]), info

    def step(self, action):
        if (
            self._spec is None
            or self._last_obs is None
            or self._opponent is None
            or self._detector is None
        ):
            raise RuntimeError("`step` was called before `reset`.")
        action = np.asarray(action)
        opponent_action = self._opponent.act(self._last_obs[self.other])
        observations, rewards, terminations, truncations, infos = self.env.step(
            {self.learner: action, self.other: opponent_action}
        )
        self._last_obs = observations
        self._turn += 1
        status = self._detector.update(observations[self.learner])
        newly = status.success and not self._succeeded
        if newly:
            self._succeeded = True
            self._success_turn = self._turn

        if self.reward_mode == "success":
            reward = 1.0 if newly else 0.0
        elif self.reward_mode == "injury":
            reward = float(rewards[self.learner])
        else:
            reward = 0.0
        game_terminated = bool(terminations[self.learner])
        game_truncated = bool(truncations[self.learner])
        terminated = game_terminated or (self.terminate_on_success and newly)
        truncated = (
            game_truncated or self._turn >= self._spec.horizon_turns
        ) and not terminated
        info = self._info(status, infos)
        if terminated or truncated:
            delta = info["injury_delta"]
            info["winner"] = (
                "learner" if delta > 0 else "opponent" if delta < 0 else None
            )
        return (
            self._task_obs(observations[self.learner]),
            reward,
            terminated,
            truncated,
            info,
        )

    def close(self):
        self.env.close()
