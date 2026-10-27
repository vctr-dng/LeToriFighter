"""The task wrapper and the evaluation harness on a fake two-player env (no game)."""

import json

import numpy as np
import pytest

from letorifighter.eval import evaluate, run_episode
from letorifighter.tasks import Definitions, EpisodeSpec, LeToriTaskEnv
from letorifighter.testing import (
    FakeTwoPlayerEnv,
    make_observation,
    obs_at_distance,
    obs_with_hand_contact,
)

DEFS = Definitions.load("v1")


def spec(task="reach", preroll=0, opponent="passive", horizon=None, **kwargs):
    task_spec = DEFS.tasks[task]
    return EpisodeSpec(
        split="val",
        index=0,
        task=task,
        task_id=task_spec.id,
        instruction="Touch the opponent.",
        paraphrase="train:0",
        opponent=opponent,
        preroll_turns=preroll,
        pose_seed=kwargs.get("pose_seed", 0),
        episode_seed=kwargs.get("episode_seed", 5),
        horizon_turns=horizon or task_spec.horizon_turns,
    )


def pair(obs):
    return {"player_0": obs, "player_1": obs}


def contact_from(turn_start, **kwargs):
    """Observation function: far until `turn_start`, then a hand on the opponent's head."""

    def fn(turn, actions):
        if turn >= turn_start:
            return pair(obs_with_hand_contact(match_frame=6.0 * turn, **kwargs))
        return pair(make_observation(match_frame=6.0 * turn))

    return fn


def make_env(observation_fn=None, injury_fn=None, **kwargs):
    fake = FakeTwoPlayerEnv(observation_fn=observation_fn, injury_fn=injury_fn)
    return fake, LeToriTaskEnv(fake, **kwargs)


def hold(obs):
    return np.full(22, 2)


def test_reset_observation_matches_space_and_exposes_task():
    fake, env = make_env()
    obs, info = env.reset(options={"episode": spec()})
    assert env.observation_space.contains(obs)
    assert obs["instruction"] == "Touch the opponent."
    assert obs["task_id"] == 0
    assert info["task"] == "reach" and info["success"] is False
    assert env.action_space.nvec.tolist() == [4] * 22
    assert fake.seeds == [5]
    assert fake.settings.get("matchframes") == 40 * 6


def test_reach_succeeds_after_two_contact_turns_and_reward_is_sparse():
    _fake, env = make_env(contact_from(3))
    env.reset(options={"episode": spec(horizon=10)})
    rewards, infos = [], []
    terminated = truncated = False
    while not (terminated or truncated):
        _obs, reward, terminated, truncated, info = env.step(hold(None))
        rewards.append(reward)
        infos.append(info)
    assert len(rewards) == 10 and truncated and not terminated
    assert rewards.count(1.0) == 1 and rewards[3] == 1.0  # turns 3 and 4 in contact
    assert infos[2]["success"] is False and infos[3]["success"] is True
    assert infos[-1]["success_turn"] == 4
    assert infos[-1]["winner"] is None  # no damage on either side


def test_info_reports_damage_relative_to_task_start_and_winner():
    # Both took damage during the pre-roll; only the task-phase difference counts.
    _fake, env = make_env(
        injury_fn=lambda turn: (1000.0, 1000.0 + (500.0 if turn > 2 else 0.0))
    )
    env.reset(options={"episode": spec(preroll=2, horizon=4)})
    info = None
    terminated = truncated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, info = env.step(hold(None))
    assert info["learner_injury"] == 0.0
    assert info["opponent_injury"] == 500.0
    assert info["injury_delta"] == 500.0
    assert info["winner"] == "learner"


def test_preroll_is_random_deterministic_and_hidden():
    fake, env = make_env()
    env.reset(options={"episode": spec(preroll=3, horizon=4, pose_seed=11)})
    first = [h["player_0"].copy() for h in fake.history]
    obs, _ = env.reset(options={"episode": spec(preroll=3, horizon=4, pose_seed=11)})
    # the game is not over, so reset drained nothing here; compare the 2nd pre-roll
    second = [h["player_0"] for h in fake.history[-3:]]
    assert len(first) == 3
    assert all((a == b).all() for a, b in zip(first, second, strict=True))
    env.reset(options={"episode": spec(preroll=3, horizon=4, pose_seed=12)})
    third = [h["player_0"] for h in fake.history[-3:]]
    assert not all((a == b).all() for a, b in zip(first, third, strict=True))
    # match_frame the policy sees starts at zero after the pre-roll
    assert obs["match_frame"][0] == 0.0
    assert fake.settings.get("matchframes") == (3 + 4) * 6


def test_opponent_acts_from_its_own_observation():
    fake, env = make_env()
    env.reset(options={"episode": spec(opponent="scripted:flail")})
    env.step(hold(None))
    env.step(hold(None))
    assert (fake.history[0]["player_0"] == 2).all()
    assert not (fake.history[0]["player_1"] == fake.history[1]["player_1"]).all()


def test_terminate_on_success_and_drain_before_next_reset():
    fake, env = make_env(contact_from(1), terminate_on_success=True)
    env.reset(options={"episode": spec(horizon=10)})
    _, _, terminated, truncated, info = env.step(hold(None))
    _, _, terminated, truncated, info = env.step(hold(None))
    assert terminated and not truncated and info["success"]
    assert fake.agents  # the fake game is still running
    env.reset(options={"episode": spec(horizon=10)})  # drains, then resets
    assert fake.seeds == [5, 5]


def test_trivial_start_flag():
    _, env = make_env(contact_from(0))
    _, info = env.reset(options={"episode": spec(horizon=4)})
    assert info["trivial_start"] is True
    _fake, env = make_env()
    _, info = env.reset(options={"episode": spec(horizon=4)})
    assert info["trivial_start"] is False


def test_sample_from_split_when_no_episode_given():
    _fake, env = make_env(split="test_composition")
    obs, info = env.reset(seed=0)
    assert DEFS.tasks[info["task"]].kind == "sequence"
    assert obs["instruction"][0].isupper()
    assert env.observation_space.contains(obs)


def test_reward_modes_and_validation():
    _fake, env = make_env(reward="none")
    env.reset(options={"episode": spec(horizon=2)})
    assert env.step(hold(None))[1] == 0.0
    with pytest.raises(ValueError):
        LeToriTaskEnv(FakeTwoPlayerEnv(), reward="bogus")
    with pytest.raises(RuntimeError):
        make_env()[1].step(hold(None))


def test_evaluate_reports_metrics_and_writes_json(tmp_path):
    # Hand on the head from turn 2: reach succeeds, nothing else does.
    _fake, env = make_env(contact_from(2), injury_fn=lambda turn: (0.0, 100.0 * turn))
    sleeps = []

    def policy(obs):
        sleeps.append(obs["task_id"])
        return np.full(22, 2)

    path = tmp_path / "out" / "report.json"
    report = evaluate(policy, env, "val", episodes=6, seed=0, out_path=path)
    saved = json.loads(path.read_text())
    assert saved["overall"]["episodes"] == 6
    assert report["split"] == "val" and report["definitions_version"] == "2.0.0"
    assert report["definitions_digest"] == Definitions.load().digest
    overall = report["overall"]
    reach_episodes = [e for e in report["episodes"] if e["spec"]["task"] == "reach"]
    assert reach_episodes and all(e["success"] for e in reach_episodes)
    assert overall["win_rate"] == 1.0 and overall["tie_rate"] == 0.0
    assert overall["injury_delta_mean"] > 0
    assert overall["latency_ms"]["turns"] == len(sleeps)
    assert 0.0 <= overall["success_rate"] <= 1.0
    assert (
        overall["success_ci95"][0]
        <= overall["success_rate"]
        <= overall["success_ci95"][1]
    )
    assert set(report["by_task"]) == {e["spec"]["task"] for e in report["episodes"]}
    assert "by_opponent" in report and "by_pose" in report


def test_evaluate_is_reproducible_in_its_episodes():
    _fake, env = make_env()
    a = evaluate(hold, env, "test_pose", episodes=5, seed=2)
    b = evaluate(hold, env, "test_pose", episodes=5, seed=2)
    assert [e["spec"] for e in a["episodes"]] == [e["spec"] for e in b["episodes"]]
    assert {e["spec"]["preroll_turns"] for e in a["episodes"]} == {6}


def test_policy_reset_is_called_with_episode_seed():
    seeds = []

    class Policy:
        def reset(self, seed):
            seeds.append(seed)

        def __call__(self, obs):
            return np.full(22, 2)

    _fake, env = make_env()
    episode = spec(horizon=2, episode_seed=77)
    record = run_episode(Policy(), env, episode)
    assert seeds == [77] and record["turns"] == 2
    assert len(record["latency_ms"]) == 2


def test_escape_task_success_with_distance():
    def far(turn, actions):
        return pair(obs_at_distance(1.3 if turn < 2 else 3.5, match_frame=6.0 * turn))

    _fake, env = make_env(far)
    env.reset(
        options={
            "episode": spec("escape", opponent="scripted:advance_grab", horizon=10)
        }
    )
    info = None
    for _ in range(10):
        _, _, _, _, info = env.step(hold(None))
    # five turns at 3.5 (turns 2 to 6)
    assert info["success"] and info["success_turn"] == 6
