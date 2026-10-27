"""Opponent policies and the episode runner, without the game."""

import subprocess
import sys

import numpy as np
import pytest

from letorifighter.bodies import GRIP_ON, HOLD, LIMB, SLOT
from letorifighter.opponents import (
    BEHAVIOURS,
    FrozenNetworkOpponent,
    Opponent,
    PassiveOpponent,
    RandomOpponent,
    ScriptedOpponent,
    check_action,
    flatten_observation,
    make_opponent,
)
from letorifighter.opponents.network import OBS_DIM
from letorifighter.opponents.scripted import GRAB_DISTANCE, hand_gap
from letorifighter.rollout import play_episode
from letorifighter.testing import FakeTwoPlayerEnv, make_observation

OBS = make_observation()


def all_opponents():
    return [
        PassiveOpponent(),
        RandomOpponent(seed=1),
        *[ScriptedOpponent(name) for name in BEHAVIOURS],
    ]


@pytest.mark.parametrize("opponent", all_opponents(), ids=lambda o: type(o).__name__)
def test_actions_are_valid_and_protocol_holds(opponent):
    assert isinstance(opponent, Opponent)
    opponent.reset(0)
    for _ in range(8):
        action = opponent.act(OBS)
        assert action.shape == (22,)
        assert action.dtype == np.int64
        check_action(action)


def test_passive_holds_joints_and_opens_hands():
    action = PassiveOpponent().act(OBS)
    assert (action[:20] == HOLD).all()
    assert (action[20:] < 2).all()


def test_random_is_reproducible_from_seed():
    a, b = RandomOpponent(), RandomOpponent()
    a.reset(7)
    b.reset(7)
    first = [a.act(OBS) for _ in range(5)]
    assert all((x == b.act(OBS)).all() for x in first)
    b.reset(8)
    assert not all((x == b.act(OBS)).all() for x in first)
    a.reset(7)
    assert (a.act(OBS) == first[0]).all()


def test_random_covers_all_values():
    opponent = RandomOpponent(seed=0)
    seen = np.concatenate([opponent.act(OBS) for _ in range(50)])
    assert set(seen.tolist()) == {0, 1, 2, 3}


@pytest.mark.parametrize("name", sorted(BEHAVIOURS))
def test_scripted_is_deterministic_and_resettable(name):
    a, b = ScriptedOpponent(name), ScriptedOpponent(name)
    first = [a.act(OBS) for _ in range(6)]
    assert all((x == b.act(OBS)).all() for x in first)
    a.reset()
    assert all((x == a.act(OBS)).all() for x in first)


def test_flail_changes_over_time_and_advance_grab_does_not_hold_still():
    flail = ScriptedOpponent("flail")
    actions = [flail.act(OBS) for _ in range(4)]
    assert len({a.tobytes() for a in actions}) == 4
    advance = ScriptedOpponent("advance_grab").act(OBS)
    assert (advance != PassiveOpponent().act(OBS)).any()
    assert (advance[20:] == GRIP_ON).all()


def test_advance_grab_clamps_when_a_hand_is_close():
    near = make_observation()
    near["opponent"]["positions"][LIMB["chest"]] = near["self"]["positions"][
        LIMB["r_hand"]
    ] + [0.0, -0.1, 0.0]
    assert hand_gap(near) < GRAB_DISTANCE
    action = ScriptedOpponent("advance_grab").act(near)
    assert (action[:20] == HOLD).all()
    assert (action[20:] == GRIP_ON).all()
    far = ScriptedOpponent("advance_grab").act(OBS)
    assert far[SLOT["abs"]] != HOLD


def test_unknown_names_raise():
    with pytest.raises(ValueError):
        ScriptedOpponent("nope")
    with pytest.raises(ValueError):
        make_opponent("nope")
    assert isinstance(make_opponent("scripted:flail"), ScriptedOpponent)
    assert isinstance(make_opponent("passive"), PassiveOpponent)


def test_check_action_rejects_bad_actions():
    with pytest.raises(ValueError):
        check_action(np.zeros(21, dtype=np.int64))
    with pytest.raises(ValueError):
        check_action(np.full(22, 4))
    with pytest.raises(ValueError):
        check_action(np.zeros(22, dtype=np.float32))


def test_flatten_observation_dimension():
    flat = flatten_observation(OBS)
    assert flat.shape == (OBS_DIM,)
    assert flat.dtype == np.float32


def test_play_episode_with_fake_env():
    env = FakeTwoPlayerEnv(turns=4, injury_fn=lambda turn: (0.0, 100.0 * turn))
    seen = []

    def learner(obs):
        seen.append(obs)
        return np.full(22, 2)

    result = play_episode(env, learner, PassiveOpponent(), seed=3)
    assert env.seeds == [3]
    assert result.turns == 4
    assert result.truncated and not result.terminated
    assert result.winner == "learner"
    assert result.opponent_injury == 400.0
    assert result.learner_return == pytest.approx(
        sum(100.0 * t for t in range(1, 5)) / 5000
    )
    assert len(result.learner_latency) == 4
    assert len(seen) == 4
    assert (env.history[0]["player_1"] == PassiveOpponent().act(OBS)).all()


def test_play_episode_as_player_one_and_hooks():
    env = FakeTwoPlayerEnv(turns=3, injury_fn=lambda turn: (50.0, 0.0))
    calls = []
    result = play_episode(
        env,
        lambda obs: np.full(22, 1),
        RandomOpponent(),
        seed=0,
        learner_agent="player_1",
        on_turn=lambda obs, actions, infos: calls.append(actions),
    )
    assert result.winner == "learner"  # player_1 took no damage
    assert result.learner_injury == 0.0
    assert len(calls) == 3
    assert (env.history[0]["player_1"] == 1).all()


def test_play_episode_rejects_bad_learner_action():
    env = FakeTwoPlayerEnv(turns=2)
    with pytest.raises(ValueError):
        play_episode(env, lambda obs: np.zeros(3, dtype=np.int64), PassiveOpponent())


def test_frozen_network_opponent_with_torch():
    torch = pytest.importorskip("torch")
    net = torch.nn.Linear(OBS_DIM, 88)
    with torch.no_grad():
        net.weight.zero_()
        net.bias.zero_()
        net.bias[2::4] = 5.0  # favour value 2 everywhere
    opponent = FrozenNetworkOpponent(net)
    action = opponent.act(OBS)
    assert (action == 2).all()
    assert not any(p.requires_grad for p in net.parameters())
    sampler = FrozenNetworkOpponent(net, deterministic=False, seed=1)
    first = [sampler.act(OBS) for _ in range(3)]
    sampler.reset(1)
    assert all((x == sampler.act(OBS)).all() for x in first)


def test_package_import_does_not_import_torch():
    code = "import sys, letorifighter.opponents, letorifighter.rollout; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-I", "-c", code], check=True)
