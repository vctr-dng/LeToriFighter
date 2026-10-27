"""PPO on a toy env and on the fake two-player env (no game). Needs torch."""

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

torch = pytest.importorskip("torch")

from letorifighter.eval import evaluate
from letorifighter.ppo import (
    PPOAgent,
    PPOConfig,
    RolloutBuffer,
    ShapedReward,
    TaskMixWrapper,
)
from letorifighter.ppo.envs import potential
from letorifighter.tasks import Definitions, LeToriTaskEnv
from letorifighter.testing import (
    FakeTwoPlayerEnv,
    make_observation,
    obs_at_distance,
)

SMALL = PPOConfig(
    rollout_steps=256,
    minibatch_size=64,
    hidden=64,
    task_dim=8,
    epochs=6,
    learning_rate=1e-3,
    anneal_lr=False,
    seed=0,
)


class ToyEnv(gym.Env):
    """Per task a hidden target action; reward is the fraction of 22 heads that match it."""

    def __init__(self, horizon: int = 6, seed: int = 0) -> None:
        self.horizon = horizon
        rng = np.random.default_rng(seed)
        self.targets = rng.integers(0, 4, (2, 22))
        self.action_space = spaces.MultiDiscrete([4] * 22)
        self.observation_space = spaces.Dict({"task_id": spaces.Discrete(2)})

    def _obs(self):
        obs = make_observation()
        obs["task_id"] = np.int64(self.task)
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.task = int(self.np_random.integers(2))
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        match = float(np.mean(np.asarray(action) == self.targets[self.task]))
        self.t += 1
        self.last = match
        truncated = self.t >= self.horizon
        return self._obs(), match, False, truncated, {"success": match > 0.8}


def test_gae_matches_hand_computation_with_episode_cut():
    buffer = RolloutBuffer(3, 1)
    for reward, value, done in [(1.0, 0.5, 0), (2.0, 1.0, 1), (3.0, 2.0, 0)]:
        buffer.add([0], 0, np.zeros(22), 0.0, value, reward, done)
    gamma, lam = 0.9, 0.5
    buffer.compute_gae(last_value=4.0, gamma=gamma, lam=lam)
    a2 = 3.0 + gamma * 4.0 - 2.0
    a1 = 2.0 - 1.0  # done: no bootstrap, no carry-over
    a0 = (1.0 + gamma * 1.0 - 0.5) + gamma * lam * a1
    np.testing.assert_allclose(buffer.advantages, [a0, a1, a2], rtol=1e-5)
    np.testing.assert_allclose(buffer.returns, buffer.advantages + buffer.values)


def test_policy_shapes_and_deterministic_mode():
    env = ToyEnv()
    agent = PPOAgent(2, SMALL)
    obs, _ = env.reset(seed=0)
    action, logp, _ = agent.act(obs)
    assert action.shape == (22,) and action.min() >= 0 and action.max() < 4
    assert logp <= 0
    policy = agent.policy()
    assert np.array_equal(policy(obs), policy(obs))
    assert env.action_space.contains(policy(obs))


@pytest.fixture(scope="module")
def trained():
    env = ToyEnv()
    agent = PPOAgent(2, SMALL)
    returns = []
    agent.learn(env, 12_000, lambda a, m: returns.append(m["return_mean"]))
    return env, agent, returns


def test_ppo_learns_toy_task_and_return_improves(trained):
    env, agent, returns = trained
    assert np.mean(returns[:3]) < 2.5  # random play: 6 * 0.25 = 1.5
    assert np.mean(returns[-3:]) > 4.0  # optimal: 6.0
    policy = agent.policy()
    obs, _ = env.reset(seed=1)
    matches = []
    for _ in range(12):
        obs, reward, _, truncated, _ = env.step(policy(obs))
        matches.append(reward)
        if truncated:
            obs, _ = env.reset()
    assert np.mean(matches) > 0.7


def test_task_embedding_is_used(trained):
    # The tasks have different targets: a policy ignoring task_id would play one action.
    env, agent, _ = trained
    policy = agent.policy()
    obs, _ = env.reset(seed=3)
    actions = {}
    for task in (0, 1):
        obs["task_id"] = np.int64(task)
        actions[task] = policy(obs)
    for task in (0, 1):
        assert np.mean(actions[task] == env.targets[task]) > 0.7


def test_checkpoint_roundtrip_and_resume(tmp_path):
    env = ToyEnv()
    agent = PPOAgent(2, SMALL)
    agent.learn(env, 512)
    path = tmp_path / "ck" / "agent.pt"
    agent.save(path, meta={"note": "x"})
    assert not path.with_suffix(".pt.tmp").exists()
    loaded = PPOAgent.load(path)
    assert loaded.turns == agent.turns == 512 and loaded.meta == {"note": "x"}
    obs, _ = env.reset(seed=0)
    assert np.array_equal(agent.policy()(obs), loaded.policy()(obs))
    loaded.learn(env, 768)  # resumes where it stopped
    assert loaded.turns == 768 and loaded.updates == 3


def test_truncation_bootstraps_and_termination_does_not():
    class OneStep(ToyEnv):
        terminate = False

        def reset(self, *, seed=None, options=None):
            _, info = super().reset(seed=seed)
            self.task = 0
            return self._obs(), info

        def step(self, action):
            obs, _, _, _, info = super().step(action)
            return obs, 1.0, self.terminate, not self.terminate, info

    for terminate in (False, True):
        env = OneStep()
        env.terminate = terminate
        agent = PPOAgent(2, PPOConfig(rollout_steps=4, minibatch_size=4, hidden=8))
        buffer = RolloutBuffer(4, agent.obs_dim)
        agent.collect(env, buffer)
        value = agent.value_of(env._obs())
        expected = 1.0 + (0.0 if terminate else agent.config.gamma * value)
        np.testing.assert_allclose(buffer.rewards, expected, rtol=1e-4)


def make_task_env(**kwargs):
    def obs_fn(turn, actions):
        return dict.fromkeys(
            ("player_0", "player_1"), obs_at_distance(1.3, match_frame=6.0 * turn)
        )

    return LeToriTaskEnv(FakeTwoPlayerEnv(observation_fn=obs_fn), **kwargs)


def test_task_mix_wrapper_restricts_tasks_and_opponents():
    env = TaskMixWrapper(
        make_task_env(), tasks=["reach", "escape"], opponents=["scripted:advance_grab"]
    )
    seen = set()
    for i in range(15):
        _, info = env.reset(seed=i)
        seen.add((info["task"], info["opponent"]))
    assert seen == {
        ("reach", "scripted:advance_grab"),
        ("escape", "scripted:advance_grab"),
    }
    with pytest.raises(ValueError):
        TaskMixWrapper(make_task_env(), tasks=["reach_then_escape"])  # not in train
    with pytest.raises(ValueError):
        TaskMixWrapper(make_task_env(), tasks=["escape"], opponents=["passive"])


def test_shaped_reward_is_potential_based():
    env = ShapedReward(TaskMixWrapper(make_task_env(), tasks=["escape"]), gamma=1.0)
    env.reset(seed=0)
    total = 0.0
    for _ in range(4):
        _, _, _, _, info = env.step(np.full(22, 2))
        total += info["shaping"]
    assert abs(total) < 1e-5  # constant potential: shaping telescopes to 0
    near, far = obs_at_distance(1.0), obs_at_distance(2.5)
    assert potential("escape", far) > potential("escape", near)


def test_trained_agent_runs_in_eval_harness(tmp_path):
    env = make_task_env(split="val")
    train_env = TaskMixWrapper(make_task_env(), tasks=["reach"])
    n = max(Definitions.load().task_ids.values()) + 1
    agent = PPOAgent(n, SMALL)
    agent.learn(train_env, 256)
    report = evaluate(
        agent.policy(), env, "val", episodes=3, seed=0, out_path=tmp_path / "r.json"
    )
    assert report["overall"]["episodes"] == 3
    assert (tmp_path / "r.json").is_file()
