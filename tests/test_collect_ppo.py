"""A PPO checkpoint as collection policy (needs torch; no game)."""

import pytest

pytest.importorskip("torch")

from test_collect import DEFS, make_env, short

from letorifighter.data.collect import CollectConfig, build_plan, run_collection
from letorifighter.data.policies import build_policy
from letorifighter.data.shards import read_jsonl
from letorifighter.ppo import PPOAgent, PPOConfig


def test_ppo_checkpoint_policy_with_noise_in_a_collection(tmp_path):
    ckpt = tmp_path / "expert" / "best.pt"
    PPOAgent(num_tasks=9, config=PPOConfig(seed=1)).save(ckpt)
    policy = build_policy({"policy": f"ppo:{ckpt}", "epsilon": 0.1, "noise": 0.05})
    assert policy.policy_id == "ppo:expert/best.pt+eps0.1+noise0.05"
    sample = build_policy(f"ppo-sample:{ckpt}")
    assert sample.policy_id == "ppo-sample:expert/best.pt"

    cfg = CollectConfig.from_dict(
        {
            "name": "p",
            "out_dir": str(tmp_path / "run"),
            "capture_size": 16,
            "sources": [
                {
                    "name": "expert",
                    "policy": {"policy": f"ppo:{ckpt}", "noise": 0.05},
                    "episodes": {"reach": 2},
                }
            ],
        }
    )
    from test_collect import FakeWriter

    _, env = make_env()
    run_collection(
        cfg,
        DEFS,
        env,
        plan=short(build_plan(cfg, DEFS)),
        writer_factory=FakeWriter,
        log=lambda _: None,
    )
    records = read_jsonl(cfg.out / "shards" / "w0" / "episodes.jsonl")
    assert len(records) == 2
    assert {r["policy_id"] for r in records} == {"ppo:expert/best.pt+eps0+noise0.05"}
