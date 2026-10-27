"""From-scratch PPO for Dict observations and `MultiDiscrete([4] * 22)` actions.

Needs the optional `torch` dependency group (`uv sync --group torch`); nothing
here is imported by the default package, so the default test set stays torch-free.

* `networks.ActorCritic`: task-id embedding + observation MLP, 22 categorical heads of 4.
* `buffer.RolloutBuffer`: single-env rollout storage with GAE.
* `agent.PPOAgent` / `PPOConfig`: rollout, clipped update, checkpoints.
* `agent.PPOPolicy`: `policy(obs) -> action`, deterministic by default, usable with
  `letorifighter.eval.evaluate`.
* `envs.TaskMixWrapper` / `envs.ShapedReward`: training-time task mix and shaping.
"""

from .agent import PPOAgent, PPOConfig, PPOPolicy, featurize
from .buffer import RolloutBuffer
from .envs import ShapedReward, TaskMixWrapper
from .networks import ActorCritic, RunningNorm

__all__ = [
    "ActorCritic",
    "PPOAgent",
    "PPOConfig",
    "PPOPolicy",
    "RolloutBuffer",
    "RunningNorm",
    "ShapedReward",
    "TaskMixWrapper",
    "featurize",
]
