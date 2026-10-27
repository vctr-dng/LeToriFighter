"""Shared fixtures. Torch and LeRobot are optional; fixtures import them lazily."""

import json
import os
import shutil
import types

import pytest


@pytest.fixture(scope="session")
def xvfb_instance():
    """Keyword arguments of one ToriLLE `InstancePool` instance (own game copy, own Xvfb).

    Native tests only: the game runs in direct-launch mode on a private virtual
    display, never on the desktop. Skips when Xvfb or `torille.pool` is missing.
    """
    if shutil.which("Xvfb") is None:
        pytest.skip("Xvfb is not installed")
    pool_module = pytest.importorskip("torille.pool")
    from letorifighter.direct import check_env_kwargs

    with pool_module.InstancePool(1, display="xvfb") as pool:
        kwargs = next(iter(pool))
        yield check_env_kwargs(kwargs)


@pytest.fixture
def game_env(xvfb_instance):
    """Environment for a game subprocess: instance kwargs in `LETORI_ENV_KWARGS`, no desktop display."""
    env = {
        k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY")
    }
    env["LETORI_ENV_KWARGS"] = json.dumps(xvfb_instance)
    return env


@pytest.fixture
def tiny_smolvla_config(monkeypatch):
    """Factory for a tiny randomly initialised SmolVLA config (no downloads).

    The VLM config and processor lookups of LeRobot are replaced by a hand-built
    `SmolVLMConfig` (hidden 64, 2 layers) and a stub tokenizer namespace, so the
    policy classes can be built and run on CPU in a second.
    """
    pytest.importorskip("torch")
    pytest.importorskip("lerobot")
    pytest.importorskip("transformers")
    import lerobot.policies.smolvla.smolvlm_with_expert as sw
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from transformers import SmolVLMConfig

    tiny = SmolVLMConfig(
        vision_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "image_size": 32,
            "patch_size": 16,
        },
        text_config={
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "vocab_size": 128,
            "max_position_embeddings": 256,
        },
        scale_factor=2,
        vocab_size=128,
        image_token_id=3,
    )
    tokenizer = types.SimpleNamespace(fake_image_token_id=1, global_image_token_id=2)
    monkeypatch.setattr(sw.AutoConfig, "from_pretrained", lambda *a, **k: tiny)
    monkeypatch.setattr(
        sw.AutoProcessor,
        "from_pretrained",
        lambda *a, **k: types.SimpleNamespace(tokenizer=tokenizer),
    )

    def make(chunk_size: int = 4, state_dim: int = 8, **overrides):
        kwargs = {
            "chunk_size": chunk_size,
            "n_action_steps": 2,
            "max_state_dim": state_dim,
            "resize_imgs_with_padding": (32, 32),
            "load_vlm_weights": False,
            "num_vlm_layers": 2,
            "train_expert_only": False,
            "device": "cpu",
            "vlm_model_name": "tiny",
            "input_features": {
                "observation.images.front": PolicyFeature(
                    FeatureType.VISUAL, (3, 32, 32)
                ),
                "observation.state": PolicyFeature(FeatureType.STATE, (state_dim,)),
            },
            "output_features": {"action": PolicyFeature(FeatureType.ACTION, (22,))},
        }
        kwargs.update(overrides)
        return SmolVLAConfig(**kwargs)

    return make


@pytest.fixture
def tiny_batch():
    """Factory for a random batch with the keys SmolVLA expects."""
    torch = pytest.importorskip("torch")

    def make(
        chunk_size: int = 4, state_dim: int = 8, batch_size: int = 2, seed: int = 0
    ):
        g = torch.Generator().manual_seed(seed)
        states = torch.randint(0, 4, (batch_size, chunk_size, 22), generator=g)
        return {
            "observation.images.front": torch.rand(batch_size, 3, 32, 32, generator=g),
            "observation.state": torch.randn(batch_size, state_dim, generator=g),
            "observation.language.tokens": torch.randint(
                0, 100, (batch_size, 6), generator=g
            ),
            "observation.language.attention_mask": torch.ones(
                batch_size, 6, dtype=torch.bool
            ),
            "action": states.float() * (2 / 3) - 1,
            "action_states": states,
            "action_is_pad": torch.zeros(batch_size, chunk_size, dtype=torch.bool),
        }

    return make
