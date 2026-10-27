"""End-to-end smoke tests with the real pretrained SmolVLA on the GPU.

Marked `gpu`: deselected by default. Run with
`LETORIFIGHTER_DATA=<data dir> uv run --group vla pytest -m gpu`. The first run
downloads about 3 GB into `$HF_HOME` (default `<data>/hf`).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lerobot")

pytestmark = pytest.mark.gpu

from letorifighter.vla.adapter import load_policy
from letorifighter.vla.config import VLAConfig
from letorifighter.vla.synthetic import make_synthetic_dataset
from letorifighter.vla.train import train


@pytest.fixture(scope="module")
def synthetic_root(tmp_path_factory):
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    root = tmp_path_factory.mktemp("syn") / "ds"
    make_synthetic_dataset(root, episodes=8, length=12, image_size=64)
    return root


@pytest.mark.parametrize("head", ["relaxed", "ce"])
def test_train_checkpoint_and_adapter(head, synthetic_root, tmp_path, monkeypatch):
    monkeypatch.setenv("LETORIFIGHTER_DATA", str(tmp_path))
    cfg = VLAConfig(
        dataset=str(synthetic_root),
        repo_id="local/synthetic",
        head=head,
        steps=6,
        batch_size=4,
        eval_every=6,
        eval_frames=8,
        val_episodes=2,
        log_every=2,
        out_dir=str(tmp_path / "run"),
        run_name=f"gpu-{head}",
    )
    summary = train(cfg)
    assert summary["peak_gpu_alloc_gb"] < 7
    assert np.isfinite(summary["final_train_loss"])
    assert (tmp_path / "run" / "final" / "model.pt").exists()

    policy = load_policy(tmp_path / "run" / "final")
    state = np.zeros(300, np.float32)
    obs = {
        "state": state,
        "instruction": "Stay on your feet.",
        "task_id": 1,
        "image": np.zeros((64, 64, 3), np.uint8),
    }
    policy.state_fn = lambda o: o["state"]
    policy.reset(seed=3)
    first = policy.predict_chunk(obs)
    policy.reset(seed=3)
    again = policy.predict_chunk(obs)
    assert first.shape == (cfg.chunk_size, 22)
    np.testing.assert_array_equal(first, again)
