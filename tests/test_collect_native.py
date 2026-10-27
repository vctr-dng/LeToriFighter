"""Collect two short episodes with real RGB from the game. Run with `pytest -m native`.

Runs `collect_dataset.py run-pool --workers 1`: a ToriLLE `InstancePool(1)` (game copy
plus Xvfb), direct launch, no `DISPLAY`, no game lock; then the merge. Needs ToriLLE
with `capture_size`, Xvfb and `--group vla` (LeRobot writer).
"""

import inspect
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.native

pytest.importorskip("lerobot")
torch = pytest.importorskip("torch")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "collect_dataset.py"
SIZE = 128
TURNS = 6


def run_script(*args):
    env = {
        k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY")
    }
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )


def test_two_short_episodes_with_real_rgb(tmp_path):
    from torille.envs import TwoPlayerToriEnv

    if shutil.which("Xvfb") is None:
        pytest.skip("Xvfb is not installed")

    if "capture_size" not in inspect.signature(TwoPlayerToriEnv.__init__).parameters:
        pytest.skip("this ToriLLE has no RGB capture")
    from letorifighter.data.card import build_card
    from letorifighter.data.shards import DEMOS, read_jsonl
    from letorifighter.vla.data import VLADataset

    out = tmp_path / "run"
    config = {
        "name": "native",
        "out_dir": str(out),
        "capture_size": SIZE,
        "horizon_override": TURNS,
        "segment_episodes": 2,
        "sources": [
            {
                "name": "random",
                "policy": "random",
                "episodes": {"reach": 2},
                "opponents": ["passive"],
            }
        ],
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    result = run_script("run-pool", "--workers", "1", "--config", str(path))
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert "game lock" not in result.stdout  # direct launch: no global lock
    assert "worker exit codes: [0]" in result.stdout

    records = read_jsonl(out / "merged" / "episodes.jsonl")
    assert len(records) == 2
    turns = [r["turns"] for r in records]
    # an episode may end before the horizon if the game ends early
    assert all(2 <= t <= TURNS for t in turns), turns

    class Text:
        def __call__(self, text):
            return torch.zeros(3, dtype=torch.long), torch.ones(3, dtype=torch.bool)

    ds = VLADataset(
        out / "merged" / DEMOS, 2, repo_id="local/native-demos", text_encoder=Text()
    )
    assert len(ds) == sum(turns)
    frames = torch.stack([ds[i]["observation.images.front"] for i in range(len(ds))])
    assert frames.shape == (sum(turns), 3, SIZE, SIZE)
    assert frames.std() > 0.02, "frames are constant"
    # the game moves, so consecutive frames of an episode differ
    assert (frames[1 : turns[0]] - frames[: turns[0] - 1]).abs().mean() > 1e-3
    # the state is the observation, not constant either
    assert np.asarray(ds.states).std(axis=0).max() > 0
    assert build_card(out)["integrity"]["ok"]
