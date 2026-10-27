"""M6 VLA pipeline: SmolVLA on Toribash episodes with three action interfaces.

Importing this package is cheap and needs neither torch nor lerobot; `levels`
and `chunking` are numpy only. Everything else (`data`, `model`, `train`,
`adapter`, `synthetic`) imports torch and lerobot on use (`uv sync --group vla`).

Large files (Hugging Face cache, checkpoints, MLflow store) go under the data
directory, `$LETORIFIGHTER_DATA` or `./data`. The Hugging Face cache defaults to
`<data>/hf` unless `HF_HOME` is already set.
"""

import os
from pathlib import Path


def data_dir() -> Path:
    """Root for caches, checkpoints and MLflow runs (gitignored)."""
    return Path(os.environ.get("LETORIFIGHTER_DATA", "data")).resolve()


os.environ.setdefault("HF_HOME", str(data_dir() / "hf"))
