"""Write Toribash episodes to a LeRobot v3 dataset.

Each episode is a sequence of aligned steps. A step holds an RGB frame
(uint8 HxWx3), a state vector, a 22-component action with values in {0..3}
(one per joint, the Toribash joint states), and optionally extra per-step
fields. The language instruction is stored as the LeRobot task string.

Requires the optional `vla` dependency group (`uv sync --group vla`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import numpy as np

if TYPE_CHECKING:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

ACTION_DIM = 22
ACTION_MAX = 3
IMAGE_KEY = "observation.images.front"
STATE_KEY = "observation.state"
ACTION_KEY = "action"


class LeRobotEpisodeWriter:
    """Create a LeRobot v3 dataset and append whole episodes to it."""

    def __init__(
        self,
        root: str | Path,
        repo_id: str,
        *,
        fps: int,
        image_shape: tuple[int, int, int],
        state_dim: int,
        extra_features: Mapping[str, tuple[str, tuple[int, ...]]] | None = None,
        use_videos: bool = True,
        robot_type: str = "toribash",
    ) -> None:
        """
        Args:
            root: Directory to create. It must not exist yet.
            fps: Steps per second of the stored timeline (frames per turn rate).
            image_shape: (height, width, 3) of the frames.
            state_dim: Length of the state vector.
            extra_features: Optional per-step fields as {name: (dtype, shape)},
                for example {"progress": ("float32", (2,))}. Shape (1,)
                is rejected (LeRobot 0.4.4 cannot write it).
            use_videos: Encode frames to mp4 (LeRobot default, needs ffmpeg
                codecs). If False, frames are stored as PNG inside parquet.
        """
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        if len(image_shape) != 3 or image_shape[2] != 3:
            raise ValueError(f"image_shape must be (H, W, 3), got {image_shape}")
        self.image_shape = tuple(image_shape)
        self.state_dim = state_dim
        self.extra_features = dict(extra_features or {})
        features: dict[str, dict[str, Any]] = {
            IMAGE_KEY: {
                "dtype": "video" if use_videos else "image",
                "shape": self.image_shape,
                "names": ["height", "width", "channels"],
            },
            STATE_KEY: {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": [f"state_{i}" for i in range(state_dim)],
            },
            # Integral values in {0..3}, stored as float32 so policies can
            # normalise them like any continuous action.
            ACTION_KEY: {
                "dtype": "float32",
                "shape": (ACTION_DIM,),
                "names": [f"joint_{i}" for i in range(ACTION_DIM)],
            },
        }
        for name, (dtype, shape) in self.extra_features.items():
            if tuple(shape) == (1,):
                # LeRobot 0.4.4 maps shape (1,) to a scalar HF column but
                # validates frames as shape (1,), so writing always fails.
                raise ValueError(f"{name}: shape (1,) is unsupported, use (2,) or more")
            features[name] = {"dtype": dtype, "shape": tuple(shape), "names": None}
        self.dataset: LeRobotDataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=features,
            root=Path(root),
            robot_type=robot_type,
            use_videos=use_videos,
        )
        self.num_episodes = 0
        self._finalized = False

    def write_episode(
        self,
        frames: np.ndarray,
        states: np.ndarray,
        actions: np.ndarray,
        language: str,
        extras: Mapping[str, np.ndarray] | None = None,
    ) -> int:
        """Append one episode and return its index.

        Args:
            frames: uint8 array (T, H, W, 3).
            states: array (T, state_dim).
            actions: array (T, 22) with integer values in {0..3}.
            language: Instruction stored as the episode's task string.
            extras: Arrays (T, *shape) for each configured extra feature.
        """
        extras = dict(extras or {})
        n = len(frames)
        if n == 0:
            raise ValueError("An episode needs at least one step")
        if frames.dtype != np.uint8 or frames.shape != (n, *self.image_shape):
            raise ValueError(
                f"frames must be uint8 {(n, *self.image_shape)}, "
                f"got {frames.dtype} {frames.shape}"
            )
        if states.shape != (n, self.state_dim):
            raise ValueError(
                f"states must have shape {(n, self.state_dim)}, got {states.shape}"
            )
        if actions.shape != (n, ACTION_DIM):
            raise ValueError(
                f"actions must have shape {(n, ACTION_DIM)}, got {actions.shape}"
            )
        if not np.array_equal(actions, np.round(actions)) or (
            actions.min() < 0 or actions.max() > ACTION_MAX
        ):
            raise ValueError(f"actions must be integers in 0..{ACTION_MAX}")
        if set(extras) != set(self.extra_features):
            raise ValueError(
                f"extras must be exactly {sorted(self.extra_features)}, "
                f"got {sorted(extras)}"
            )

        for t in range(n):
            frame: dict[str, Any] = {
                IMAGE_KEY: frames[t],
                STATE_KEY: np.asarray(states[t], dtype=np.float32),
                ACTION_KEY: np.asarray(actions[t], dtype=np.float32),
                "task": language,
            }
            for name, (dtype, _) in self.extra_features.items():
                frame[name] = np.asarray(extras[name][t], dtype=dtype)
            self.dataset.add_frame(frame)
        self.dataset.save_episode()
        self.num_episodes += 1
        return self.num_episodes - 1

    def write_episodes(self, episodes: Sequence[Mapping[str, Any]]) -> None:
        """Write several episodes given as kwargs dicts of `write_episode`."""
        for episode in episodes:
            self.write_episode(**episode)

    def finalize(self) -> None:
        """Flush parquet footers. The dataset is only loadable after this."""
        if not self._finalized:
            self.dataset.finalize()
            self._finalized = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.finalize()
