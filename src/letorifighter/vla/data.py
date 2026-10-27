"""Torch dataset over a LeRobot v3 dataset written by `LeRobotEpisodeWriter`.

Each item is one frame with the next `chunk_size` actions as target and the keys
SmolVLA's `forward` expects:

* `observation.images.front` float (3,H,W) in [0,1]. With `use_images=False` a
  black 64x64 image and `observation.images.front_padding_mask=False`, so the
  image tokens are masked out of attention (state-only ablation);
* `observation.state` float (S,), standardised with `state_stats`;
* `observation.language.tokens` long (L,) and `observation.language.attention_mask`
  bool (L,): the instruction (`conditioning="language"`) or the text
  `"task <id>"` (`conditioning="task_id"`);
* `action` float (k,22): the target as levels in {-1,-1/3,1/3,1} (interface a);
  `action_states` long (k,22): the same as states 0..3 (interface b/c);
  `action_is_pad` bool (k,): True beyond the episode end;
* `task_id`, `episode_index`, `frame_index`.
"""
# pyrefly: ignore-errors
# (torch/lerobot/transformers are optional and loosely typed; checked by the tests)

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from letorifighter.data.lerobot_writer import ACTION_KEY, IMAGE_KEY, STATE_KEY

from .chunking import chunk_indices
from .levels import states_to_levels
from .text import TextEncoder, task_id_text

NO_IMAGE_SIZE = 64
MASK_KEY = f"{IMAGE_KEY}_padding_mask"
TOKENS_KEY = "observation.language.tokens"
ATTENTION_KEY = "observation.language.attention_mask"


def state_statistics(states: np.ndarray) -> dict[str, np.ndarray]:
    """Mean and std per state dimension (std floored so constant dims pass)."""
    return {
        "mean": states.mean(axis=0).astype(np.float32),
        "std": np.maximum(states.std(axis=0), 1e-3).astype(np.float32),
    }


class VLADataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str | Path,
        chunk_size: int,
        *,
        repo_id: str = "local/toribash",
        episodes: Sequence[int] | None = None,
        conditioning: str = "language",
        task_ids: Mapping[str, int] | None = None,
        use_images: bool = True,
        text_encoder: TextEncoder | None = None,
        state_stats: Mapping[str, np.ndarray] | None = None,
        preload_images: bool = True,
    ) -> None:
        """
        Args:
            root: Dataset directory (written by `LeRobotEpisodeWriter`).
            chunk_size: Number of future actions per target.
            episodes: Episode indices to use (default all).
            conditioning: `"language"` (instruction text) or `"task_id"`.
            task_ids: instruction -> task id, needed for `"task_id"` conditioning
                (see `text.instruction_task_ids`).
            use_images: False masks the image out (state-only).
            text_encoder: Tokenizer wrapper (built from the default SmolVLM2 if None).
            state_stats: `{"mean","std"}` to standardise states; computed from
                the selected frames if None (pass the train set's stats to val).
            preload_images: Decode all frames once into a uint8 tensor (fast,
                needs RAM: N*3*H*W bytes). If False, frames are decoded per item.
        """
        if conditioning not in ("language", "task_id"):
            raise ValueError(
                f"conditioning must be language or task_id: {conditioning}"
            )
        self.chunk_size = chunk_size
        self.conditioning = conditioning
        self.use_images = use_images
        self.text_encoder = text_encoder or TextEncoder()
        self.lerobot = LeRobotDataset(repo_id, root=Path(root))
        table = self.lerobot.hf_dataset.with_format("numpy")
        columns = table.select_columns(
            [STATE_KEY, ACTION_KEY, "episode_index", "frame_index", "task_index"]
        )[:]
        episode_index = np.asarray(columns["episode_index"]).reshape(-1)
        all_actions = np.asarray(columns[ACTION_KEY], dtype=np.float32)
        if not np.array_equal(all_actions, np.round(all_actions)):
            raise ValueError("actions in the dataset are not integral states 0..3")
        self.all_actions = all_actions.astype(np.int64)
        self.all_chunk_idx, self.all_chunk_pad = chunk_indices(
            episode_index, chunk_size
        )
        keep = np.arange(len(episode_index))
        if episodes is not None:
            keep = keep[np.isin(episode_index, list(episodes))]
        if len(keep) == 0:
            raise ValueError(f"no frames for episodes {episodes}")
        self.rows = keep  # global row index of each item
        self.episode_index = episode_index
        self.frame_index = np.asarray(columns["frame_index"]).reshape(-1)
        states = np.asarray(columns[STATE_KEY], dtype=np.float32)
        self.state_dim = states.shape[1]
        self.stats = (
            {k: np.asarray(v, np.float32) for k, v in state_stats.items()}
            if state_stats is not None
            else state_statistics(states[keep])
        )
        self.states = torch.from_numpy(
            (states - self.stats["mean"]) / self.stats["std"]
        ).float()
        tasks = self.lerobot.meta.tasks
        index_to_text = {
            int(row.task_index): str(text) for text, row in tasks.iterrows()
        }
        task_index = np.asarray(columns["task_index"]).reshape(-1)
        self.texts = [index_to_text[int(i)] for i in task_index]
        if conditioning == "task_id":
            if task_ids is None:
                raise ValueError("task_id conditioning needs task_ids")
            missing = {t for t in self.texts if t not in task_ids}
            if missing:
                raise ValueError(
                    f"instructions without a task id: {sorted(missing)[:3]}"
                )
            self.task_ids = np.array([task_ids[t] for t in self.texts])
        else:
            self.task_ids = np.full(len(self.texts), -1)
        self.images: torch.Tensor | None = None
        if use_images and preload_images:
            self.images = self._decode_all()

    def _frame(self, row: int) -> torch.Tensor:
        return self.lerobot[int(row)][IMAGE_KEY]

    def _decode_all(self) -> torch.Tensor:
        first = self._frame(self.rows[0])
        out = torch.empty((len(self.rows), *first.shape), dtype=torch.uint8)
        for i, row in enumerate(self.rows):
            out[i] = (self._frame(row) * 255).round().to(torch.uint8)
        return out

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        row = int(self.rows[i])
        if self.conditioning == "language":
            text = self.texts[row]
        else:
            text = task_id_text(self.task_ids[row])
        tokens, attention = self.text_encoder(text)
        chunk = self.all_actions[self.all_chunk_idx[row]]
        if self.use_images:
            if self.images is not None:
                image = self.images[i].float() / 255.0
            else:
                image = self._frame(row)
        else:
            image = torch.zeros(3, NO_IMAGE_SIZE, NO_IMAGE_SIZE)
        return {
            IMAGE_KEY: image,
            MASK_KEY: torch.tensor(self.use_images),
            STATE_KEY: self.states[row],
            TOKENS_KEY: tokens,
            ATTENTION_KEY: attention,
            ACTION_KEY: torch.from_numpy(states_to_levels(chunk)).float(),
            "action_states": torch.from_numpy(chunk),
            "action_is_pad": torch.from_numpy(self.all_chunk_pad[row]),
            "task_id": torch.tensor(int(self.task_ids[row])),
            "episode_index": torch.tensor(int(self.episode_index[row])),
            "frame_index": torch.tensor(int(self.frame_index[row])),
        }
