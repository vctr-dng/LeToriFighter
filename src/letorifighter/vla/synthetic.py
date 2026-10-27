"""A small learnable LeRobot v3 dataset for testing the pipeline without the game.

The world: a hidden scene `s_t` in `0..n_scenes-1` advances by one each step
(`s_{t+1} = (s_t + 1) mod n_scenes`). The instruction of an episode picks one of
`n_instructions` random lookup tables of shape `(n_scenes, 22)` with entries in
0..3, and the action at step t is `table[instruction][s_t]`. So

* the scene must be read from the observation: `info="image"` draws a bright
  square on grid cell `s_t` of a noisy image and fills the state with noise;
  `info="state"` encodes the scene in the state (a fixed random projection of the
  one-hot plus noise) and fills the image with noise; `info="both"` does both;
* the instruction (or task id) is needed to pick the table, so removing language
  conditioning cannot exceed chance on the tables' disagreements;
* a chunk of k future actions is `table[(s_t + j) mod n_scenes]`, a deterministic
  function of the current scene, so chunked prediction is learnable.

Chance per component is 25%.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from letorifighter.data.lerobot_writer import ACTION_DIM, LeRobotEpisodeWriter

INSTRUCTIONS = ("Reach the opponent's head.", "Stay on your feet.", "Run away.")
GRID = (4, 2)  # columns, rows of scene cells


def make_synthetic_dataset(
    root: str | Path,
    *,
    episodes: int = 48,
    length: int = 24,
    image_size: int = 64,
    state_dim: int = 300,
    n_scenes: int = 8,
    n_instructions: int = 2,
    info: str = "both",
    seed: int = 0,
    use_videos: bool = False,
    repo_id: str = "local/synthetic",
) -> dict[str, Any]:
    """Write the dataset to `root` (must not exist) and return its description
    (instruction -> task id, the lookup tables, settings)."""
    if info not in ("both", "image", "state"):
        raise ValueError(info)
    if n_scenes > GRID[0] * GRID[1] or n_instructions > len(INSTRUCTIONS):
        raise ValueError("too many scenes or instructions")
    rng = np.random.default_rng(seed)
    tables = rng.integers(0, 4, (n_instructions, n_scenes, ACTION_DIM))
    projection = rng.normal(size=(state_dim, n_scenes)).astype(np.float32)
    cell = image_size // GRID[0]
    with LeRobotEpisodeWriter(
        root,
        repo_id,
        fps=10,
        image_shape=(image_size, image_size, 3),
        state_dim=state_dim,
        use_videos=use_videos,
    ) as writer:
        for _ in range(episodes):
            instruction = int(rng.integers(n_instructions))
            scenes = (int(rng.integers(n_scenes)) + np.arange(length)) % n_scenes
            frames = rng.integers(0, 90, (length, image_size, image_size, 3)).astype(
                np.uint8
            )
            states = rng.normal(size=(length, state_dim)).astype(np.float32)
            for t, s in enumerate(scenes):
                if info in ("both", "image"):
                    x, y = (s % GRID[0]) * cell, (s // GRID[0]) * cell
                    frames[t, y : y + cell, x : x + cell] = 230
                if info in ("both", "state"):
                    states[t] = projection[:, s] + 0.3 * rng.normal(size=state_dim)
            writer.write_episode(
                frames, states, tables[instruction][scenes], INSTRUCTIONS[instruction]
            )
    description = {
        "task_ids": {INSTRUCTIONS[i]: i for i in range(n_instructions)},
        "tables": tables.tolist(),
        "episodes": episodes,
        "length": length,
        "info": info,
        "n_scenes": n_scenes,
        "repo_id": repo_id,
    }
    (Path(root) / "synthetic.json").write_text(json.dumps(description))
    return description
