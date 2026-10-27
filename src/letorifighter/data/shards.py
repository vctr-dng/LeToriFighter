"""Shard directories of a collection run, and merging shards into one dataset.

Layout under the run's `out_dir`:

    config.json                      the collection config (written by every worker)
    shards/w<id>/episodes.jsonl      sidecar: one JSON record per finished episode
    shards/w<id>/segments/<kind>-<n>/  LeRobot v3 datasets of up to `segment_episodes`
                                     episodes; `kind` is `demos` or `clips`
    merged/<kind>/                   one LeRobot dataset per kind (after `merge`)
    merged/episodes.jsonl            sidecar of the merged datasets
    merged/manifest.json

A worker never reopens a finished LeRobot dataset (the writer only creates new
ones), so it writes a new *segment* every `segment_episodes` episodes. A segment
is finalised (loadable) before its records reach the sidecar. A crash loses at
most the open segment: on restart, segment directories no sidecar line points to
are deleted, and the episodes are collected again (resume skips by episode seed).
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, Self

import numpy as np

from .lerobot_writer import LeRobotEpisodeWriter

DEMOS = "demos"
CLIPS = "clips"
KINDS = (DEMOS, CLIPS)
SIDECAR = "episodes.jsonl"

# Per-step extra columns (VLADataset ignores them; they serve the WAM/value ablations).
EXTRA_FEATURES: dict[str, tuple[str, tuple[int, ...]]] = {
    "info.success": (
        "float32",
        (1,),
    ),  # the detector has latched at this step's observation
    "info.stage": ("float32", (1,)),  # current stage index at this step's observation
    "info.has_action": (
        "float32",
        (1,),
    ),  # 0 for action-free clips (actions are zero fill)
    "info.task_id": ("int64", (1,)),
}


def dir_size_bytes(path: str | Path) -> int:
    """Total size of the files under `path` (0 if it does not exist)."""
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def finished_seeds(out_dir: str | Path) -> set[int]:
    """Episode seeds already written by any shard under `out_dir`."""
    seeds = set()
    for sidecar in Path(out_dir).glob(f"shards/*/{SIDECAR}"):
        seeds.update(int(r["spec"]["episode_seed"]) for r in read_jsonl(sidecar))
    return seeds


def _default_writer(*args, **kwargs):
    return LeRobotEpisodeWriter(*args, **kwargs)


class _Segment:
    def __init__(self, writer, name: str, root: Path) -> None:
        self.writer, self.name, self.root = writer, name, root
        self.records: list[dict] = []


class ShardWriter:
    """Writes episodes of one worker: LeRobot segments plus the sidecar."""

    def __init__(
        self,
        shard_dir: str | Path,
        *,
        name: str,
        fps: int,
        image_shape: tuple[int, int, int],
        state_dim: int,
        segment_episodes: int = 50,
        use_videos: bool = True,
        writer_factory: Callable[..., Any] = _default_writer,
    ) -> None:
        self.shard = Path(shard_dir)
        self.name = name
        self.fps, self.image_shape, self.state_dim = fps, image_shape, state_dim
        self.segment_episodes = segment_episodes
        self.use_videos = use_videos
        self.writer_factory = writer_factory
        self.segments_dir = self.shard / "segments"
        self.segments_dir.mkdir(parents=True, exist_ok=True)
        self.sidecar = self.shard / SIDECAR
        self._open: dict[str, _Segment] = {}
        self._counter = self._recover()

    def _recover(self) -> int:
        referenced = {r["segment"] for r in read_jsonl(self.sidecar)}
        highest = -1
        for path in sorted(self.segments_dir.iterdir()):
            if path.name not in referenced:
                shutil.rmtree(path)  # unfinished segment of a crashed run
            else:
                highest = max(highest, int(path.name.rsplit("-", 1)[1]))
        return highest + 1

    def _segment(self, kind: str) -> _Segment:
        if kind not in self._open:
            segment = f"{kind}-{self._counter:04d}"
            self._counter += 1
            root = self.segments_dir / segment
            writer = self.writer_factory(
                root,
                f"local/{self.name}-{self.shard.name}-{segment}",
                fps=self.fps,
                image_shape=self.image_shape,
                state_dim=self.state_dim,
                extra_features=EXTRA_FEATURES,
                use_videos=self.use_videos,
            )
            self._open[kind] = _Segment(writer, segment, root)
        return self._open[kind]

    def add(self, episode, record: dict) -> dict:
        """Write one `CollectedEpisode`; `record` is completed and queued for the sidecar."""
        kind = CLIPS if record["action_free"] else DEMOS
        segment = self._segment(kind)
        n = len(episode.frames)
        has_action = 0.0 if record["action_free"] else 1.0
        actions = episode.actions if has_action else np.zeros_like(episode.actions)
        extras = {
            "info.success": episode.success_flags.reshape(n, 1).astype(np.float32),
            "info.stage": episode.stages.reshape(n, 1).astype(np.float32),
            "info.has_action": np.full((n, 1), has_action, np.float32),
            "info.task_id": np.full((n, 1), record["spec"]["task_id"], np.int64),
        }
        index = segment.writer.write_episode(
            episode.frames,
            episode.states,
            actions,
            episode.instruction,
            extras,
        )
        record = {
            **record,
            "dataset": kind,
            "segment": segment.name,
            "episode_index": index,
        }
        segment.records.append(record)
        if len(segment.records) >= self.segment_episodes:
            self._close(kind)
        return record

    def _close(self, kind: str) -> None:
        segment = self._open.pop(kind)
        segment.writer.finalize()
        with self.sidecar.open("a") as handle:
            for record in segment.records:
                handle.write(json.dumps(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def close(self) -> None:
        """Finalise open segments so that everything written so far is loadable."""
        for kind in list(self._open):
            self._close(kind)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ----------------------------------------------------------------------------- merge
def _default_aggregate(repo_ids, aggr_repo_id, roots, aggr_root) -> None:
    from lerobot.datasets.aggregate import aggregate_datasets

    aggregate_datasets(
        repo_ids=repo_ids, aggr_repo_id=aggr_repo_id, roots=roots, aggr_root=aggr_root
    )


def dataset_info(root: str | Path) -> dict:
    return json.loads((Path(root) / "meta" / "info.json").read_text())


def merge_shards(
    out_dir: str | Path,
    *,
    name: str,
    aggregate: Callable[..., None] = _default_aggregate,
    delete_shards: bool = False,
    force: bool = False,
    log: Callable[[str], None] = print,
) -> dict:
    """
    Combine all shards of `out_dir` into `merged/<kind>/` (one LeRobot dataset per
    kind) and `merged/episodes.jsonl`, with episode indices consistent between the
    two: record `episode_index` is the index in the merged dataset (the shard-local
    index is kept as `shard_episode_index`). Segments are ordered by shard id and
    segment number. Returns the manifest (also written as `merged/manifest.json`).
    """
    out = Path(out_dir)
    merged = out / "merged"
    shards = sorted(p for p in (out / "shards").glob("w*") if p.is_dir())
    if not shards:
        raise FileNotFoundError(f"no shards under {out / 'shards'}")
    if merged.exists():
        if not force:
            raise FileExistsError(f"{merged} exists (use force to rebuild it)")
        shutil.rmtree(merged)
    merged.mkdir(parents=True)

    by_segment: dict[tuple[str, str], list[dict]] = {}
    seen: set[int] = set()
    for shard in shards:
        for record in read_jsonl(shard / SIDECAR):
            seed = int(record["spec"]["episode_seed"])
            if seed in seen:
                raise ValueError(f"episode seed {seed} appears twice across shards")
            seen.add(seed)
            record = {**record, "shard": shard.name}
            by_segment.setdefault((shard.name, record["segment"]), []).append(record)

    all_records: list[dict] = []
    manifest: dict[str, Any] = {"name": name, "datasets": {}}
    for kind in KINDS:
        keys = sorted(k for k in by_segment if k[1].startswith(f"{kind}-"))
        if not keys:
            continue
        roots, repo_ids, offset = [], [], 0
        kind_records: list[dict] = []
        for shard_name, segment in keys:
            root = out / "shards" / shard_name / "segments" / segment
            records = sorted(
                by_segment[(shard_name, segment)], key=lambda r: r["episode_index"]
            )
            have = dataset_info(root)["total_episodes"]
            if have != len(records):
                raise ValueError(
                    f"{root}: {have} episodes but {len(records)} sidecar records"
                )
            for r in records:
                kind_records.append(
                    {
                        **r,
                        "shard_episode_index": r["episode_index"],
                        "episode_index": offset + r["episode_index"],
                    }
                )
            offset += len(records)
            roots.append(root)
            repo_ids.append(f"local/{name}-{shard_name}-{segment}")
        log(f"merging {len(roots)} {kind} segments, {offset} episodes")
        aggregate(repo_ids, f"local/{name}-{kind}", roots, merged / kind)
        info = dataset_info(merged / kind)
        if info["total_episodes"] != offset:
            raise ValueError(
                f"merged {kind}: {info['total_episodes']} != {offset} episodes"
            )
        all_records.extend(kind_records)
        manifest["datasets"][kind] = {
            "repo_id": f"local/{name}-{kind}",
            "episodes": info["total_episodes"],
            "frames": info["total_frames"],
            "segments": len(roots),
        }
    with (merged / SIDECAR).open("w") as handle:
        for record in all_records:
            handle.write(json.dumps(record) + "\n")
    (merged / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    if delete_shards:
        shutil.rmtree(out / "shards")
    return manifest
