"""Dataset card of an M5 collection: counts, size, split policy, seeds, versions, leakage check.

`build_card(out_dir)` reads the merged sidecar (`merged/episodes.jsonl`, or the
shard sidecars if the run was not merged) and returns a JSON-serialisable dict;
`render_card(card)` makes Markdown. It needs neither torch nor lerobot (the
lerobot version is read from the package metadata when installed). The integrity
check re-validates every record against the frozen definitions (train split
only, see `data.leakage`) and compares the sidecar with the LeRobot `info.json`
episode and frame counts of the merged datasets.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections import Counter, defaultdict
from importlib import metadata
from pathlib import Path

from letorifighter.tasks import Definitions

from .leakage import check_spec
from .shards import SIDECAR, dataset_info, dir_size_bytes, read_jsonl

TORIBASH_APP_ID = 248570


def git_state(path: str | Path) -> dict:
    """Commit hash of the git repository at `path` and whether it has local changes."""

    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(path), *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {
            "commit": run("rev-parse", "HEAD"),
            "dirty": bool(run("status", "--porcelain")),
        }
    except OSError, subprocess.CalledProcessError:
        return {"commit": None, "dirty": None}


def toribash_build_id() -> str | None:
    """Steam build id of Toribash from its app manifest (None if Steam is not found)."""
    for root in (
        Path.home() / ".local/share/Steam/steamapps",
        Path.home() / ".steam/steam/steamapps",
    ):
        manifest = root / f"appmanifest_{TORIBASH_APP_ID}.acf"
        if manifest.exists():
            match = re.search(r'"buildid"\s+"(\d+)"', manifest.read_text())
            if match:
                return match.group(1)
    return None


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def collect_versions() -> dict:
    import torille

    import letorifighter

    return {
        "letorifighter": git_state(Path(letorifighter.__file__).resolve().parents[2]),
        "torille": git_state(Path(torille.__file__).resolve().parents[1]),
        "toribash_build_id": toribash_build_id(),
        "lerobot": package_version("lerobot"),
        "python": ".".join(map(str, __import__("sys").version_info[:3])),
    }


def load_records(out_dir: str | Path) -> tuple[list[dict], str]:
    """Sidecar records of the run and where they came from (`merged` or `shards`)."""
    out = Path(out_dir)
    merged = out / "merged" / SIDECAR
    if merged.exists():
        return read_jsonl(merged), "merged"
    records = []
    for sidecar in sorted(out.glob(f"shards/*/{SIDECAR}")):
        records.extend(read_jsonl(sidecar))
    return records, "shards"


def _group(records: list[dict], key) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[key(record)].append(record)
    return {
        name: {
            "episodes": len(items),
            "successes": sum(r["success"] for r in items),
            "frames": sum(r["turns"] for r in items),
        }
        for name, items in sorted(groups.items())
    }


def integrity_check(
    out_dir: str | Path, records: list[dict], definitions: Definitions
) -> dict:
    """No val/test episode in the data, and sidecar and datasets agree."""
    violations: list[str] = []
    for record in records:
        for problem in check_spec(definitions, record["spec"]):
            violations.append(f"seed {record['spec']['episode_seed']}: {problem}")
    seeds = [int(r["spec"]["episode_seed"]) for r in records]
    duplicates = [s for s, n in Counter(seeds).items() if n > 1]
    for seed in duplicates:
        violations.append(f"episode seed {seed} appears more than once")
    other = {
        name: sum(lo <= s <= hi for s in seeds)
        for name, split in definitions.splits.items()
        if name != "train"
        for lo, hi in [split["episode_seed_range"]]
    }
    merged = Path(out_dir) / "merged"
    for kind in ("demos", "clips"):
        root = merged / kind
        if not (root / "meta" / "info.json").exists():
            continue
        info = dataset_info(root)
        mine = [r for r in records if r.get("dataset") == kind]
        if info["total_episodes"] != len(mine):
            violations.append(
                f"{kind}: info.json has {info['total_episodes']} episodes, sidecar {len(mine)}"
            )
        frames = sum(r["turns"] for r in mine)
        if info["total_frames"] != frames:
            violations.append(
                f"{kind}: info.json has {info['total_frames']} frames, sidecar {frames}"
            )
    return {
        "ok": not violations,
        "checked_episodes": len(records),
        "episodes_in_other_split_seed_ranges": other,
        "violations": violations[:50],
        "violation_count": len(violations),
    }


def build_card(out_dir: str | Path, definitions: Definitions | None = None) -> dict:
    out = Path(out_dir)
    records, source = load_records(out)
    config_path = out / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    definitions = definitions or Definitions.load(config.get("definitions", "v2"))
    train = definitions.splits["train"]
    seeds = [int(r["spec"]["episode_seed"]) for r in records]
    pose_seeds: dict[int, list[int]] = defaultdict(list)
    for r in records:
        pose_seeds[int(r["spec"]["preroll_turns"])].append(int(r["spec"]["pose_seed"]))
    demos = [r for r in records if not r["action_free"]]
    clips = [r for r in records if r["action_free"]]
    sizes = {
        "total": dir_size_bytes(out),
        "merged": dir_size_bytes(out / "merged"),
        "shards": dir_size_bytes(out / "shards"),
    }
    return {
        "name": config.get("name", out.name),
        "records_from": source,
        "episodes": len(records),
        "frames": sum(r["turns"] for r in records),
        "demo_episodes": len(demos),
        "demo_frames": sum(r["turns"] for r in demos),
        "clip_episodes": len(clips),
        "clip_frames": sum(r["turns"] for r in clips),
        "successes": sum(r["success"] for r in records),
        "failures": sum(not r["success"] for r in records),
        "by_task": _group(records, lambda r: r["task"]),
        "by_opponent": _group(records, lambda r: r["opponent"]),
        "by_pose": _group(records, lambda r: f"preroll{r['spec']['preroll_turns']}"),
        "by_source": _group(records, lambda r: r["source"]),
        "by_policy": _group(records, lambda r: r["policy_id"]),
        "by_kind": _group(records, lambda r: "clips" if r["action_free"] else "demos"),
        "by_task_source": _group(records, lambda r: f"{r['task']} / {r['source']}"),
        "size_bytes": sizes,
        "capture_size": config.get("capture_size"),
        "fps": config.get("fps"),
        "split_policy": {
            "split": "train",
            "definitions_version": definitions.version,
            "definitions_digest": definitions.digest,
            "tasks": train["tasks"],
            "paraphrases": train["paraphrases"],
            "opponents": train["opponents"],
            "poses": train["poses"],
            "episode_seed_range": train["episode_seed_range"],
        },
        "seeds": {
            "plan_seed": config.get("plan_seed"),
            "episode_seed_min": min(seeds, default=None),
            "episode_seed_max": max(seeds, default=None),
            "unique_episode_seeds": len(set(seeds)),
            "pose_seeds": {
                f"preroll{k}": [min(v), max(v)] for k, v in sorted(pose_seeds.items())
            },
        },
        "versions": collect_versions(),
        "integrity": integrity_check(out, records, definitions),
    }


def _table(title: str, groups: dict[str, dict]) -> list[str]:
    lines = [
        f"| {title} | episodes | successes | success rate | frames |",
        "|---|---|---|---|---|",
    ]
    for name, g in groups.items():
        rate = g["successes"] / g["episodes"] if g["episodes"] else 0.0
        lines.append(
            f"| {name} | {g['episodes']} | {g['successes']} | {rate:.2f} | {g['frames']} |"
        )
    return lines + [""]


def human_size(size: int) -> str:
    if size >= 1024**3:
        return f"{size / 1024**3:.2f} GB"
    return f"{size / 1024**2:.1f} MB"


def render_card(card: dict) -> str:
    split = card["split_policy"]
    versions = card["versions"]
    integrity = card["integrity"]

    def commit(entry: dict) -> str:
        if not entry["commit"]:
            return "unknown"
        return entry["commit"][:12] + (" (dirty)" if entry["dirty"] else "")

    lines = [
        f"# Dataset card: {card['name']}",
        "",
        (
            "Toribash demonstration data (M5): per turn an RGB frame, a flat state vector, the 22-component "
            "action, and a language instruction, in LeRobot v3 format (`demos`) plus action-free clips "
            "(`clips`)."
        ),
        "",
        "## Contents",
        "",
        (
            f"- Episodes: {card['episodes']} ({card['demo_episodes']} demos with actions, "
            f"{card['clip_episodes']} action-free clips)"
        ),
        f"- Frames (turns): {card['frames']} ({card['demo_frames']} in demos, {card['clip_frames']} in clips)",
        f"- Success (detector fired): {card['successes']} episodes; kept failures: {card['failures']}",
        (
            f"- Frame size: {card['capture_size']} x {card['capture_size']} RGB, nominal {card['fps']} fps "
            "(one step per turn, `turnframes=5`)"
        ),
        (
            f"- Size on disk: {human_size(card['size_bytes']['total'])} (merged "
            f"{human_size(card['size_bytes']['merged'])}, shards {human_size(card['size_bytes']['shards'])})"
        ),
        f"- Records read from: {card['records_from']}",
        "",
        "## Split policy",
        "",
        (
            f"Only the `{split['split']}` split of definitions `{split['definitions_version']}` "
            f"(sha256 `{split['definitions_digest'][:12]}`): tasks {', '.join(split['tasks'])}; "
            f"`{split['paraphrases']}` paraphrases only; `{split['opponents']}` opponent pool; "
            f"`{split['poses']}` pose pool; episode seeds in "
            f"{split['episode_seed_range'][0]}-{split['episode_seed_range'][1]}. No `val` or `test_*` episode "
            "is in the data (checked below). Compositions and held-out paraphrases, opponents and poses are "
            "never collected."
        ),
        "",
        "## Seeds",
        "",
        f"- Plan seed: {card['seeds']['plan_seed']}",
        (
            f"- Episode seeds: {card['seeds']['unique_episode_seeds']} unique, "
            f"{card['seeds']['episode_seed_min']} to {card['seeds']['episode_seed_max']}"
        ),
        f"- Pose seeds per pre-roll: {json.dumps(card['seeds']['pose_seeds'])}",
        "",
        "## Versions",
        "",
        f"- LeToriFighter: {commit(versions['letorifighter'])}",
        f"- ToriLLE: {commit(versions['torille'])}",
        f"- Toribash build id: {versions['toribash_build_id']}",
        f"- LeRobot: {versions['lerobot']}",
        f"- Python: {versions['python']}",
        "",
        "## Integrity check",
        "",
        (
            f"- Result: **{'PASS' if integrity['ok'] else 'FAIL'}** "
            f"({integrity['checked_episodes']} episodes checked, {integrity['violation_count']} violations)"
        ),
        "- Episodes with a seed in another split's range: "
        + ", ".join(
            f"{k} {v}"
            for k, v in integrity["episodes_in_other_split_seed_ranges"].items()
        ),
        *(f"- {v}" for v in integrity["violations"][:10]),
        "",
        "## Counts",
        "",
    ]
    for title, key in (
        ("task", "by_task"),
        ("opponent", "by_opponent"),
        ("pose pool", "by_pose"),
        ("source", "by_source"),
        ("policy id", "by_policy"),
        ("dataset", "by_kind"),
        ("task / source", "by_task_source"),
    ):
        lines += [f"### By {title}", "", *_table(title, card[key])]
    return "\n".join(lines).rstrip() + "\n"


def write_card(
    out_dir: str | Path, path: str | Path | None = None
) -> tuple[Path, dict]:
    """Build the card and write it (default `<out_dir>/DATASET_CARD.md`, plus `card.json`)."""
    card = build_card(out_dir)
    path = Path(path) if path else Path(out_dir) / "DATASET_CARD.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_card(card))
    (Path(out_dir) / "card.json").write_text(json.dumps(card, indent=1) + "\n")
    return path, card
