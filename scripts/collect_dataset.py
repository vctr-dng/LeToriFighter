"""Collect the M5 demonstration dataset, merge worker shards, or show progress.

    # Steam-launch mode (one game per machine): the script takes /tmp/toribash-game.lock itself.
    # Do NOT wrap it in `flock` (a second lock on the same file deadlocks).
    uv run --group vla --group torch python scripts/collect_dataset.py run --config configs/collect_m5.yaml

    # direct-launch workers (own game instance each, ToriLLE instance pool): no lock
    uv run ... scripts/collect_dataset.py run --config c.yaml --worker-id 1 --num-workers 4 --no-lock

    uv run --group vla python scripts/collect_dataset.py merge --config c.yaml   # shards -> merged/
    uv run python scripts/collect_dataset.py status --config c.yaml

The config is YAML or JSON (`letorifighter.data.collect.CollectConfig`); `--set key=value`
overrides top-level keys (YAML values). Re-running `run` resumes: episodes whose seed is
already in a shard are skipped. Only the train split is collected.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | None, overrides: list[str]):
    from letorifighter.data.collect import CollectConfig

    raw: dict = {}
    if path:
        text = Path(path).read_text()
        if path.endswith(".json"):
            raw = json.loads(text)
        else:
            import yaml

            raw = yaml.safe_load(text)
    for item in overrides:
        key, _, value = item.partition("=")
        import yaml

        raw[key] = yaml.safe_load(value)
    config = CollectConfig.from_dict(raw)
    out = Path(config.out_dir)
    if not out.is_absolute():
        config.out_dir = str(ROOT / out)
    return config


def cmd_run(args, config) -> int:
    from letorifighter.data.collect import game_lock, make_env, run_collection
    from letorifighter.tasks import Definitions

    if args.max_disk_gb is not None:
        config.max_disk_gb = args.max_disk_gb
    if args.env_kwargs:
        config.env_kwargs.update(json.loads(args.env_kwargs))
    definitions = Definitions.load(config.definitions)

    def work() -> int:
        env = make_env(config, definitions, args.worker_id)
        try:
            summary = run_collection(
                config,
                definitions,
                env,
                worker_id=args.worker_id,
                num_workers=args.num_workers,
                max_episodes=args.max_episodes,
            )
        finally:
            env.close()
        return 0 if summary["stop"] != "disk_cap" else 3

    if args.no_lock:
        return work()
    with game_lock():
        return work()


def cmd_merge(args, config) -> int:
    from letorifighter.data.shards import merge_shards

    manifest = merge_shards(
        config.out_dir,
        name=config.name,
        delete_shards=args.delete_shards,
        force=args.force,
    )
    print(json.dumps(manifest, indent=1))
    return 0


def cmd_status(args, config) -> int:
    from letorifighter.data.collect import build_plan
    from letorifighter.data.shards import dir_size_bytes, finished_seeds
    from letorifighter.tasks import Definitions

    plan = build_plan(config, Definitions.load(config.definitions))
    done = finished_seeds(config.out_dir)
    finished = sum(item.spec.episode_seed in done for item in plan)
    print(
        f"{finished}/{len(plan)} planned episodes finished, "
        f"{dir_size_bytes(config.out_dir) / 1024**3:.2f} GB of {config.max_disk_gb} GB cap"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "merge", "status"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--set", nargs="*", default=[], dest="overrides")
        if name == "run":
            p.add_argument("--worker-id", type=int, default=0)
            p.add_argument("--num-workers", type=int, default=1)
            p.add_argument(
                "--no-lock", action="store_true", help="direct-launch worker"
            )
            p.add_argument("--max-episodes", type=int)
            p.add_argument("--max-disk-gb", type=float)
            p.add_argument(
                "--env-kwargs", help="JSON dict for TwoPlayerToriEnv, this worker"
            )
        if name == "merge":
            p.add_argument("--delete-shards", action="store_true")
            p.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config, args.overrides)
    return {"run": cmd_run, "merge": cmd_merge, "status": cmd_status}[args.command](
        args, config
    )


if __name__ == "__main__":
    sys.exit(main())
