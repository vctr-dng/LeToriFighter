"""Collect the M5 demonstration dataset, merge worker shards, or show progress.

    # default: direct launch on Xvfb, nothing appears on the desktop. `run-pool` starts a
    # ToriLLE InstancePool(N) (a game copy and an Xvfb each), N worker processes, then merges.
    env -u DISPLAY -u WAYLAND_DISPLAY uv run --group vla --group torch python \\
        scripts/collect_dataset.py run-pool --config configs/collect_m5.yaml --workers 4

    # one worker (a pool of 1 of its own), or one worker of an external pool
    env -u DISPLAY -u WAYLAND_DISPLAY uv run ... scripts/collect_dataset.py run --config c.yaml
    ... run --config c.yaml --worker-id 1 --num-workers 4 \\
        --env-kwargs '{"launch_mode":"direct","game_dir":"...","display":":101","capture_display":":101"}'

    # old Steam launch on the desktop display (refused by the guard unless
    # LETORIFIGHTER_ALLOW_STEAM=1): takes /tmp/toribash-game.lock itself, never wrap in flock
    LETORIFIGHTER_ALLOW_STEAM=1 uv run ... scripts/collect_dataset.py run --config c.yaml --launch steam

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

    pool = None
    if args.launch == "direct" and "game_dir" not in {
        **config.env_kwargs,
        **(
            config.worker_env_kwargs[args.worker_id]
            if args.worker_id < len(config.worker_env_kwargs)
            else {}
        ),
    }:
        from torille.pool import InstancePool

        pool = InstancePool(1, display="xvfb").start()
        config.env_kwargs.update(next(iter(pool)))

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

    try:
        if args.launch == "direct" or args.no_lock:  # own instance: no global lock
            return work()
        with game_lock():
            return work()
    finally:
        if pool is not None:
            pool.close()


def cmd_run_pool(args, config) -> int:
    from letorifighter.data.collect import run_pool_workers

    script = str(Path(__file__).resolve())

    def command(worker_id: int, kwargs: dict) -> list[str]:
        cmd = [
            sys.executable,
            script,
            "run",
            "--config",
            args.config,
            "--worker-id",
            str(worker_id),
            "--num-workers",
            str(args.workers),
            "--env-kwargs",
            json.dumps(kwargs),
        ]
        if args.max_episodes is not None:
            cmd += ["--max-episodes", str(args.max_episodes)]
        if args.max_disk_gb is not None:
            cmd += ["--max-disk-gb", str(args.max_disk_gb)]
        if args.overrides:  # last: `--set` takes every following argument
            cmd += ["--set", *args.overrides]
        return cmd

    codes = run_pool_workers(args.workers, command)
    print(f"worker exit codes: {codes}", flush=True)
    if any(code not in (0, 3) for code in codes):  # 3: disk cap reached
        return max(codes)
    if args.no_merge:
        return 0
    return cmd_merge(args, config)


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
    for name in ("run", "run-pool", "merge", "status"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--set", nargs="*", default=[], dest="overrides")
        if name in ("run", "run-pool"):
            p.add_argument("--max-episodes", type=int)
            p.add_argument("--max-disk-gb", type=float)
        if name == "run":
            p.add_argument("--worker-id", type=int, default=0)
            p.add_argument("--num-workers", type=int, default=1)
            p.add_argument("--launch", choices=("direct", "steam"), default="direct")
            p.add_argument(
                "--no-lock", action="store_true", help="steam mode: skip the game lock"
            )
            p.add_argument(
                "--env-kwargs", help="JSON dict for TwoPlayerToriEnv, this worker"
            )
        if name == "run-pool":
            p.add_argument("--workers", type=int, required=True)
            p.add_argument("--no-merge", action="store_true")
        if name in ("merge", "run-pool"):
            p.add_argument("--delete-shards", action="store_true")
            p.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config, args.overrides)
    commands = {
        "run": cmd_run,
        "run-pool": cmd_run_pool,
        "merge": cmd_merge,
        "status": cmd_status,
    }
    return commands[args.command](args, config)


if __name__ == "__main__":
    sys.exit(main())
