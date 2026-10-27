"""Train a PPO expert on the task environment (M4).

Run (default `launch: direct`: the script starts its own ToriLLE `InstancePool(1)`,
one game copy on a private Xvfb, so nothing appears on your desktop and several runs
can go at once; always unset the desktop display variables):

    env -u DISPLAY -u WAYLAND_DISPLAY uv run --group torch --group train \\
        python scripts/train_ppo.py --config configs/ppo_reach.yaml \\
        [--set total_turns=50000 ppo.seed=1 name=ppo_reach_s1]

An externally managed pool instance: `--game-dir DIR --display :N`. `launch: steam`
(`--set launch=steam`) starts the game through the Steam client on the desktop
display; it takes the global game lock (`/tmp/toribash-game.lock`, do not wrap in
`flock`; with an outer flock pass `--no-lock`) and is refused by the guard unless
`LETORIFIGHTER_ALLOW_STEAM=1`.

Config: `TrainConfig` below, as YAML; `--set key=value` overrides (YAML values,
`ppo.` prefix for `PPOConfig` fields). Only the `train` split is used for
training; `val` is used for periodic evaluation. Outputs under `data/` (gitignored):
`data/ppo/<name>/{latest.pt,best.pt,config.yaml,eval/}` and the MLflow file store
`data/mlruns`. Training resumes from `latest.pt` when it exists (`resume: true`).
"""

import argparse
import fcntl
import math
import os
import signal
import sys
import time
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = "/tmp/toribash-game.lock"


@dataclass
class TrainConfig:
    name: str = "ppo_reach"
    tasks: list[str] = field(default_factory=lambda: ["reach"])  # train-split tasks
    opponents: list[str] | None = None  # None: the train pool of each task
    eval_opponents: list[str] | None = None  # None: same as `opponents`
    definitions: str = "v2"
    total_turns: int = 200_000
    turnframes: int = 5  # fixed by the definitions file; checked
    draw_game: bool = False  # PPO needs no capture
    launch: str = "direct"  # "direct": own InstancePool(1) on Xvfb; "steam": desktop
    seed: int = 0
    reward: str = "success"  # sparse success reward (detector based)
    shaping_scale: float = 0.0  # >0 adds potential-based shaping to the training reward
    eval_every_turns: int = 20_000
    eval_episodes: int = 20
    final_eval_episodes: int = 50
    checkpoint_every_updates: int = 5
    out_dir: str = "data/ppo"
    mlflow_uri: str = "data/mlruns"  # "" disables MLflow
    resume: bool = True
    ppo: dict = field(default_factory=dict)  # PPOConfig overrides


def load_config(path: str | None, overrides: list[str]) -> TrainConfig:
    raw = yaml.safe_load(Path(path).read_text()) if path else {}
    for item in overrides:
        key, _, value = item.partition("=")
        target = raw
        *parents, leaf = key.split(".")
        for parent in parents:
            target = target.setdefault(parent, {})
        target[leaf] = yaml.safe_load(value)
    known = {f.name for f in fields(TrainConfig)}
    unknown = set(raw) - known
    if unknown:
        raise SystemExit(f"unknown config keys: {sorted(unknown)}")
    return TrainConfig(**raw)


def acquire_game_lock() -> int:
    fd = os.open(LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"waiting for the game lock {LOCK_PATH} ...", flush=True)
        fcntl.flock(fd, fcntl.LOCK_EX)
    print("game lock acquired", flush=True)
    return fd  # held until the process exits


def val_specs(definitions, cfg: TrainConfig, episodes: int, seed: int):
    """`episodes` val episodes restricted to the training tasks (and opponents)."""
    opponents = cfg.eval_opponents or cfg.opponents
    chosen = []
    for spec in definitions.episodes("val", 100 * episodes, seed):
        if spec.task in cfg.tasks and (not opponents or spec.opponent in opponents):
            chosen.append(spec)
            if len(chosen) == episodes:
                return chosen
    raise SystemExit("no val episodes match the tasks and opponents")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config")
    parser.add_argument("--set", nargs="*", default=[], dest="overrides")
    parser.add_argument("--no-lock", action="store_true", help="steam mode only")
    parser.add_argument("--game-dir", help="direct mode: instance of an external pool")
    parser.add_argument("--display", help="direct mode: its Xvfb display, e.g. :101")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    if cfg.launch not in ("direct", "steam"):
        raise SystemExit("launch must be 'direct' or 'steam'")
    if bool(args.game_dir) != bool(args.display):
        raise SystemExit("--game-dir and --display go together")

    # SIGTERM must unwind (finally: close the game and the pool) instead of killing us
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    pool = None
    env_kwargs: dict = {}
    if cfg.launch == "steam":
        if not args.no_lock:
            acquire_game_lock()
    elif args.game_dir:
        env_kwargs = {
            "launch_mode": "direct",
            "game_dir": args.game_dir,
            "display": args.display,
            "capture_display": args.display,
        }
    else:
        from torille.pool import InstancePool

        pool = InstancePool(1, display="xvfb").start()
        env_kwargs = next(iter(pool))
        print(f"instance: {env_kwargs}", flush=True)
    if cfg.launch == "direct":
        from letorifighter.direct import check_env_kwargs

        check_env_kwargs(env_kwargs)

    from letorifighter.eval import evaluate
    from letorifighter.ppo import (
        PPOAgent,
        PPOConfig,
        ShapedReward,
        TaskMixWrapper,
    )
    from letorifighter.tasks import Definitions, LeToriTaskEnv

    definitions = Definitions.load(cfg.definitions)
    if definitions.turnframes != cfg.turnframes:
        raise SystemExit(
            f"turnframes={cfg.turnframes} but definitions {cfg.definitions} "
            f"use {definitions.turnframes}"
        )
    out = ROOT / cfg.out_dir / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(asdict(cfg)))

    base = LeToriTaskEnv(
        split="train",
        definitions=definitions,
        reward=cfg.reward,
        draw_game=cfg.draw_game,
        env_kwargs=env_kwargs,
    )
    env = TaskMixWrapper(base, cfg.tasks, cfg.opponents)
    if cfg.shaping_scale > 0:
        env = ShapedReward(env, cfg.shaping_scale, PPOConfig().gamma)

    latest = out / "latest.pt"
    num_tasks = max(definitions.task_ids.values()) + 1
    if cfg.resume and latest.exists():
        agent = PPOAgent.load(latest)
        print(f"resumed from {latest} at {agent.turns} turns", flush=True)
    else:
        agent = PPOAgent(num_tasks, replace(PPOConfig(seed=cfg.seed), **cfg.ppo))
    run_id = agent.meta.get("mlflow_run_id")

    mlflow = None
    if cfg.mlflow_uri:
        # the file store is in maintenance mode in MLflow 3 and needs this opt-in
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        import mlflow

        uri = Path(cfg.mlflow_uri)
        mlflow.set_tracking_uri((uri if uri.is_absolute() else ROOT / uri).as_uri())
        mlflow.set_experiment("ppo_baselines")
        run = mlflow.start_run(run_id=run_id, run_name=cfg.name)
        run_id = run.info.run_id
        if not agent.meta.get("mlflow_run_id"):
            flat = {f"cfg.{k}": str(v) for k, v in asdict(cfg).items() if k != "ppo"}
            flat.update({f"ppo.{k}": str(v) for k, v in asdict(agent.config).items()})
            flat["definitions_digest"] = definitions.digest[:12]
            mlflow.log_params(flat)

    def log(metrics: dict, step: int) -> None:
        if mlflow is not None:
            mlflow.log_metrics(
                {
                    k: v
                    for k, v in metrics.items()
                    if isinstance(v, int | float) and not math.isnan(v)
                },
                step=step,
            )

    state = {"next_eval": agent.turns + cfg.eval_every_turns, "best": -1.0}
    best_file = out / "best.json"
    if best_file.exists():
        state["best"] = yaml.safe_load(best_file.read_text())["success_rate"]

    def run_eval(episodes: int, tag: str) -> dict:
        specs = val_specs(definitions, cfg, episodes, seed=cfg.seed)
        report = evaluate(
            agent.policy(),
            base,  # the shared game instance
            "val",
            episodes,
            seed=cfg.seed,
            out_path=out / "eval" / f"{tag}.json",
            policy_name=cfg.name,
            specs=specs,
        )
        agent._obs = None  # the training episode was interrupted
        overall = report["overall"]
        log(
            {
                "val/success_rate": overall["success_rate"],
                "val/win_rate": overall["win_rate"],
                "val/injury_delta": overall["injury_delta_mean"],
            }
            | {
                f"val/{t}/success": s["success_rate"]
                for t, s in report["by_task"].items()
            },
            agent.turns,
        )
        print(
            f"[eval {tag}] val success {overall['success_rate']:.2f} "
            f"{overall['success_ci95']} win {overall['win_rate']:.2f}",
            flush=True,
        )
        if overall["success_rate"] > state["best"]:
            state["best"] = overall["success_rate"]
            agent.save(out / "best.pt", meta=agent.meta)
            best_file.write_text(
                yaml.safe_dump({"success_rate": state["best"], "turns": agent.turns})
            )
        return report

    def on_update(a: PPOAgent, metrics: dict) -> None:
        log({f"train/{k}": v for k, v in metrics.items()}, a.turns)
        print(
            f"turns {a.turns:>8} upd {a.updates:>4} ret {metrics['return_mean']:.3f} "
            f"succ {metrics['success_rate']:.2f} ent {metrics['entropy']:.2f} "
            f"kl {metrics['kl']:.4f} {metrics['turns_per_second']:.1f} turns/s",
            flush=True,
        )
        if a.updates % cfg.checkpoint_every_updates == 0:
            a.save(latest, meta={**a.meta, "mlflow_run_id": run_id})
        if cfg.eval_every_turns and a.turns >= state["next_eval"]:
            state["next_eval"] = a.turns + cfg.eval_every_turns
            run_eval(cfg.eval_episodes, f"val_{a.turns}")
            a.save(latest, meta={**a.meta, "mlflow_run_id": run_id})

    started = time.perf_counter()
    start_turns = agent.turns
    try:
        agent.meta["mlflow_run_id"] = run_id
        agent.learn(env, cfg.total_turns, on_update)
        agent.save(latest, meta=agent.meta)
        if cfg.final_eval_episodes:
            run_eval(cfg.final_eval_episodes, "val_final")
    finally:
        elapsed = time.perf_counter() - started
        print(
            f"{agent.turns - start_turns} turns in {elapsed:.0f} s "
            f"({(agent.turns - start_turns) / max(elapsed, 1e-9):.2f} turns/s incl. eval)",
            flush=True,
        )
        if mlflow is not None:
            mlflow.end_run()
        base.close()
        if pool is not None:
            pool.close()  # removes the game copy and the Xvfb


if __name__ == "__main__":
    main()
