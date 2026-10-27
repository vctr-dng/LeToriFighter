# LeToriFighter

Language-conditioned control in Toribash with VLA and WAM models. See
`docs/project-plan.md`. The game environment lives in the sibling repository
`../ToriLLE` (editable path dependency).

## Layout

- `src/letorifighter/opponents/`: opponent policies for
  `torille.envs.TwoPlayerToriEnv` (`passive`, `random`, `scripted:<behaviour>`,
  frozen torch network). Use `make_opponent("scripted:advance_grab")`.
- `src/letorifighter/rollout.py`: `play_episode(env, learner, opponent, seed)`.
- `src/letorifighter/testing.py`: synthetic observations and a fake two-player
  env for tests without the game.
- `src/letorifighter/data/`: LeRobot v3 episode writer.
- `src/letorifighter/tasks/`: M3 tasks (Reach, Control, Escape and compositions),
  success detectors, language templates, the frozen split definitions
  (`tasks/splits/v1.json`) and `LeToriTaskEnv`, a single-agent Gymnasium
  wrapper (task id and instruction in the observation). See `docs/m3-tasks-eval.md`.
- `src/letorifighter/eval/`: evaluation harness. Run a split with
  `uv run python -m letorifighter.eval ...` (it starts the game itself; see `letorifighter/eval` for the display options).

## Tests

```
uv run pytest              # fast, no game, no torch
uv run pytest -m native    # drives the real game (see below)
```

Native tests (and the training/collection scripts) run the game in ToriLLE direct-launch mode on a
private Xvfb display from an `InstancePool` instance (about 0.65 GB of disk per instance under
`~/.cache/torille/instances`, removed at exit), so nothing appears on the desktop and no global lock is
needed. The Steam client must be running (IPC only). Always unset the desktop display variables:

```
env -u DISPLAY -u WAYLAND_DISPLAY uv run pytest -m native
env -u DISPLAY -u WAYLAND_DISPLAY uv run --group torch --group train python scripts/train_ppo.py --config configs/ppo_reach.yaml
env -u DISPLAY -u WAYLAND_DISPLAY uv run --group vla --group torch python scripts/collect_dataset.py run-pool --config configs/collect_m5.yaml --workers 4
```

`letorifighter.direct.check_env_kwargs` refuses to start a game on `:0`, on the original `$DISPLAY`, or in
Steam-launch mode (unless `LETORIFIGHTER_ALLOW_STEAM=1`). See `docs/m4-ppo.md` and `docs/m5-dataset.md`.

Frozen-network opponents need torch: `uv sync --group torch`. Torch is imported
lazily, only when a `FrozenNetworkOpponent` is built.
