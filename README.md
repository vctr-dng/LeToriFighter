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
  `flock /tmp/toribash-game.lock uv run python -m letorifighter.eval --split val --episodes 20 --policy random --out results/eval.json`.

## Tests

```
uv run pytest              # fast, no game, no torch
uv run pytest -m native    # drives the real game (see below)
```

Only one Toribash instance can run on the machine. Native tests start the game
in a subprocess wrapped in `flock /tmp/toribash-game.lock`, so do not wrap
`pytest -m native` itself in `flock` (it would deadlock). Any other command
that launches the game must be wrapped: `flock /tmp/toribash-game.lock <cmd>`.

Frozen-network opponents need torch: `uv sync --group torch`. Torch is imported
lazily, only when a `FrozenNetworkOpponent` is built.
