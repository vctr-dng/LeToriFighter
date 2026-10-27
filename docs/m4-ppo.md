# M4: PPO expert baselines

Code: `src/letorifighter/ppo/` (PyTorch, from scratch), `scripts/train_ppo.py`, configs in `configs/`. Tests: `tests/test_ppo.py` (needs `--group torch`; skipped otherwise), `tests/test_train_script.py`.

## Implementation

- Policy: observation of the learner (`flatten_observation`, 300 values (`OBS_DIM`)) normalised by a running mean/std, concatenated with a learned `task_id` embedding (16-d), two tanh MLPs (256, 256) for actor and critic. The actor outputs 22 x 4 logits: a factored categorical (log-prob and entropy sum over the 22 heads; the entropy bonus uses the mean per head). The text instruction is not used (task-id conditioning).
- PPO: GAE (gamma 0.99, lambda 0.95), clipped objective (0.2), per-minibatch advantage normalisation, value loss, entropy bonus 0.01, grad clip 0.5, linear LR decay, KL early stop. Single env, rollout of 1024 turns, 10 epochs, minibatch 256. Truncated episodes (the horizon) bootstrap from the value of the final observation; terminated ones do not. The normaliser is refreshed after each update, not during it.
- Eval: `agent.policy()` is a callable `policy(obs) -> action` (argmax per head, `deterministic=False` samples), with `reset(seed)`, so `letorifighter.eval.evaluate` takes it directly.
- Checkpoints: `PPOAgent.save/load` (model, optimiser, normaliser, RNG state, counters, config; about 2.5 MB). Writes are atomic.
- `TaskMixWrapper` restricts training episodes to chosen tasks and opponents (rejection sampling over the `train` split, so only train pools are reachable). `ShapedReward` adds potential-based shaping to the training reward only; `info["success"]` is still the detector.
- Unit tests show learning without the game: on a toy two-task env whose optimal action depends on `task_id`, the mean return goes from about 1.5 (random) to above 4 of 6, and the deterministic policy plays each task's target.

## Training script

```
# Default `launch: direct`: the script starts its own ToriLLE InstancePool(1) (a copy of the game
# plus a private Xvfb), so no window appears on the desktop and no global game lock is needed.
# Always unset the desktop display variables. The Steam client must be running (IPC only).
env -u DISPLAY -u WAYLAND_DISPLAY uv run --group torch --group train python scripts/train_ppo.py --config configs/ppo_reach.yaml
env -u DISPLAY -u WAYLAND_DISPLAY uv run --group torch --group train python scripts/train_ppo.py --config configs/ppo_reach.yaml --set total_turns=100000 ppo.seed=1 name=ppo_reach_s1
```

Direct mode (Xvfb) details:

- Every run makes its own pool instance (`~/.cache/torille/instances/pool-*/instance-0`, about 0.65 GB of disk, plus roughly 1.6 GB of RAM for the game) and removes it at exit (also on SIGTERM and Ctrl-C). Training and the val evaluation share that instance. Several runs (different `name`, tasks, seeds) can run at the same time; each has its own checkpoint dir `data/ppo/<name>`. Do not start two runs with the same `name`.
- `--game-dir DIR --display :N` uses an instance of an externally managed pool instead (`InstancePool(n)` in your own script); the script then does not create or remove anything.
- A hard guard (`letorifighter.direct.check_env_kwargs`) refuses to start a game on `:0`, on the `$DISPLAY` the process started with, without an explicit display, or in Steam-launch mode. `--set launch=steam` is the old Steam launch on the desktop display (it takes `/tmp/toribash-game.lock`, `--no-lock` if you wrap it in `flock`); the guard refuses it unless `LETORIFIGHTER_ALLOW_STEAM=1`.
- `draw_game: false` (default) works on Xvfb. Measured on the busy shared machine: about 3 turns/s per run (not faster than `draw_game=true`, 3.0 vs 2.9 turns/s) because the game renders with software GL and still uses about 5 cores; three concurrent runs got 1.6 to 1.7 turns/s each (about 5 in total, load average 24). Scaling to concurrent runs is therefore CPU-bound; plan wall time with about 1.5 to 3 turns/s per run, not the 16 to 20 turns/s of the earlier Steam-launch runs on `:0`.

Config keys: `name`, `tasks`, `opponents` (null = the train pool of each task), `eval_opponents`, `definitions` (default `v2`), `total_turns`, `turnframes` (must equal the definitions' 5), `draw_game`, `seed`, `reward`, `shaping_scale`, `eval_every_turns`, `eval_episodes`, `final_eval_episodes`, `checkpoint_every_updates`, `out_dir` (`data/ppo`), `mlflow_uri` (`data/mlruns`, empty disables), `resume`, `ppo` (any `PPOConfig` field). Outputs: `data/ppo/<name>/{latest.pt,best.pt,best.json,config.yaml,eval/val_*.json}` (`data/` is gitignored; a run uses well under 100 MB). `best.pt` is the checkpoint with the best val success rate. Validation uses the `val` split restricted to the training tasks and opponents, on the same game instance as training (`evaluate(..., specs=...)`). Re-running the same command resumes from `latest.pt` (also the MLflow run). MLflow 3 file store (`MLFLOW_ALLOW_FILE_STORE` is set by the script): `mlflow ui --backend-store-uri data/mlruns`; logged: config params, `train/*` per update, `val/*` per evaluation.

## Smoke run (real game)

`configs/ppo_smoke.yaml`: Reach vs passive, `draw_game=False`, `turnframes=5`, 3072 turns, rollout 512.

- Training throughput: 16 to 20 turns/s (rollouts including the network forward and update; the update costs about 0.1 s per rollout). Whole run including 3 evaluations (5 + 5 + 10 val episodes of 40 turns): 3072 turns in 206 s, 14.9 turns/s overall.
- Loop works end to end: rollouts, GAE update, periodic val eval with `letorifighter.eval`, checkpoints, MLflow, resume (rerun with a larger `total_turns` continued from 3072; that rerun ran at only 6.5 turns/s while another process was using the machine, so budget for 2x slower than the figures below under contention). At 3k turns nothing has been learned, as expected (val success 0.6 of 5 episodes after 1.5k turns, 0.0 of 5, then 0.1 of 10: noise around the random baseline 0.2).

## Success thresholds (proposal)

Baselines on `val` (`docs/m3-tasks-eval.md`, v2, 10 episodes per task, so +-0.2): hold 0.0 to 0.1, random 0.1 to 0.2, scripted advance_grab as learner 0.0 to 0.5 (reach 0.4, keep_close 0.2, keep_distance 0.5, clinch 0.3, escape 0.0). Threshold = a clear margin over the best of these and a lower Wilson bound above random. Measure on 50 val episodes per task (CI half-width about 0.13):

| Task | Best trivial baseline | Pass threshold (val success) |
|---|---|---|
| reach (vs passive/random/advance_grab) | 0.4 (advance_grab) | at least 0.70 |
| control_keep_close | 0.2 | at least 0.60 |
| control_keep_distance (vs chasers) | 0.5 (advance_grab) | at least 0.75 |
| control_clinch | 0.3 | at least 0.50 |
| escape (vs chasers) | 0.2 (random) | at least 0.50 |

Report the number for the final `best.pt` on `val` with 100 episodes plus the three test splits `test_opponent`, `test_pose`, `test_paraphrase` (paraphrase is irrelevant for task-id PPO, so it is a no-op check). Seeds: 3 per task, report the median and range. These thresholds are guesses until one task is trained; revise them after the first Reach run.

## Recommended full runs

Throughput at 15 to 20 turns/s (Steam launch on `:0`; on Xvfb measured 1.6 to 3 turns/s, see above) means 100k turns is about 1.5 to 2 hours (`draw_game=False`), 300k about 4.5 to 5.5 hours (including periodic evaluation at 20k turns: 20 episodes x 40 to 50 turns is about 1 minute each). With direct mode they can run concurrently (each its own Xvfb instance), but see the CPU-bound throughput above: with N runs the per-run rate drops, so 2 to 3 at once is reasonable on this machine. The wall times in the table assume the earlier 16 to 20 turns/s and are optimistic for Xvfb.

| Run | Command (after `uv run --group torch --group train python scripts/train_ppo.py`) | Turns | Wall time |
|---|---|---|---|
| Reach, all train opponents | `--config configs/ppo_reach.yaml` | 300k | about 5 h |
| Reach, shaped | `--config configs/ppo_reach.yaml --set name=ppo_reach_shaped shaping_scale=1.0` | 300k | about 5 h |
| keep_close | `--config configs/ppo_reach.yaml --set name=ppo_keep_close 'tasks=[control_keep_close]'` | 300k | about 5 h |
| keep_distance | same with `name=ppo_keep_distance 'tasks=[control_keep_distance]'` | 300k | about 5 h |
| clinch | same with `name=ppo_clinch 'tasks=[control_clinch]'` | 300k | about 5 h |
| escape | same with `name=ppo_escape 'tasks=[escape]'` | 300k | about 5 h |
| multi-task | same with `name=ppo_multi 'tasks=[reach,control_keep_close,control_keep_distance,control_clinch,escape]' total_turns=1000000` | 1M | about 15 h |

Run the first Reach job and look at the curve before committing the others: if val success is no better than 0.3 at 100k turns, switch to the curriculum below. Run seeds 1 and 2 with `ppo.seed=N name=..._sN`. Total disk: well under 1 GB.

## Curriculum suggestion if Reach needs shaping

Reach success pays 1.0 only on the turn contact has held twice (rare under random play with 22 binary-ish joints, about 0.2 per episode), so the sparse reward may be enough, but credit assignment over 40 turns is poor. Ladder, each stage resumed from the previous `latest.pt` (set `resume` by using the same `name`, and change `opponents` or `shaping_scale` in `--set`):

1. `opponents=[passive]` with `shaping_scale=1.0` (potential = minus the hand-to-torso margin; `ppo/envs.py`): 100k turns, until val success on passive is at least 0.7.
2. `opponents=[passive, random]`, shaping on: 100k turns.
3. All train opponents including `scripted:advance_grab`, shaping annealed: set `shaping_scale=0.3`, then 0.

Shaping is potential-based so it does not change the optimal policy, and success is always read from the detector. Other tasks have shaping potentials too (distance to a target band, or distance gain for escape), off by default. Alternative if shaping is not enough: initialise from random-restart poses of `train` (already used) and raise `ent_coef` to 0.02 for exploration.

## Limitations

- One game instance per run, one environment: PPO sees highly correlated data and 5k to 20k policy updates is slow; expect Reach to need several hundred thousand turns.
- The `val` evaluation uses the same game, so it pauses training for about a minute each time.
- The reset fast-forward cost (`LeToriTaskEnv`) is included in the measured turns/s only in the sense that every episode plays to its horizon.
- Python 3.14 plus PyTorch: the torch group is not in the default set; install with `uv sync --group torch --group train` (several GB for the CUDA wheels). The agent runs on CPU by default (`ppo.device=cpu`, 2 threads): the network is tiny and the game dominates.
