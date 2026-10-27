# M3: tasks, success detectors, language, splits and evaluation harness

Code: `src/letorifighter/tasks/` and `src/letorifighter/eval/`. Opponents (M1) are in `src/letorifighter/opponents/`. Frozen definition files: `src/letorifighter/tasks/splits/v1.json` (version 1.0.0, sha256 `d45ad80b...c909c37`) and `v2.json` (version 2.0.0, sha256 `8063693e...dde43a3`), both pinned in `tests/test_tasks_definitions.py`. A change to a released definition means a new version file, not an edit. `Definitions.load()` and `LeToriTaskEnv` default to v2; `Definitions.load("v1")` still loads the original. v2 differs from v1 only in the opponents of `control_keep_distance` (see below); detectors, splits, instructions and poses are identical.

## Conventions

- Learner = `player_0`, opponent = `player_1`, `turnframes=5` (6 `match_frame` per turn). The learner sees its own `TwoPlayerToriEnv` observation, plus `task_id` and `instruction`.
- Detectors read only the learner's observation at turn boundaries (never reward) and latch the first time their condition has held for `hold_turns` consecutive observations. The reset observation is not fed.
- Body parts and indices: `letorifighter/bodies.py`.

## Contact definition

Toribash gives limb centres and velocities, no contact flags. Contact is a distance threshold between the centres of named limbs (`tasks/geometry.py`). Measured with `scripts/calibrate_contact.py` (36 episodes, 1800 turn boundaries, random and scripted fighters, `results/contact_calibration.json`): limbs cannot interpenetrate, so a hand-to-limb distance has a hard floor. Hand to the other fighter's head: floor 0.213 (p1 0.214, grabs pile up at 0.21-0.25). Hand to breast, chest, stomach, groin: smallest seen 0.35, 0.33, 0.40, 0.39 (rarely touched, so upper bounds on the floors). Thresholds = floor + about 0.1 margin (box orientation, one sample per turn): head 0.30, trunk parts 0.45. Damage is not usable as a contact signal: within 0.25 of the opponent's torso the opponent took damage in 0 of 222 turns (grabs do not hurt), against 5.5% of all turns.

## Tasks and detectors (`tasks/detectors.py`)

| Task id (name) | Success | Horizon |
|---|---|---|
| 0 `reach` | a hand within the contact threshold of the opponent's head or trunk for 2 consecutive turns | 40 turns |
| 1 `control_keep_close` | groin-groin distance in [0.4, 1.0] and learner groin height >= 1.1 (standing; start 1.49) for 5 turns | 50 |
| 2 `control_keep_distance` | distance in [1.6, 2.0], standing, for 10 turns; v2: opponent is a chaser (train `advance_grab`, held-out `rush`) | 50 |
| 3 `control_clinch` | a hand in contact with head or trunk and that hand's grip closed, for 5 turns | 50 |
| 4 `escape` | distance >= 3.0 and learner groin height >= 1.1 for 5 turns (opponent: a chaser) | 50 |
| 5-8 compositions | `reach_then_control_clinch`, `reach_then_control_keep_close`, `control_keep_close_then_escape`, `reach_then_escape`: stages in order, a stage starts counting on the observation after the previous one succeeded | 80 |

Start distance is 1.3, so no task is met at reset. The standing and distance thresholds came from 120 recorded episodes: fighters that hold or move randomly topple and drift apart (median distance 1.9 after 20 turns, 2.5 after 50), so a bare distance rule is met by doing nothing.

## Language (`splits/v1.json`)

Each atomic task has 5 `train` and 2 `heldout` paraphrases (disjoint, checked at load) and 3 short clauses. Compositions use 4 templates ("{a}, then {b}.", "First {a}, and after that {b}.", ...) filled with clauses of the steps; no training instruction is a composition.

## Opponents and poses

- Opponent pools: train `passive`, `random`, `scripted:advance_grab`; held-out `scripted:flail`, `scripted:guard`, `scripted:rush`. Escape (and compositions ending in escape) use chasers only: train `advance_grab`, held-out `rush`.
- Initial poses (`tasks/poses.py`): the game start is deterministic, so a pose is made by a random pre-roll: after `reset`, both fighters play K turns of uniform random actions from `np.random.default_rng(pose_seed)`. Measured: the same `(K, pose_seed)` gives bitwise identical positions (native test); K=2 moves the groin distance to 0.9-1.7 (start 1.3), K=6 to 1.0-2.1. Pre-roll damage and time are excluded from the task (detector starts after it, injuries are relative to that point, `match_frame` restarts at 0). A pose where the first stage already holds is flagged `trivial_start` (about 2-8% of episodes; reported separately).
- Pose pools: train K=0 and K=2 with seeds 0-999; val K=0 and K=2 seeds 1000-1999; held-out K=6 seeds 2000-2999.

## Splits (frozen v1)

| Split | Tasks | Instructions | Opponents | Poses | Episode seeds |
|---|---|---|---|---|---|
| `train` | atomic 0-4 | train | train | train | 0-999999 |
| `val` | atomic | train | train | val | 1000000-1099999 |
| `test_paraphrase` | atomic | held-out | train | val | 2000000-2099999 |
| `test_opponent` | atomic | train | held-out | val | 3000000-3099999 |
| `test_pose` | atomic | train | train | held-out | 4000000-4099999 |
| `test_composition` | 5-8 | template + train clauses | train | val | 5000000-5099999 |

`Definitions.load("v1").episodes(split, n, seed)` returns the same `EpisodeSpec` list every time (round-robin over tasks, other fields drawn from the pools).

## Wrapper and harness

`LeToriTaskEnv(env=None, split="train", reward="success", terminate_on_success=False)`: Gymnasium env; observation `Dict` (learner obs + `task_id` + `instruction`), action `MultiDiscrete([4]*22)`, `info` with `success`, `success_turn`, `stage`, injuries since task start, `winner`. `reset(options={"episode": spec})` plays an exact episode. See the module docstring of `tasks/env.py`.

`evaluate(policy, env, split, episodes, seed, out_path)` (`eval/harness.py`) writes JSON: success rate with Wilson 95% interval, win and tie rate (win = opponent took more damage than the learner during the task), mean injury delta (opponent minus learner), per-turn policy latency (mean, p50, p95, max), the same by task, opponent and pose group, and every episode. CLI: `flock /tmp/toribash-game.lock uv run python -m letorifighter.eval --split val --episodes 20 --policy random --out results/eval.json` (policies `random`, `hold`, `scripted:<behaviour>`).

## keep_distance tightened in v2

In v1, `control_keep_distance` was met by doing nothing (hold policy: 4 of 10 episodes) because standing fighters lean apart into the band while the opponent sits still. v2 plays it only against chasers (the same pools as `escape`): a chaser closes the distance to grab, so holding still ends below 1.6, and the learner has to keep its distance actively. The detector code is unchanged, the 10-turn hold stays. Hold policy on `val`: 0.4 (v1, n=10) to 0.0 (v2, n=10); random 0.1 to 0.2; scripted advance_grab as learner 0.4 to 0.5 (n=10 each, so about +-0.2: advance_grab walks backwards into the band while the chaser opponent does the same, which is still not a learned skill). v1 numbers: `results/m3_baselines_v1.json`.

## Trivial baselines (`scripts/m3_baselines.py`, `results/m3_baselines.json`, definitions v2)

Success rate, seed 0, 10 episodes per task on `val` (so about +-0.2 per task); the table below is the v2 rerun (v1 differs only in keep_distance: hold 0.4, random 0.1, advance_grab 0.4, rush 0.4; overall 0.10, 0.14, 0.26, 0.12):

| Policy | overall | reach | keep_close | keep_distance | clinch | escape |
|---|---|---|---|---|---|---|
| hold | 0.02 | 0.0 | 0.1 | 0.0 | 0.0 | 0.0 |
| random | 0.16 | 0.2 | 0.1 | 0.2 | 0.1 | 0.2 |
| scripted advance_grab as learner | 0.28 | 0.4 | 0.2 | 0.5 | 0.3 | 0.0 |
| scripted rush as learner | 0.04 | 0.1 | 0.1 | 0.0 | 0.0 | 0.0 |

Random policy on the test splits (24-25 episodes, v2): paraphrase 0.12, opponent 0.20, pose 0.16, composition 0.08. A learned policy has to beat these floors; M4 thresholds should be set relative to them.

## Limitations

- `keep_distance` (v1) was partly met by doing nothing; fixed in v2 by using chaser opponents (above). Random play still reaches it in 1 of 5 episodes by drifting.
- Contact is sampled once per turn and was fitted with `turnframes=5`, classic mod; trunk thresholds rest on few touches.
- Random pre-roll can produce poses that already satisfy a task (flagged) or are hard for reasons unrelated to the policy (fallen fighter).
- Early termination (`terminate_on_success`) costs a fast-forward of the remaining turns at the next `reset()` because ToriLLE only resets a finished game; the default plays every episode to its horizon.
- The game ran with `draw_game=False` in all measurements; latencies in the report are the policy call only, not the bridge.
- One native game at a time: evaluating many episodes is serial (about 18 turns/s without drawing, so 50 episodes of 50 turns takes about 2 minutes).
