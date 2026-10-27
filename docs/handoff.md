# Handoff: moving from the 8 GB laptop to the 16 GB GPU machine

Written 2026-10-27 at the end of the first working session (M0 to M6 code). Everything described here is
merged: ToriLLE `dev` and LeToriFighter `main`. Nothing is waiting in an unmerged branch.

## 1. Where things stand

The plan is `docs/project-plan.md`. The goal of this phase was "continue until the M6 VLA baseline is
completed". **M6 is not completed.** All code exists and is tested on synthetic data and on the real game,
but no PPO expert has been trained, no real dataset has been collected, and SmolVLA has never been trained on
Toribash data. The M6 exit criterion (the VLA approaches the PPO expert on in-distribution Reach, with a
sample-efficiency curve) is untested.

| Milestone | State | Where |
|---|---|---|
| M0 foundations | Done | `docs/m0-report.md` |
| M1 environment hardening | Done | ToriLLE: Dict observations, `seed()`, determinism (PR 6), `TwoPlayerToriEnv` (7), per-turn overhead (8), parallelism spike (9), direct launch and `InstancePool` (11). LeToriFighter: opponents, smoke test (PRs 1, 3). |
| M2 video capture | Done | ToriLLE `capture_size`, `rgb` observation, `render()` (PR 10); LeRobot v3 writer (LeToriFighter PR 1). Cosmos Policy format not written (M7). |
| M3 tasks, language, splits, eval | Done, with a caveat | `docs/m3-tasks-eval.md`, splits v2 (v1 pinned). `control_keep_distance` is still partly satisfied by a scripted baseline (advance_grab gets 0.5), so do not draw conclusions from it before checking trained experts. |
| M4 PPO baselines | Code done, **no trained expert** | `docs/m4-ppo.md`. Only a 3k-turn smoke run exists (learned nothing, as expected). |
| M5 dataset | Code done, **no real dataset** | `docs/m5-dataset.md`. Collection run with real RGB frames passed as a native test; 13-episode measurement earlier. |
| M6 VLA | Code done, **not trained on Toribash** | `docs/m6-vla.md`, `docs/m6-smolvla-feasibility.md`. Interfaces (a) relaxed flow matching, (b) parallel 22x4 CE head, (c) discrete diffusion all fit synthetic data. |
| M7 to M11 | Not started | |

## 2. The most important finding for the new machine: rendering speed

- Steam launch on the desktop display `:0` (Intel iGPU) ran about **16 to 20 turns/s** with `draw_game=False`
  and about 8 to 10 with drawing (M0, and ToriLLE `docs/hacking.md`).
- Direct launch on Xvfb renders with **software GL (llvmpipe, on the CPU)**: about **3 turns/s per instance**
  on a loaded laptop, one game process using about 5 cores, `draw_game=False` gives no speedup, and three
  concurrent instances got 1.6 to 1.7 turns/s each (load average about 24, so pessimistic). At 3 turns/s a
  300k-turn PPO run takes about 28 h, not the 5 h in the tables of `docs/m4-ppo.md`.
- The quiet-machine spike (no capture, no pool) reached about 10 turns/s per instance at N=1 and 39 aggregate
  at N=4. A rerun on an idle machine with capture and the pool has **not** been done.
- Why Xvfb was chosen: the user did not want game windows on their desktop. If the new machine can run the game
  on a real display with a GPU (or if a window on that desktop is acceptable), the Steam launch is several
  times faster. Both paths exist:
  - Xvfb (default, guarded): `env -u DISPLAY -u WAYLAND_DISPLAY uv run ...`; the guard in
    `letorifighter/direct.py` refuses `:0` and the starting `$DISPLAY`.
  - Steam launch on a real display: set `LETORIFIGHTER_ALLOW_STEAM=1`, then `--set launch=steam` for
    `train_ppo.py` and `--launch steam` for `collect_dataset.py run`; this takes the global lock
    `/tmp/toribash-game.lock` and allows one instance only.
  - Untried: Xvfb with NVIDIA GL (a headless X server with the NVIDIA driver, or VirtualGL) so direct-launch
    instances get hardware rendering. This is the most promising way to get both parallel instances and speed.
- First thing to do on the new machine: measure turns/s for the options above (`ToriLLE/scripts/pool_benchmark.py`
  and `scripts/turn_throughput.py`), pick one, and re-estimate the M4/M5 wall times.

## 3. Setting up the new machine

1. Install Steam, log in with the account that owns Toribash, install Toribash (app 248570). The pinned build
   is **20840792** (`appmanifest_248570.acf`); a different build may change the physics, so re-run the
   determinism tests. Keep the Steam client running while the game runs (direct launch uses it for IPC).
2. Install `uv` (0.12.x was used), `ffmpeg` and `Xvfb` (package `xvfb`). Python 3.14 comes from uv.
3. Clone the repos **side by side** (LeToriFighter depends on `../ToriLLE` as an editable path dependency):
   ```
   git clone https://github.com/vctr-dng/ToriLLE.git && git -C ToriLLE checkout dev   # default branch is master, work is on dev
   git clone https://github.com/vctr-dng/LeToriFighter.git
   cd LeToriFighter && uv sync --group dev && uv run pytest        # fast, no game, no torch
   ```
4. GPU stack: `uv sync --group vla --group torch --group train`. On the laptop torch resolved to
   2.10.0+cu128. A newer GPU (for example a 50-series card) needs wheels with a matching CUDA compute capability;
   check `torch.cuda.is_available()` and run a small matmul. `[tool.uv] override-dependencies` forces
   `numpy>=2.5.4` because lerobot 0.6.1 caps numpy below 2.3; the vla tests exercise this.
   torchcodec cannot load without system FFmpeg libraries and LeRobot falls back to PyAV (noisy but works).
5. The first SmolVLA use downloads `lerobot/smolvla_base` (907 MB) and SmolVLM2-500M (1.9 GB) into
   `data/hf` (gitignored). `data/` also holds checkpoints and MLflow (`data/mlruns/mlflow.db`); none of it was
   copied, everything is regenerable.
6. `uv run pytest -m native` on the new machine is the canary that the game, bridge and capture work there
   (see section 2 for display options).
7. Pre-commit hooks (ruff, pyrefly, and StyLua in ToriLLE) exist. In a git worktree pyrefly needs the
   worktree's `.venv/bin` on `PATH`.

## 4. What to run next, in order

All commands from `LeToriFighter/`. On Xvfb they run as `env -u DISPLAY -u WAYLAND_DISPLAY uv run ...`.

1. **Reach PPO expert** (`docs/m4-ppo.md` has the full table, thresholds and the shaped curriculum):
   ```
   uv run --group torch --group train python scripts/train_ppo.py --config configs/ppo_reach.yaml
   ```
   Look at the curve at 100k turns. If val success is below 0.3 there, use the curriculum in `docs/m4-ppo.md`
   (passive opponent with `shaping_scale=1.0`, then add opponents). Proposed pass thresholds are guesses
   (Reach at least 0.70). Several seeds and tasks can run at once, each with its own instance, CPU permitting.
   Other tasks: change `name` and `tasks`, as in the table.
2. **Collect the dataset** (target 1500 episodes, 5 tasks x 300; the scripted config runs without experts):
   ```
   uv run --group vla --group torch python scripts/collect_dataset.py run-pool --config configs/collect_m5.yaml --workers 4
   uv run python scripts/make_dataset_card.py <dataset dir> --out docs/m5-dataset-card.md
   ```
   Storage is small (about 263 MB per 1000 episodes in shards, double until shards are deleted). Time is the
   limit. The collector checks that every episode comes from the train split (seven leakage checks).
3. **Train SmolVLA** on the merged dataset (`docs/m6-vla.md` has all switches and ablation flags):
   ```
   uv run --group vla python scripts/train_vla.py --head relaxed|ce|diffusion --dataset <merged/demos> --repo-id <id> --lr 3e-4 ...
   ```
   Measured on the 8 GB card (synthetic data, bf16, 1 camera, chunk 10): (a) and (b) about 3 GB at batch 32,
   (c) needs batch 8 there (batch 32 hit 7.5 GB); full fine-tuning about 4.7 GB. On 16 GB, try larger batches
   and the full fine-tuning ablation. The default lr (1e-4) was too slow for short runs; 3e-4 worked on the
   synthetic data.
4. **Evaluate**: `uv run python -m letorifighter.eval --split val ... --policy vla:<ckpt>` (never tested
   against the game) and the test splits (`test_paraphrase`, `test_opponent`, `test_pose`,
   `test_composition`). Report success with the Wilson interval it prints.
5. Then the M6 ablations (pretrained vs scratch, state-only vs RGB, language vs task id, chunk size) and the
   sample-efficiency curve (train on subsets of the dataset).

## 5. Known problems and things not verified

- **No learning result of any kind on real Toribash data.** Everything "fits" only on synthetic data that is
  too easy to separate pretrained from scratch.
- Steam-mode (default) native tests were **not re-run** after the direct-launch change; `test_native_two_player.py`
  was not run on the final code. Earlier near-final run: 13 passed, 1 timing-based failure
  (`test_bridge_does_not_desync_over_hundreds_of_turns[False-1]`, passed alone). Run the full
  `uv run pytest -m native` in ToriLLE once on the new machine.
- `scripts/calibrate_contact.py` and `scripts/m0_throughput.py` still use the Steam launch without the guard.
- Direct launch starts the game outside the Steam launcher on the user's own licence. **The Toribash and Steam
  terms were not reviewed** (the user chose the opt-in mode knowing this).
- Direct-launch leftovers if Python is SIGKILLed: game and Xvfb processes stay, and
  `torille.pool.remove_stale_pools()` only removes directories. Copies live in `~/.cache/torille/instances`.
- The Steam client must stay running and logged in for any direct-launch instance.
- Contact is a distance proxy (Toribash exposes no contact flags), fitted on 36 episodes at `turnframes=5`.
- Random pre-roll poses can already satisfy a task (2 to 8% of episodes, flagged `trivial_start`).
- `reset()` is allowed only after a terminal state, so there is no mid-episode abort or state restore. This is
  why search-based demonstrations (CEM, random shooting) were not built; `docs/m5-dataset.md` explains it and
  lists the ToriLLE features that would unlock it (`save_state`/`load_state`, mid-episode reset).
- One collection oddity not investigated: with a short `horizon_override` and a K=2 pose pre-roll, one episode
  ended after 3 turns instead of 6.
- `SmolVLAPolicy.from_pretrained` fails on Python 3.14 (draccus); `letorifighter.vla.build_policy` uses a
  documented workaround. Not checked: whether a newer draccus or lerobot fixes it.
- The discrete diffusion head follows arXiv 2508.20072 at SmolVLM2-500M scale, with documented deviations; it
  is unvalidated on real data.
- Disk on the laptop was tight (uv cache about 40 GB, shared with other projects). The project's own cap was
  50 GB.

## 6. Conventions used in this project

- Branch per change, PR, merge; incremental commits; plain commit messages with **no attribution lines**.
  ToriLLE PRs target `dev`, LeToriFighter PRs target `main`. Pushing works over HTTPS with `gh` credentials.
- ToriLLE is environment-only; everything that uses the environment lives in LeToriFighter
  (`docs/project-plan.md` section 2.1). ToriLLE must not import LeToriFighter.
- Game launches in LeToriFighter go through `letorifighter.direct` (guard) and a private `InstancePool`
  instance. Do not run the game on the user's desktop display without asking.
- Success is always computed from game state by the detectors, never from a shaped reward. The split file
  `tasks/splits/v2.json` is frozen and its digest is pinned in a test; change it only by adding v3.

## 7. Merged PRs (for reference)

ToriLLE (`dev`): 5 gitignore, 6 Dict observation and seeding, 7 two-player API, 8 per-turn overhead,
9 parallelism spike, 10 RGB capture, 11 direct launch and `InstancePool`, 12 StyLua hook.
LeToriFighter (`main`): 1 smoke test and LeRobot writer, 2 SmolVLA feasibility, 3 opponents,
4 tasks and eval, 5 PPO and splits v2, 6 VLA pipeline, 7 discrete diffusion head, 8 dataset collection,
9 Xvfb scripts.
