# M5: demonstration dataset collection

Code: `src/letorifighter/data/{collect,policies,shards,leakage,card}.py`, scripts `scripts/collect_dataset.py` and `scripts/make_dataset_card.py`, configs `configs/collect_m5_scripted.yaml` and `configs/collect_m5.yaml`. Tests: `tests/test_collect*.py` (the default set needs neither torch nor lerobot; `--group vla` adds the round-trip through `VLADataset`; `-m native` collects from the real game).

## What is recorded

Per turn of an episode (the learner is `player_0`, `turnframes=5`, one step per turn, nominal 10 fps in the LeRobot metadata):

| Column | Content |
|---|---|
| `observation.images.front` | RGB frame of the observation before the action, `capture_size` x `capture_size` (default 224), stored as mp4 video |
| `observation.state` | `flatten_observation` of the learner's observation, **300 float32 values**: for `self` then `opponent` limb positions (21x3, groin-centred, in the learner's groin frame), velocities (21x3), groin height (1), joint states (20, values 1..4 as the game reports them), grips (2); then `match_frame` (from the task start) and `frames_next_turn`. (`docs/m4-ppo.md` says 272 in one place; the code constant `OBS_DIM` is 300, which is what `VLADataset` reads.) |
| `action` | 22 states in 0..3 (20 joints, 2 grips) taken at that turn, float32 |
| task string | the episode's instruction, a paraphrase from the TRAIN paraphrases of the task |
| `info.success` | the task detector has latched at this step's observation (0/1) |
| `info.stage` | index of the current detector stage (0 for single-stage tasks) |
| `info.has_action` | 1 for demos, 0 for action-free clips (their `action` column is zero fill) |
| `info.task_id` | task id of the definitions |

An episode of `horizon_turns` turns has that many steps; the terminal observation has no action and is not stored. The random pose pre-roll (K=2 poses) is not recorded: the episode starts at the posed state.

**Failures are kept** (episodes whose detector never fired) and labelled `failure` in the sidecar; success rate per task, source and policy is in the dataset card, so a consumer can train on successes only, on everything, or weight them.

**Action-free clips**: sources with `action_free: true` go into a separate LeRobot dataset (`clips`) with the same schema, zero actions and `info.has_action = 0`, intended for video/world-model co-training (UWM, Fast-WAM style). Their frames and states are real play (any policy mix, typically more random and exploratory). They are kept out of `demos` so a policy loss never trains on their zero actions.

**Sidecar** `episodes.jsonl` (one JSON object per episode, merged and shard level): `spec` (full `EpisodeSpec`: split, index, task, task_id, instruction, paraphrase id, opponent, preroll_turns, pose_seed, episode_seed, horizon_turns), `task`, `opponent`, `pose` (`preroll<K>:<pose_seed>`), `seed`, `split` (always `train`), `source`, `policy_id`, `policy_component` (the acting component of a mixture), `action_free`, `success`, `success_turn`, `label`, `final_stage`, `trivial_start`, `winner`, `injury_delta`, `turns`, `instruction`, `capture_size`, `definitions_version/digest`, `worker`, `wall_seconds`, `dataset` (`demos`/`clips`), `episode_index` (index inside that LeRobot dataset after the merge), `shard`, `segment`, `shard_episode_index`.

## Split discipline

Only `Definitions` v2 `train` is used. `build_plan` refuses any task outside the train split (so no compositions), and every planned spec is checked by `data.leakage.check_spec` on seven axes (split name, episode seed in the train range and in no other split's range, train task, train paraphrase, train opponent pool, train pose pool). The same check runs again on every sidecar record when the dataset card is built, plus a duplicate-seed check and a comparison of the sidecar with the LeRobot `info.json` episode and frame counts. `make_dataset_card.py` exits 1 on any violation. `val`, `test_opponent`, `test_pose`, `test_paraphrase` and `test_composition` episodes can therefore not enter the dataset through this code; the models' own validation episodes must come from `val` via `letorifighter.eval`.

## Policies (`data/policies.py`)

`random`, `hold`, `passive`, `scripted:<behaviour>`, `ppo:<checkpoint>` (argmax) and `ppo-sample:<checkpoint>` (samples the distribution; `PPOPolicy` from `letorifighter.ppo`, needs torch), `{mix: [{policy, weight}, ...]}` (one component per episode, drawn from the episode seed) and `{policy: ..., epsilon: 0.1, noise: 0.05}` for diversity: with probability `epsilon` per turn the whole action is uniformly random, otherwise each of the 22 components is resampled with probability `noise`. All randomness comes from the episode seed, so an episode is replayable. The policy id (`ppo:ppo_multi/best.pt+eps0.1+noise0.05`) is stored per episode.

## Plan, workers, resume, disk guard

`build_plan(config)`: for every source and task (`episodes: {task: n}` or one number for all train tasks) it draws train specs with `Definitions.sample("train", index, plan_seed*1000 + source_index)` until it has `n` with the right task (and an allowed opponent), with episode seeds unique across the whole plan, then interleaves the (source, task) groups round-robin so that stopping early leaves a balanced mix. The plan is a pure function of the config, so every worker computes the same plan and takes the items `i % num_workers == worker_id`.

Output layout (`out_dir`): `config.json`, `shards/w<id>/{episodes.jsonl, segments/<kind>-<n>/}`, then after `merge`: `merged/{demos,clips}/` (LeRobot v3 datasets), `merged/episodes.jsonl`, `merged/manifest.json`. `LeRobotEpisodeWriter` only creates new datasets, so a worker writes a new LeRobot dataset ("segment") every `segment_episodes` episodes (default 50), finalises it, and only then appends its records to the sidecar. A crash loses at most the open segment; on restart unreferenced segment directories are deleted and those episodes are collected again.

- **Resume**: re-run the same command. Episodes whose seed is in any shard's sidecar are skipped (also when the number of workers changed).
- **Disk guard**: before every episode the size of `out_dir` is measured; above `max_disk_gb` (default 15; project cap 50 GB for everything) the worker stops with exit code 3 after closing its segment. The merge needs space for a second copy of the videos (shards plus `merged/`); use `merge --delete-shards` once the merged dataset is verified, or budget twice the data size.
- **Progress log**: one line per episode with cumulative turns/s, episodes/hour and the disk size.
- **Lock**: in the default Steam-launch mode the script takes `/tmp/toribash-game.lock` itself (`fcntl`, blocking, same as `train_ppo.py`) before launching the game; never wrap it in `flock`. `--no-lock` is for direct-launch workers that own their game instance.
- **Per-worker env kwargs (for the ToriLLE instance pool, not merged yet)**: `env_kwargs` (all workers) and `worker_env_kwargs` (a list indexed by `--worker-id`) are passed unchanged to `TwoPlayerToriEnv(...)` on top of the defaults `matchframes`, `turnframes`, `capture_size`, `draw_game=True`; `--env-kwargs '{"display": ":11"}'` sets them for one invocation. When `torille.pool` lands, per-instance `game_dir`, `display` and `capture_display` go in there with no code change; until then a second worker can only run in Steam-launch mode and has to wait for the lock.

## Commands

```bash
uv sync --group vla --group torch --group train   # lerobot, torch (PPO checkpoints), pyyaml

# smoke test: 6 episodes
uv run --group vla python scripts/collect_dataset.py run --config configs/collect_m5_scripted.yaml \
    --set out_dir=data/m5/smoke --max-episodes 6

# the target run (below); resumable, stops at the disk cap
uv run --group vla python scripts/collect_dataset.py run --config configs/collect_m5_scripted.yaml
uv run --group vla --group torch python scripts/collect_dataset.py run --config configs/collect_m5.yaml

# direct-launch workers once torille.pool exists (4 workers, one command each)
for i in 0 1 2 3; do
  uv run ... python scripts/collect_dataset.py run --config c.yaml --worker-id $i --num-workers 4 --no-lock &
done; wait

uv run --group vla python scripts/collect_dataset.py status --config configs/collect_m5.yaml
uv run --group vla python scripts/collect_dataset.py merge  --config configs/collect_m5.yaml [--delete-shards]
uv run python scripts/make_dataset_card.py data/m5/v1 --out docs/m5-dataset-card.md
```

Train from the merged dataset: `scripts/train_vla.py --dataset data/m5/v1/merged/demos --repo-id local/m5-demos ...` (`VLADataset` reads the merged `demos` dataset directly; the `clips` dataset is for video co-training).

## Measurements and the target run

Measured once on the real game on `:0`, Steam launch, capture at 224 px with `draw_game=True`, `scripted:advance_grab` and `random` learners over all five train tasks (13 episodes, 577 turns, horizons 40 to 50, including the game launch and mp4 encoding):

- Throughput: 5.1 turns/s cumulative, 5.5 turns/s after the first episode, about 415 episodes/hour per instance (the machine was shared with other jobs, so expect up to 2x variation). Every episode played its full horizon (40 or 50 turns).
- Disk: 3.4 MB for the 13 episodes in the shards, about **263 MB per 1000 episodes** (about 6 KB per turn; AV1 compresses the mostly static game view well; random-heavy play is the easy case, the number may grow with more motion). The merge duplicates it (3.5 MB merged for the same data), so budget about 0.55 GB per 1000 episodes until the shards are deleted. The 15 GB default cap is therefore far above what throughput allows (about 50k episodes); time is the limit, not disk.
- Merge and card were run on this measured set: the merged dataset loads in LeRobot, the integrity check passed (13 episodes, 0 violations). Success counts of 13 episodes mean nothing; read them from the real run's card.

**Proposed target: 1500 episodes** (5 train tasks x 300, about 72k turns at the horizons): about 3.5 hours on one instance (about 1 hour on 4 direct-launch workers at the same per-instance speed), about 0.4 GB in the shards. Mix per task (300 episodes), `configs/collect_m5.yaml` (with PPO experts) or `configs/collect_m5_scripted.yaml` (without):

| Source | Per task | Purpose |
|---|---|---|
| expert (PPO, argmax) / scripted advance_grab | 90 / 100 | clean demos |
| expert_noisy (PPO sampled, eps 0.05, noise 0.05) / scripted_noisy (eps 0.1, noise 0.1) | 60 / 80 | diversity, recovery behaviour |
| scripted+random mixture / scripted | 40 / 50 | mixed quality |
| random | 30 | failure-heavy play |
| clips (action-free, mixture, eps 0.2) | 50 | frames and state only, `clips` dataset |

Opponents follow the train pools (about 60% advance_grab because keep_distance and escape only use chasers, 20% passive, 20% random), poses are K=0 and K=2 train poses.

Native test (`tests/test_collect_native.py`): **pending**. It ran once and collected real RGB frames, then a later run failed in the script step; the game on `:0` may not be used any more, so it was not investigated or re-run. It is to be run on the Xvfb direct-launch mode once ToriLLE provides it (`worker_env_kwargs` plus `--no-lock`).


## Search-based demonstrations (CEM, random shooting): not implemented, and why

The plan (M5) asks for search-based demonstrations where PPO struggles. This is not implemented because a planner that scores candidate actions needs to evaluate "what happens if I take action sequence A from the current state" many times from the same state. The current ToriLLE environment cannot do that:

1. **No state restore.** `ToribashControl.get_state()` returns positions, velocities, groin rotations, joint states, grips, injuries and frame counters, but there is no way to write a state back into the game. Joint angles, limb velocities and contact/constraint warm-start state of the physics world are internal to the Toribash process, so reconstructing a state from an observation is not possible even in principle (the observation also lacks angular velocities and joint angles).
2. **Reset only after a terminal state.** `reset()` raises unless the episode has ended (`docs/envs.md` of ToriLLE). A candidate rollout that is cut short cannot be abandoned: it has to be played to the end of the match (`LeToriTaskEnv._drain` does exactly this, holding all joints), at about 6 turns/s with capture (17 without drawing).
3. **The only substitute is replay from the start.** The game is deterministic given the reset seed, the pre-roll `(K, pose_seed)` and the action sequence (measured in M3 and by ToriLLE's determinism tests), so a state can be re-created by resetting and replaying the action prefix. Evaluating one candidate then costs one full episode: pre-roll, the prefix, the candidate and the drain to the end of the match. For a 40-turn Reach episode that is about 7 s at 6 turns/s. A closed-loop (receding-horizon) shooter with 16 candidates per decision over 40 decisions is 640 episodes, about 75 minutes of game time for one demonstration, and the cost per decision grows with the prefix length. For comparison the whole 1500-episode dataset is about 3.5 hours.
4. **What is possible is open-loop search over whole action sequences** (CEM over a 40x22 categorical sequence, random shooting of full sequences against a deterministic opponent). It needs one full episode per candidate, so it is success-filtered random play, not planning: random play reaches Reach success in about 16 to 20% of episodes (`docs/m3-tasks-eval.md`), so best-of-N rejection sampling is already available in this dataset by filtering on `label == "success"` of the noisy-policy sources. A CEM with population 32 and 5 iterations is 160 episodes (about 20 minutes per demonstration), which is not scalable to hundreds of demonstrations on one instance and gives sequences that exploit one opponent seed. I did not implement it.

**What would unlock it in ToriLLE** (any one of these is enough, in decreasing order of usefulness):

- `env.save_state()` / `env.load_state(blob)`: a snapshot of the full physics world (all body and joint state plus the game frame) that can be restored mid-episode. Then CEM/random shooting is one restore plus H turns per candidate, about 0.17 s per candidate turn instead of a full episode, and a 16-candidate, 10-turn-horizon planner costs about 30 s per decision instead of 7 s per candidate of full replay. Whether the Toribash binary can expose this is unverified (its Lua API has replay and `edit_game` style functions, which I have not checked for a usable save/load).
- `env.reset(from_turn=k)` / a mid-episode `reset()` that aborts the match instantly and replays a recorded action prefix inside the game (without a Python round trip per turn and without the drain). This keeps the replay-prefix approach but removes the drain and cuts the per-turn bridge overhead for the prefix; it makes the closed-loop shooter about 3x to 5x cheaper but still O(prefix) per candidate.
- `env.rollout(actions_batch)`: evaluate N candidate action sequences from a snapshot in one call, optionally without drawing (`draw_game=False` is about 2x faster, no capture needed for scoring because the success detectors read state only).
- Independent of the above: a state-only fast simulator instance (no rendering) used for planning, with the capture instance only replaying the chosen action sequence to record frames. This works with replay-from-start alone and is the cheapest way to get open-loop search demonstrations today, at the cost described in point 4.

## What is not done

- No PPO expert checkpoint existed when this was written (`data/ppo/` is empty), so the PPO sources of `configs/collect_m5.yaml` were only tested with an untrained checkpoint on the fake env; the measured run uses scripted and random sources. Re-run with experts once `train_ppo.py` has produced them and compare their success rates in the dataset card.
- Search-based demonstrations: see above.
- Cosmos Policy format (256x256 frames plus T5 embeddings, M2) is not written.
- Direct-launch parallel workers were not run (the instance pool is not merged); only the plumbing exists and is tested on the fake env.
- Video storage uses LeRobot's default AV1 mp4 at `fps=10`; frames are lossy-compressed. `VLADataset` decodes them with `pyav` (torchcodec cannot load here, harmless).
- `terminate_on_success` is off, so every episode plays its full horizon even after success (the detector latch is stored per step in `info.success`).
- The card's Toribash build id is read from the Steam app manifest, and the git states are those of the checkouts that ran `make_dataset_card.py`, not those recorded at collection time. Run the card on the machine and commits that collected.
- The merged dataset's `meta/stats` are computed by LeRobot's aggregate; `VLADataset` computes its own standardisation.
