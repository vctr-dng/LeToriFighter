# Toribash-VLA/WAM: Project Plan

Language-conditioned control in a competitive physics game, using VLA and WAM (world action model) foundation models.

Effort estimates are engineering guesses for one developer, not measurements. Paper claims are tagged **V** (read in full or in part via alphaXiv) or **S** (seen in a search listing only). All benchmark numbers are the authors' own.

## 1. Goal

Find out whether pretrained VLA and WAM models help a policy in Toribash, a two-player, contact-rich game with a discrete 22x4 action space and rendered-game video. Concretely, compare against a state-based PPO baseline and a from-scratch model on:

- generalization to unseen initial poses and unseen opponent policies;
- held-out language instructions and held-out compositions of known instructions;
- sample efficiency (pretrained vs scratch);
- whether video prediction (the WAM ingredient) improves the policy, and whether the world model is accurate enough to plan with.

Negative results are acceptable outcomes. The plan is built so that each question has a controlled ablation.

## 2. Constraints and decisions

| Item | Decision |
|---|---|
| Training machine | RTX 5080, 16 GB VRAM, 20 cores, 31 GB RAM |
| Cloud | Not used. If WAM fine-tuning does not fit locally, shrink the WAM (see M7). |
| Video source | Screen capture of the Toribash window on the desktop display `:0` (no RGB exists in ToriLLE). Xvfb was ruled out in M0 because the game opens on the running Steam client's display (see `m0-report.md`). |
| Toribash build | Steam app 248570, build id 20840792 (pinned in M0) |
| Platform | Linux, Steam-native Toribash driven by the reworked ToriLLE (Python 3.14, Gymnasium) |
| Draft file | `docs/project-draft.md` is not modified |
| Repositories | `ToriLLE/` is the environment only. `LeToriFighter/` holds everything that uses the environment (section 2.1). |

### 2.1 Repository split

| `ToriLLE/` (environment only) | `LeToriFighter/` (everything else) |
|---|---|
| Launching and controlling the game (Steam launch, file bridge, cleanup) | Task wrappers: Reach, Control, Escape, success detectors, language templates, splits |
| Gymnasium envs, observation and action spaces, `Dict` observations | Opponent policies (scripted, random, frozen network), self-play and league code |
| Two-player control API (a multi-agent interface over the game) | PPO and other baselines |
| `seed()`, determinism, replay parsing | Dataset collection and writers (LeRobot v3, Cosmos Policy format) |
| Per-turn speed of the bridge | VLA and WAM models, training, evaluation, benchmark and analysis scripts |
| RGB frame capture as an optional observation or `render()` (see M2) | Throughput benchmark (`scripts/m0_throughput.py`) |
| Instance isolation features needed for parallel runs (M1 spike) | Experiment configs, results, MLflow logging |

Rules:

- `ToriLLE/experiments/` is not used, ported or extended. The old PPO, SAC and benchmark scripts there (`async_benchmark.py`, `stable_baselines_experiments/`, `rlkit_sac_experiments/`, `archive/`) are legacy. New equivalents are written fresh in `LeToriFighter/`.
- `LeToriFighter/` depends on ToriLLE as an editable path dependency with a relative path (`torille = { path = "../ToriLLE", editable = true }` in its `pyproject.toml`; `uv.lock` records `editable = "../ToriLLE"`), so no absolute path or user name is committed. Both need Python >=3.14, which they already declare. Done.
- `ToriLLE/` must not import anything from `LeToriFighter/`, and contains no training code, task definitions or policies.
- The two repositories have separate git histories. Environment fixes are committed in ToriLLE; the plan and experiments are committed in LeToriFighter.
- `LeToriFighter/` is a `uv init` scaffold (package `letorifighter`, no commits yet) with ToriLLE as its only dependency, plus `scripts/m0_throughput.py` and `results/m0_throughput_display0.json` from M0.
- Boundary calls made by this plan, open to change: frame capture lives in ToriLLE because it is about observing the game; task wrappers live in LeToriFighter because they define experiments rather than the game.

## 3. Corrections to the draft

These come from reading `ToriLLE/` (Explore pass over the reworked repo).

| Draft claim | Reality |
|---|---|
| Gym API, 4-tuple `step` | Gymnasium; `step` returns `(obs, reward, terminated, truncated, info)` (`torille/envs/gym_env.py:103-137`). |
| 22 action components | 20 joints + 2 grips, 4 states each (`MultiDiscrete([4]*22)`). The Duo env uses 44. |
| Python 3.5/3.6, old binary | Python >=3.14, gymnasium>=1.4, numpy>=2.5. Linux launches `toribash_steam` via `steam -applaunch 248570` and talks to it over a file bridge in `<game dir>/.torille/`. |
| Parallel simulation "needs investigation" | Linux allows one Toribash instance (launch lock; refuses to start if one is running). A real display is needed; there is no true headless mode. |
| 200-500 fps | A 2019 number. Unmeasured now. `draw_game=False` runs about 2x real time. `experiments/async_benchmark.py` is legacy and conflicts with the one-instance rule. |
| PPO examples exist | `experiments/` is unported (TF stable-baselines, old gym, rlkit) and, per section 2.1, will not be used. `examples/` has no PPO. |
| Observations include velocities | The state has 334 values (positions, velocities, groin rotations, joint states, grips, injuries, winner, frame info), but the Gym observations expose only normalised positions. |
| RGB observations | None. `render()` raises. Must be built. |
| Two-player support | No multi-agent API. `DuoToriEnv` is one agent controlling both bodies. |
| Latent action of 16 dimensions | RoboStriker (V) uses 32 dimensions on a unit hypersphere. Treat the size as a hyperparameter. |

## 4. Evidence from the literature

| Finding | Source | Consequence for this plan |
|---|---|---|
| Cosmos Policy fine-tunes Cosmos-Predict2-2B with no architecture change. Actions, proprioception and a value are written into latent frames. Full fine-tuning used 8 to 64 H100s for 48 h per run. Inference needs 7.55 GiB and 0.61 s per chunk on an A100 (5 steps). Code: `github.com/NVlabs/cosmos-policy` (Apache-2.0, custom-dataset path with 256x256 frames and precomputed T5 embeddings). I found no LoRA path in the repo files I read. | V, arXiv 2601.16163 | Local training is the open risk; local inference is fine. A turn-based game fits its single-frame (t, t+K) design. |
| SmolWAM prunes Cosmos Policy to 0.88B (12 of 28 blocks, cached BERT text, one flow step): 95.5% on LIBERO, 2.36 GiB, 57 ms, 3.6 A100-hours. It is distilled from a trained teacher. A LoRA-distilled pruned model reached 87.5%. Adding latent-action tokens, or reward-fine-tuning the world head, collapsed success (53%; 5-17%). Only LIBERO, workshop paper. Code: `github.com/iitimii/SmolWAM`. | V, arXiv 2610.13740 | A pruned trunk is the realistic local WAM size. Do not reward-fine-tune a world head without an action anchor. |
| Fast-WAM (Wan2.2-5B video DiT + 1B action expert): video co-training during training is the dominant benefit; test-time future imagination is not needed (LIBERO 97.6% vs 93.5% without co-training; a real task drops to 10%). 190 ms on an RTX 5090D 32 GB. No embodied pretraining needed. | V, arXiv 2603.16666 | Supports a cheap "video loss on/off" ablation and skipping video decoding at inference. 6B is too large for full fine-tuning on 16 GB. |
| UWM: 12-layer DiT (dim 768), ResNet-18 encoder, frozen SDXL VAE at 224x224, trained from scratch (100k steps, 24 h on 4 A100s). One model acts as policy, forward dynamics, inverse dynamics or video generator by setting per-modality timesteps. It co-trains on action-free video. The paper has no language conditioning. | V, arXiv 2504.02792 | Fits 16 GB. Gives a from-scratch WAM arm. Language must be added. Not a foundation model. |
| DreamZero: 14B, LoRA found suboptimal, a 5B variant is much worse (21% vs 50%), inference 7 Hz on 2 GB200s. Diverse non-repetitive data beats repetitive data (33% to 50%). Failures mostly come from video-generation errors. | V, arXiv 2602.15922 | Out of reach locally. Collect diverse play data and failures, not only expert demos. |
| SmolVLA: 450M parameters (about 100M action expert; VLM frozen in pretraining), flow matching, chunk size 50 (10-50 is best; chunk 1 gives 50% vs 84%), state as a VLM prefix token, 64 visual tokens per frame. Pretrained on SO100 arm data only. Uses 6x less memory than pi0. | V, arXiv 2506.01844 | Fits 16 GB. Transfer from SO100 pretraining to Toribash is uncertain. |
| Action heads on LIBERO: OpenVLA-OFT L1 97.1, Discrete Diffusion VLA 96.4, OFT-discrete 95.5. On a plain Qwen2.5-VL backbone: discrete diffusion 95.1, continuous diffusion 94.9, parallel cross-entropy head 94.2, FAST 91.1, autoregressive 81.2. Discrete Diffusion VLA fine-tunes a 7B backbone. | V, arXiv 2508.20072 | Discrete heads are competitive (1-2 points behind), and Toribash actions are natively categorical, so quantisation loss does not apply. Skip autoregressive and FAST. A parallel 22x4 head is the natural baseline. 7B needs LoRA on 16 GB. |
| RoboStriker: three stages (mocap tracker, 32-d hypersphere latent, latent-space NFSP with eta=0.1 and a 10^6 reservoir). Trained in Isaac Lab with 4096 parallel environments on one RTX 4090. | V, arXiv 2601.22517 | Self-play at that scale is not possible with one Toribash instance. M9 must shrink or use a learned simulator. Toribash replay files can stand in for mocap. |
| Newer efficient WAMs and VLAs: GigaWorld-Policy 2603.17240, Light-WAM 2606.08242, V-JEPA Policy 2609.37250, Long-WAM 2610.10528, Q-WAM 2609.33269, TurboVLA 2607.27205 | S | Shortlist candidates for M7. Verify before use. |

## 5. Milestones

Ordering principle: de-risk the data pipeline first (throughput, video, parallelism), because it bounds every later milestone.

### M0. Foundations and throughput (week 1)

- Repo `ToriLLE/`: run `uv run pytest` and `uv run pytest -m native` against the local Steam install; pin the Toribash build.
- Repo `LeToriFighter/`: measure turns/s and frames/s with `draw_game` on/off and `turnframes` 1 and 5 (`LeToriFighter/scripts/m0_throughput.py`).
- Xvfb was tried and dropped (see `m0-report.md`); everything runs on `:0`.

**Exit:** native tests pass on `:0`; one written throughput number that sizes the M5 dataset. Done, see `m0-report.md` (about 8.6 turns/s with drawing, 17 without, at `turnframes=5`).

### M1. Environment hardening (weeks 1-3)

In `ToriLLE/` (environment):

- Add a `Dict` observation exposing velocities, joint states, grips, `match_frame` and `frames_next_turn` (all in `ToribashControl.get_state()` already).
- Implement `seed()`; test reset determinism and replay determinism.
- Add a two-player control API: both players' actions are supplied from outside and each side gets its own observation (PettingZoo-style). Policies that fill the opponent slot are not part of ToriLLE.
- **Per-turn overhead:** M0 found about 21 ms fixed cost per turn on top of physics time (bridge polling every 10 ms plus Lua file rewrites). Profile it and try a shorter poll interval.
- **Parallelism spike (environment side):** the bridge lives in the game directory, which is 436 MB, so try copies of it. A second instance also needs its own Steam client (one per user), so the spike must try a second Steam login under Xvfb with a separate `HOME`. If N=1, every later data and self-play budget shrinks. This is the largest schedule risk.

In `LeToriFighter/` (uses the environment):

- Opponent policies behind the two-player API: scripted, random, frozen network. Written fresh; nothing is ported from `ToriLLE/experiments/`.
- Multi-instance throughput benchmark, run once the spike works.
- Add ToriLLE as an editable path dependency and a smoke test that steps a ToriLLE env from `LeToriFighter/`.

**Exit:** documented answer for N; deterministic reset test; `Dict` observation env and two-player API in ToriLLE; `LeToriFighter/` imports and steps ToriLLE.

### M2. Video capture pipeline (weeks 3-4)

- In `ToriLLE/`: capture the Toribash window on `:0` with ffmpeg `x11grab` using the window geometry (the desktop is 5120x1440), exposed as an optional RGB observation or `render()`, aligned to turn boundaries using `match_frame`. Fix camera and resolution (the Lua camera API is unchecked). Target 224-256 px.
- In `LeToriFighter/`: write episodes in two formats, a LeRobot v3 dataset (for SmolVLA) and a Cosmos Policy-style set (256x256 frames, precomputed text embeddings).

**Exit:** `(frame_t, state_t, action_t)` aligned within one turn; both formats load in their training code.

### M3. Tasks, rewards and language (weeks 4-5)

All of M3 to M11 live in `LeToriFighter/`, as wrappers and code on top of ToriLLE's public API. If a milestone needs something ToriLLE does not expose, add it to ToriLLE as an environment feature rather than reaching into its internals.

- Tasks: Reach (establish contact), Control (hold a target relative configuration), Escape. Success detectors come from state, not shaped reward.
- Language templates per task, plus held-out compositions (for example Reach then Control), held-out opponents and held-out initial poses.
- Build the evaluation harness (section 6).

**Exit:** detectors working; split definitions frozen.

### M4. PPO expert baselines (weeks 5-8)

- A Gymnasium PPO on `MultiDiscrete` envs, written fresh in `LeToriFighter/` (not from `ToriLLE/experiments/`). Reproduce DestroyUke from the ToriLLE paper as a sanity check. Train per-task experts against passive and scripted opponents.

**Exit:** per-task success above a pre-set threshold; curves and seeds logged in MLflow.

### M5. Demonstration dataset (weeks 7-10)

- Roll out experts with randomised starts and opponents. Record RGB, state, action and language.
- Include diverse play data, failures and action-free clips (DreamZero, UWM and Cosmos Policy all use or benefit from them). Add search-based demonstrations (CEM or random shooting) where PPO struggles.

**Exit:** dataset card, held-out splits, target episode count reached (sized from M0).

### M6. VLA baseline (weeks 10-14)

SmolVLA through LeRobot. Compare three action interfaces:

- (a) **Continuous relaxation** through the pretrained flow-matching expert, rounded to 22x4 at inference. This is the control, closest to the pretrained model.
- (b) **Parallel 22x4 cross-entropy head.**
- (c) **Discrete diffusion head** on a small VLM (SmolVLM scale; the paper's 7B recipe would need LoRA).

Skip autoregressive and FAST heads (worse in the reference ablation).

Ablations: pretrained vs scratch, state-only vs RGB, language vs task id, chunk size k (SmolVLA's 10-50 range is the prior).

**Exit:** the VLA approaches the PPO expert on in-distribution Reach; a sample-efficiency curve exists.

### M7. WAM, local-only (weeks 14-22)

**Gate (week 14): memory smoke test.** Try one Cosmos Policy training step on the 5080: batch 1, 256 px, one camera, cached text embeddings, bf16, gradient checkpointing, 8-bit optimizer, partial or LoRA fine-tuning. Record the outcome either way.

Arms:

- **W1, UWM-style from-scratch WAM.** Fits easily. Add language conditioning. Train with and without the video loss, and with action-free clips. This is the Fast-WAM "video co-training matters" ablation at small scale.
- **W2, Cosmos Policy-derived local WAM.** If the gate passes, fine-tune it. Otherwise prune the trunk SmolWAM-style (about 0.9B) and fine-tune it directly on Toribash data. SmolWAM distills from a trained teacher that this project will not have, so direct fine-tuning of a pruned trunk is an untested substitution and must be treated as an experiment. Released Cosmos Policy checkpoints are useful only for practising local inference.
- **W3 (optional), Fast-WAM-style action decoding:** single forward pass at inference, video loss at training time only.

Toribash interface for latent-frame WAMs: map each joint's four states to {-1, -1/3, 1/3, 1} (or an 88-d one-hot) and round at decode. This is a design choice of this plan, not taken from a paper.

Experiments:

- WAM vs SmolVLA success rate.
- WAM as a world model: rollout error on contact events.
- Value-guided best-of-N using win or injury outcome as the value (Cosmos Policy's planning idea). The paper used 8 GPUs for N=8, so use a small N.

**Exit:** at least one WAM policy with an eval number comparable to M6, and a documented answer on whether a pretrained video prior helps on game rendering. If only W1 runs, the write-up must say the foundation-model WAM claim was not tested.

### M8. Hierarchical and latent actions (weeks 22-27)

- Train a low-level controller conditioned on a latent skill z (LeVERB / RoboStriker style). Try 16-32 dimensions; compare continuous latents, discrete motion tokens and task-space goals.
- Use Toribash replay files as the motion library in place of mocap. Replay parsing is an environment feature that already exists in ToriLLE (`ToriLLE/examples/replay_example.py` shows it); the skill-learning code lives in `LeToriFighter/`.
- The VLA or WAM outputs z.

**Exit:** the hierarchical model beats the end-to-end M6/M7 models on held-out compositions, or a documented negative result.

### M9. Adaptive opponents and self-play (weeks 27-34)

- Scale is set by the M1 result for N. Use a small league of frozen checkpoints with exploiters, or latent-space NFSP.
- Optionally use the WAM as a neural simulator for imagined rollouts. Do not reward-fine-tune its world head (SmolWAM's collapse result).

**Exit:** win-rate matrix across checkpoints; check for cycling and forgetting.

### M10. Stretch: cross-embodiment (week 34 onward)

Reuse the goal representation to condition a separately trained MuJoCo humanoid controller. Start only if M8 succeeds.

### M11. Write-up (in parallel from M6)

Reproducible repo, evaluation tables and ablations.

**Timeline summary:** first meaningful VLA result about week 14 (draft estimated 6-10 weeks, but assumed RGB and parallel simulation were free); first WAM result about week 22; full programme about 8-9 months.

## 6. Evaluation

| Axis | Values |
|---|---|
| Models | PPO (state), vision policy, SmolVLA heads (a, b, c), WAM (W1, W2), hierarchical |
| Generalization | unseen poses, unseen opponents, held-out instructions, held-out compositions, visual occlusion |
| Controls | pretrained vs scratch, video loss on/off, state-only vs RGB |
| Cost | inference latency per turn |

Metrics: task success (state-based detector), win rate, injury delta, WAM rollout error on contact events, exploitability (M9).

## 7. Risks

1. **Throughput and the one-instance limit** cap dataset size and self-play. Mitigation: parallelism spike in M1, tune `turnframes`.
2. **Video alignment and camera**: the in-game camera may move. Fix it through Lua if possible and test alignment.
3. **Local WAM training**: the Cosmos Policy recipe used 8+ H100s. Mitigation: gate at M7 and fall back to W1 and a pruned trunk. State clearly if the foundation-model claim is not reached.
4. **Pretraining may not transfer**: SmolVLA saw one arm type, video models saw robots and real scenes, Toribash is rendered game footage. DreamZero reports that failures come from video errors. Report null results.
5. **Breaking the pretrained action expert** when switching to discrete heads. Keep interface (a) as the control.
6. **Self-play instability and sample scarcity** (M9).
7. **Steam/Toribash build drift**: pin the build and keep the native tests as a canary.
8. **Source quality**: several 2026 WAM papers are preprints or workshop papers with self-reported numbers.

## 8. Reading list

V = read via alphaXiv (some only in part), S = search listing only.

**WAM:** DreamZero 2602.15922 (V), Cosmos Policy 2601.16163 (V), UWM 2504.02792 (V), Fast-WAM 2603.16666 (V), SmolWAM 2610.13740 (V), WAM survey 2609.16074 (V, partial), GigaWorld-Policy 2603.17240 (S), Light-WAM 2606.08242 (S), V-JEPA Policy 2609.37250 (S), Long-WAM 2610.10528 (S), Q-WAM 2609.33269 (S), FLUX 3 Action (S).

**VLA:** SmolVLA 2506.01844 (V), Discrete Diffusion VLA 2508.20072 (V), OpenVLA-OFT 2502.19645 (S; numbers cited here come via the Discrete Diffusion VLA paper), TurboVLA 2607.27205 (S), CombatVLA 2503.09527 (S).

**Hierarchy and latent actions:** RoboStriker 2601.22517 (V), LeVERB 2506.13751 (S), LAPA 2410.11758 (S).

**World models for games:** Dreamer 4 2509.24527 (S), Genie 2402.15391 (S).

**Self-play:** Emergent Complexity via Multi-Agent Competition (S), AlphaStar league (S), Robust Opponent-Aware League Training, NeurIPS 2023 (S).

**Language generalization:** Compositional Instruction Following 2501.12539 (S), BabyAI++ 2004.07200 (S).

**Base environment:** ToriLLE 1807.10110 (S).
