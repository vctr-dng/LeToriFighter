# M6: SmolVLA pipeline (interfaces a and b)

Code: `src/letorifighter/vla/`, trainer `scripts/train_vla.py`, synthetic data `scripts/make_synthetic_vla_data.py`. Tests: `tests/test_vla_*.py`. Measurements below were taken on the RTX A2000 8 GB, with the Toribash game not used. Everything here was proven on **synthetic** data only; see "What is untested".

## Install and versions

```bash
uv sync --group vla          # lerobot[smolvla,dataset,training]==0.6.1, mlflow, pyyaml
export LETORIFIGHTER_DATA=$PWD/data   # caches, checkpoints, MLflow (default: ./data, gitignored)
```

LeRobot version decision: **0.6.1** (the version `m6-smolvla-feasibility.md` measured), replacing the 0.4.4 pin of the writer branch.

- 0.6.1 caps `numpy<2.3`; ToriLLE needs `numpy>=2.5.4` on Python 3.14 (no older wheels). `pyproject.toml` therefore has `[tool.uv] override-dependencies = ["numpy>=2.5.4"]`. It is exercised by every vla test and run (numpy 2.5.4 with lerobot 0.6.1 works).
- 0.6.1 splits optional dependencies into extras; the dataset code needs `lerobot[dataset]` (`datasets`) and the trainer `lerobot[training]`, so the group asks for `smolvla,dataset,training`. Resolved torch is 2.10.0+cu128 (the probe had 2.11+cu130); `torch.cuda.is_available()` is True.
- `LeRobotEpisodeWriter` needed no code change: `tests/test_lerobot_writer.py` passes on 0.6.1 (frames as PNG and as AV1 video). The 0.4.4 restriction on extra features of shape `(1,)` is gone (LeRobot's own columns use it) and was removed from the writer, with a test.
- `torchcodec` installs but cannot load (no system `libavutil` for any FFmpeg version); LeRobot logs a long traceback and falls back to `pyav`, which decodes the mp4 files fine. Harmless noise.
- Python 3.14: `SmolVLAPolicy.from_pretrained` fails in draccus (see the feasibility doc). `letorifighter.vla.model` uses the documented workaround: `SmolVLAConfig` built from the checkpoint's `config.json`, weights copied with `load_state_dict(strict=False)` after dropping keys whose name or shape differ.

## Data

`VLADataset(root, chunk_size, ...)` (`vla/data.py`) reads the LeRobot v3 dataset written by `LeRobotEpisodeWriter` (key `observation.images.front`, `observation.state`, `action` as float32 states 0..3, language as the task string). An item is one frame with the next `chunk_size` actions (clamped to the episode end, `action_is_pad` marks the repeated tail, which is excluded from every loss), and the keys SmolVLA expects: `observation.images.front` (3,H,W) in [0,1], `observation.state` (standardised with the train-set mean/std, saved in the checkpoint), `observation.language.tokens/attention_mask` (SmolVLM2 tokenizer, fixed length 48), `action` (k,22) as levels, `action_states` (k,22) long. Frames are decoded once into RAM (`--preload-images`, N*3*H*W bytes; use `--no-preload-images` for datasets that do not fit). The last `--val-episodes` episodes are held out for the offline eval.

Chunking is `vla/chunking.py` (numpy only, unit tested), the level mapping `vla/levels.py`.

## The three interfaces

All use SmolVLA (SmolVLM2-500M truncated to 16 layers plus the 100M action expert) through LeRobot's `SmolVLAConfig` and `VLAFlowMatching`.

- **(a) `--head relaxed`.** Each state s in {0,1,2,3} becomes the level `2s/3 - 1` in {-1,-1/3,1/3,1}; the pretrained flow-matching expert is trained unchanged (MSE on the velocity; 22 components are zero-padded to `max_action_dim=32`). At inference the Euler sampler (`--num-flow-steps`, default 10) gives continuous values and each is rounded to the nearest level (`levels_to_states`). This is stock `SmolVLAPolicy`; LeRobot's normalisation processors are bypassed (levels are already in [-1,1]).
- **(b) `--head ce`** (`vla/heads.py`). `DiscreteExpert` subclasses `VLAFlowMatching` (site-packages untouched). `action_in_proj`, `action_time_mlp_*` and `action_out_proj` are removed; the expert's input is `chunk_size` learned query embeddings, attending to the prefix and bidirectionally to each other; `Linear(expert_hidden, 22*4)` gives logits (B,k,22,4); loss is cross-entropy against the state at each component, masked by `action_is_pad`; inference is one forward pass and an argmax. New parameters: the query table and the logit head (and `state_proj` when the state is larger than 32).
- **(c) `--head diffusion`**: see the section on the discrete diffusion head (branch `feature/vla-discrete-diffusion`).

`--init pretrained` loads `lerobot/smolvla_base` (907 MB; pulls SmolVLM2-500M, 1.9 GB, into the HF cache). `max_state_dim` is set to the dataset's state size when it exceeds 32 (300 for the real observation: `flatten_observation`), so `state_proj.weight` is re-initialised (its bias is kept); everything else of (a) matches the checkpoint, (b) re-initialises 4 tensors. `--init vlm` keeps only the SmolVLM2 backbone (random expert), `--init scratch` is random everywhere (VLM config and tokenizer are still needed from the hub).

## Ablation switches (all `scripts/train_vla.py` flags, also YAML keys of `VLAConfig`)

| Switch | Values |
|---|---|
| `--head` | `relaxed` (a), `ce` (b), `diffusion` (c) |
| `--init` | `pretrained`, `vlm`, `scratch` |
| `--use-images / --no-use-images` | RGB, or state-only (a black image with `padding_mask=False`: the image tokens are masked out of attention) |
| `--conditioning` | `language` (instruction text), `task_id` (the text `"task <id>"` instead; the id map is `text.instruction_task_ids`, atomic instructions only; for synthetic data `synthetic.json`) |
| `--chunk-size` k | any (SmolVLA prior 10 to 50); no measurable memory effect for (a) and (b) |
| `--n-action-steps` | actions executed per forward at rollout, 1..k |
| `--train-mode` | `expert` (VLM frozen, 100M trainable) or `full` (306M trainable; vision encoder and last layers stay frozen as in LeRobot) |
| `--image-size` | resize-with-pad target (256 default: 16 visual tokens; the pretrained default is 512: 64 tokens) |
| `--bf16` | autocast bfloat16 (default on). VLM weights stay bf16 as LeRobot builds them |

Other flags: `--steps --batch-size --lr --warmup-steps --eval-every --eval-frames --save-every --resume --out-dir --run-name --experiment --eval-game-episodes --eval-split`, `--config file.yaml` (any `VLAConfig` field; CLI overrides YAML).

## Commands

```bash
export LETORIFIGHTER_DATA=$PWD/data
# 1. synthetic learnable dataset (48 episodes x 24 steps, 64x64 frames, 300-d state)
uv run --group vla python scripts/make_synthetic_vla_data.py --out $LETORIFIGHTER_DATA/synthetic/both
# 2. train: (a) and (b)
R="uv run --group vla python scripts/train_vla.py --repo-id local/synthetic --batch-size 32 --lr 3e-4 --eval-every 50 --dataset $LETORIFIGHTER_DATA/synthetic/both"
$R --head relaxed --steps 200 --run-name proof-a-both
$R --head ce      --steps 100 --run-name proof-b-both
# 3. real data (once M5 exists): same script, --dataset <root> --repo-id <id>, defaults chunk 10, lr 1e-4
# 4. inspect runs
uv run --group vla mlflow ui --backend-store-uri sqlite:///$LETORIFIGHTER_DATA/mlruns/mlflow.db
# 5. evaluate a trained policy in the game (needs Toribash; wrap in flock like the other eval commands)
flock /tmp/toribash-game.lock uv run --group vla python -m letorifighter.eval --split val --episodes 20 \
    --policy vla:$LETORIFIGHTER_DATA/runs/proof-b-both/final --out results/vla_eval.json
# 6. tests
uv run pytest                                  # default set; vla tests skip without torch/lerobot
uv run --group vla pytest                      # + tiny-model tests (CPU, no downloads)
uv run --group vla pytest -m gpu               # + pretrained SmolVLA end-to-end on the GPU
```

Outputs under `$LETORIFIGHTER_DATA`: `hf/` (HF cache), `runs/<name>/` (`config.json`, `metrics.jsonl`, `summary.json`, `final/model.pt`, `last/` while running), `mlruns/mlflow.db`, `synthetic/`. `--resume` continues from `last/` (model, optimizer, scheduler, step); `last/` is deleted when a run completes. `--eval-game-episodes N` plays N episodes through `letorifighter.eval.evaluate` at every eval point, holding `/tmp/toribash-game.lock`; it is not tested because it needs the game.

## Policy adapter and chunk execution

`letorifighter.vla.adapter.VLAPolicy(obs) -> action` (22 ints) is a callable for `letorifighter.eval.evaluate`; `load_policy(path)` builds it from `<run>/final`, and `make_policy("vla:<path>")` / `python -m letorifighter.eval --policy vla:<path>` use it. The input is the `LeToriTaskEnv` observation: state = `flatten_observation(obs)` (300 values: the learner's and the opponent's positions, velocities, groin height, joint states, grips, `match_frame`, `frames_next_turn`), text = `obs["instruction"]` (or `task_id`), image = optional `obs["image"]` uint8 (H,W,3) (RGB capture is M2 and does not exist yet; a model trained with images raises a clear error if it is missing).

Receding horizon: a forward pass predicts `chunk_size` actions; the first `n_action_steps` are queued and played one per turn without looking at new observations; when the queue is empty the model is called again. `n_action_steps=1` replans every turn (closed loop), `n_action_steps=chunk_size` plays the full chunk open loop. Flow noise of (a) comes from a generator seeded by `reset(seed)` (the harness calls `reset(episode_seed)`), so episodes are reproducible; (b) is deterministic.

## Synthetic proof (no game)

Data (`vla/synthetic.py`): a hidden scene advances by one each step (8 scenes); the episode's instruction picks one of 2 random lookup tables (scene -> 22 actions in 0..3); the scene is visible as a bright square on a noisy 64x64 image and/or in a 300-d state (random projection plus noise); a chunk of k=10 future actions is a deterministic function of the current scene. 48 episodes x 24 steps, last 4 episodes held out (96 frames). Chance per component is 25%; a model that knows the instruction but not the scene scores about 42% (table marginals), measured below as a control.

Settings: pretrained init, bf16 autocast, batch 32, lr 3e-4, warmup 20, cosine decay, chunk 10, 256 px, 1 camera. `acc` = rounded per-component accuracy over all valid chunk positions on held-out frames; `exact_first` = all 22 components of the first action right. "wall" is the training loop including 2 to 4 evals (dataset decode and model load add about 30 to 50 s). Peak is `torch.cuda.max_memory_allocated`.

| Run | Trainable | Steps | Wall s | Peak GB | final loss | held-out acc | exact_first |
|---|---|---|---|---|---|---|---|
| (a) relaxed, state+image, language | 100M | 200 | 181 | 2.98 | 0.069 | 0.977 | 0.844 |
| (b) CE, state+image, language | 98M | 100 | 93 | 2.96 | 0.014 | 1.000 | 1.000 |
| (b) task-id conditioning | 98M | 100 | 112 | 2.96 | 0.017 | 1.000 | 1.000 |
| (b) image only (state is noise) | 98M | 100 | 87 | 2.96 | 0.053 | 0.999 | 1.000 |
| (b) state only (image masked) | 98M | 100 | 86 | 2.98 | 0.014 | 1.000 | 1.000 |
| (b) `--init vlm --train-mode full` | 305M | 100 | 93 | 4.31 | 0.016 | 1.000 | 1.000 |
| (b) `--init scratch --train-mode full` | 305M | 100 | 97 | 4.67 | 0.027 | 1.000 | 1.000 |
| (a) `--init scratch --train-mode full` | 306M | 200 | 196 | 4.69 | 0.099 | 0.947 | 0.667 |
| control (b): no scene information, language only | 98M | 100 | 81 | 2.97 | n/a | 0.426 | 0.000 |

Notes: (a) is slower to fit than (b) (0.31 -> 0.50 -> 0.90 -> 0.98 accuracy at steps 50/100/150/200; the first attempt with lr 1e-4, batch 16 reached only 0.31 after 150 steps, so the SmolVLA default lr is too low for a budget this short). Step time is 0.7 s at batch 32 (about 45 samples/s). The synthetic task is too easy to separate pretrained from scratch (scratch reaches 100% too), so these runs prove that the pipeline learns and that every switch works, not that pretraining helps. Memory is the 2.2 to 3 GB of the feasibility doc plus batch-32 activations; full fine-tuning peaks at 4.7 GB.

Disk: HF cache 2.8 GB; a finished run is 0.9 GB (`final/model.pt`, bf16 VLM plus fp32 expert); `last/` adds 1.2 to 2.5 GB while a run is in progress.

## What is untested

- **Learning quality on real Toribash data**: nothing here says whether SmolVLA, pretrained or not, learns Reach from rendered game frames. The synthetic data is trivially separable.
- The pretrained SO100 prior is not shown to help (the synthetic task cannot show it).
- The in-game eval hook and `--policy vla:<path>` were not run (they need Toribash); the adapter is tested with a tiny model on fake observations only.
- RGB observations from the game do not exist yet; the real dataset's frame size and fps decide `--image-size` and the preload memory.
- Flow-matching (a) may need far more steps on real data; (b) predicts each of the 22 components independently given the observation (no joint modelling inside a chunk step).
- Chunk size and `n_action_steps` were not tuned; video-decoded frames (AV1) were only tested at 32x32 in the writer test, and the training runs used PNG frames.
