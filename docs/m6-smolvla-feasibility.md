# M6 feasibility: SmolVLA on an 8 GB RTX A2000

Question: can SmolVLA (LeRobot) be fine-tuned and run locally for M6 on the machine we actually have (NVIDIA RTX A2000 8 GB Laptop GPU, driver 595.99.02, 31 GB RAM, 20 cores), not the 16 GB RTX 5080 assumed in `project-plan.md` section 2?

**Answer: yes, with large margin.** Fine-tuning needs 1.7 to 6.4 GB peak GPU memory depending on settings (batch 1 to 8, expert-only or full, 1 camera). Inference is about 0.5 s per 50-step action chunk at the default 10 flow steps (bf16), 0.1 s at 1 step. The Toribash game was not used. All numbers are on synthetic random data and measure memory and speed only, not learning quality.

## 1. Environment and install recipe

| Item | Value |
|---|---|
| Python | 3.14.6 (the repo's version). Works, with one workaround for loading the checkpoint (section 2) |
| LeRobot | 0.6.1 (`lerobot[smolvla]`) |
| torch / torchvision | 2.11.0+cu130 / 0.26.0 (CUDA 13.0 wheels pulled by default; `torch.cuda.is_available()` is True, compute capability 8.6) |
| transformers | 5.5.4 |
| draccus | 0.11.6 (the cause of the 3.14 problem) |
| accelerate / safetensors / num2words | 1.15.0 / 0.8.0 / 0.5.14 |
| Not installed | `peft` (not in the smolvla extra; LoRA path exists in the source but is untested here) |

```bash
uv init --bare --python 3.14 && uv add "lerobot[smolvla]"
HF_HOME=<dir> python ...
```

The install resolved cleanly on Python 3.14 with no manual pins. The probe project is throwaway and lives in the session scratchpad (`smolvla-probe`, scripts `bench.py`, `bench2.py`, `grid.sh`, `load314.py`), not in either repo.

Downloads (HF cache): `lerobot/smolvla_base` `model.safetensors` is 907 MB (906,712,520 bytes) plus a 2.3 kB `config.json`. Loading it also fetches the VLM backbone `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` (about 1.9 GB; total HF cache 2.8 GB). Budget about 3 GB of the 50 GB cap.

## 2. Loading the pretrained checkpoint (read from installed source)

Documented path: `SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")` in `lerobot/policies/smolvla/modeling_smolvla.py` (inherits `PreTrainedPolicy.from_pretrained` in `lerobot/policies/pretrained.py`). It reads `config.json` with `draccus.parse(...)`.

**Failure on Python 3.14:** `PreTrainedConfig.from_pretrained` raises `TypeError: typing.Dict[str, lerobot.configs.types.PolicyFeature] | None is not callable` inside draccus 0.11.6 (`argparsing.py` / `field_wrapper.py`). So `from_pretrained` does not work on 3.14 with these versions. Not yet checked whether a newer draccus or LeRobot fixes it, or whether 3.13 does.

**Workaround used (works):** build `SmolVLAConfig` directly from the checkpoint's `config.json` (drop `type`, `input_features`, `output_features`, `device`, `use_amp`, `push_to_hub`, `repo_id`, `private`, `tags`, `license`, `normalization_mapping`; turn the two list fields `resize_imgs_with_padding` and `optimizer_betas` into tuples), pass our own `input_features`/`output_features` as `PolicyFeature` objects, construct `SmolVLAPolicy(cfg)`, then load `model.safetensors` with `load_state_dict(..., strict=False)` after filtering to keys whose shapes match. With 22-dim actions, 1 camera, state 32 all 450.0M parameters matched (0 missing keys), because action and state are padded to 32 inside the model. See `bench.py` for the exact code.

What the base checkpoint expects: 3 cameras `observation.images.camera1..3` at 3x256x256, state dim 6, action dim 6 (SO100 arm), `chunk_size=50`, `n_action_steps=50`, `num_steps=10` flow steps, `max_state_dim=max_action_dim=32`, `resize_imgs_with_padding=(512, 512)`, `tokenizer_max_length=48`, `vlm_model_name=SmolVLM2-500M-Video-Instruct`, `num_vlm_layers=16`, `expert_width_multiplier=0.75`, `train_expert_only=True`, `freeze_vision_encoder=True`.

Batch keys for `policy.forward(batch)` and `policy.predict_action_chunk(batch)`: `observation.images.cameraN` (B,3,H,W floats in [0,1]), `observation.state` (B,S), `observation.language.tokens` (B,48) and `observation.language.attention_mask` (B,48) (in production produced by `make_smolvla_pre_post_processors`, which tokenizes the task string), `action` (B,chunk,A) and optional `action_is_pad` (B,chunk) for training. `forward` returns `(loss, dict)`; `predict_action_chunk` returns (B,chunk,A) with padding removed.

## 3. State and action dimensions

- `max_state_dim=32`, `max_action_dim=32` in `SmolVLAConfig`; `prepare_state` and `prepare_action` zero-pad with `pad_vector`, and the loss and outputs are sliced back to the real action dim (`losses[:, :, :original_action_dim]`, `actions[:, :, :original_action_dim]`).
- **22 action components fit natively** (22 <= 32, padding to 32, no weight shape change). Verified: output shape (1,50,22).
- **State of a few hundred floats does not fit the pretrained `state_proj`** (`Linear(32, hidden)`). Set `max_state_dim` to the needed size; `state_proj` is then re-initialised (the only mismatching tensor). Measured with 256 state dims: 100.10M trainable vs 99.88M, no measurable memory or time change. Alternatively compress the state to <= 32 dims and keep the pretrained projection.
- State is a single prefix token (`embed_prefix`), so its size does not affect sequence length.

## 4. Measurements

Setup: synthetic random tensors with the right shapes, 1 camera (3x256x256 input, resized with padding to 512x512 by default, 64 visual tokens), state 32, 48 language tokens, chunk 50, action dim 22, AdamW, grad clip 10. Each configuration ran in a separate process; 2 warm-up steps then 5 timed steps; `torch.cuda.max_memory_allocated` after reset. Loss values are meaningless (random data) and not reported.

Precision modes:
- `fp32`: whole policy cast with `.float()`, no autocast.
- `bf16` ("mixed"): the model as constructed by LeRobot already holds the VLM in bfloat16 (`torch_dtype="bfloat16"` in `smolvlm_with_expert.py`) and some modules in float32 (params contain both dtypes); we run under `torch.autocast("cuda", bfloat16)`. This is the default and what I recommend.
- Pure bf16 (`.to(torch.bfloat16)` on everything) fails: `RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16`, because `forward` explicitly casts the expert output to float32 before `action_out_proj`. Not pursued.

Trainable parameters: expert-only 99.9M of 450.0M; full 306.5M (the code still freezes the vision encoder, `lm_head`, final norm and the last one or two VLM layers even when `train_expert_only=False`).

### 4.1 Training, 1 camera at 512x512 (default)

| Precision | Trainable | Batch | Peak alloc GB | Peak reserved GB | Step s |
|---|---|---|---|---|---|
| fp32 | expert-only | 1 | 3.25 | 3.36 | 0.67 |
| fp32 | expert-only | 2 | 3.25 | 3.45 | 0.70 |
| fp32 | expert-only | 4 | 3.26 | 3.48 | 0.91 |
| fp32 | expert-only | 8 | 3.86 | 3.94 | 1.25 |
| bf16 (autocast) | expert-only | 1 | 1.67 | 1.78 | 0.34 |
| bf16 (autocast) | expert-only | 2 | 1.68 | 1.81 | 0.54 |
| bf16 (autocast) | expert-only | 4 | 1.76 | 1.87 | 0.65 |
| bf16 (autocast) | expert-only | 8 | 2.25 | 2.29 | 0.71 |
| fp32 | full | 1 | 6.35 | 6.50 | 0.72 |
| fp32 | full | 2 | 6.36 | 6.61 | 1.17 |
| fp32 | full | 4 | 6.38 | 6.53 | 0.99 |
| fp32 | full | 8 | 6.41 | 6.49 | 1.44 |
| bf16 (autocast) | full | 1 | 3.23 | 3.35 | 0.68 |
| bf16 (autocast) | full | 2 | 3.24 | 3.40 | 0.93 |
| bf16 (autocast) | full | 4 | 3.25 | 3.45 | 0.80 |
| bf16 (autocast) | full | 8 | 3.27 | 3.44 | 1.03 |

### 4.2 Other training variants (bf16 autocast)

| Variant | Trainable | Batch | Peak alloc GB | Step s |
|---|---|---|---|---|
| image 256x256 (16 visual tokens) | expert-only | 1 / 4 / 8 | 1.67 / 1.69 / 1.89 | 0.27 / 0.42 / 0.61 |
| image 256x256 | full | 1 / 4 / 8 | 3.23 / 3.24 / 3.26 | 0.72 / 0.72 / 0.76 |
| 3 cameras 512x512 | expert-only | 4 | 2.35 | 0.84 |
| chunk 10 | expert-only | 4 | 1.69 | 0.71 |
| state dim 256 (`max_state_dim=256`) | expert-only | 4 | 1.76 | 0.62 |

A fixed-batch sanity run (150 steps, batch 4, expert-only, lr 1e-4, bf16 and fp32) trained without NaNs or errors; the flow-matching loss on resampled noise is noisy (about 5 to 2), so this shows only that the loop runs, not convergence.

### 4.3 Inference latency per action chunk (batch 1, 50 x 22 actions, `predict_action_chunk`)

| Precision | Flow steps | Latency ms | Peak alloc GB |
|---|---|---|---|
| bf16 (autocast) | 10 (default) | 490 | 0.90 |
| bf16 (autocast) | 5 | 294 | 0.90 |
| bf16 (autocast) | 1 | 108 | 0.90 |
| fp32 | 10 | 985 | 1.75 |
| fp32 | 5 | 764 | 1.75 |
| fp32 | 1 | 323 | 1.75 |
| bf16, image 256 | 10 | 209 | 0.88 |
| bf16, 3 cameras 512 | 10 | 488 | 0.90 |
| bf16, batch 8 | 10 | 1439 | 1.06 |

### 4.4 Gradient checkpointing

Not supported: there is no gradient-checkpointing option in `configuration_smolvla.py`, `modeling_smolvla.py` or `smolvlm_with_expert.py` (the layer loop in `SmolVLMWithExpertModel.forward` is a hand-written Python loop). Not measured, and not needed: peak memory is dominated by weights, gradients and Adam state, not activations (sequence is only about 64 visual + 48 text + 1 state + 50 action tokens). Memory barely grows from batch 1 to 8.

## 5. Caveats and what was not measured

- Step times are noisy: only 5 timed steps per run, a laptop GPU that may throttle, and Python-overhead-bound small batches (for example bf16 expert-only batch 1 at 0.34 s vs batch 2 at 0.54 s is not a real trend). Treat as roughly 0.3 to 1.5 s per step; throughput is about 2 to 11 samples/s. A real run of 20k to 30k steps at batch 8 is roughly 4 to 12 hours; this extrapolation is not measured.
- No real training run or learning-quality check was done (random data). No image-resolution effect on quality measured. 256x256 vs 512x512 changes visual tokens (16 vs 64) and latency, not memory much.
- `torch.compile` (`compile_model=True`), PEFT/LoRA, Real-Time Chunking, and multi-process data loading were not tried. Dataset loading through LeRobot v3 (M5 output) was not exercised; only batch tensors were synthesised.
- Other processes may have used a little GPU memory (about 0.3 GB shown in `nvidia-smi` at the end); the measurements use PyTorch's own allocator counters, so they are per-process.
- The GPU was otherwise idle; if another agent runs the game or other GPU jobs concurrently, memory and speed will differ.

## 6. Recommended M6 settings (local, 8 GB)

- Python 3.14 with `lerobot[smolvla]` 0.6.1, load via the manual config path in section 2 until the draccus issue is fixed (or pin another Python for the VLA environment if that is preferred).
- bf16 autocast, batch 8 (peak about 2.3 GB expert-only, 3.3 GB full), 1 camera at 256x256 or 512x512 (256 is about twice as fast to run; 512 is the pretrained default), `chunk_size` 10 to 50 (chunk size has no measurable memory effect), `num_steps` 10 flow steps at deployment or 5 to trade quality for speed (to be validated).
- Start with expert-only fine-tuning (default; 100M trainable, about 2 GB), and use full fine-tuning as an ablation (still fits, about 3.3 GB). Even fp32 full fine-tuning at batch 8 fits (6.4 GB), though close to the limit if the GPU is shared.
- Action: 22 components pad to 32 natively for interface (a), the continuous relaxation. Normalisation is `MEAN_STD` for state and action by default (`normalization_mapping`); for 4-state discrete values map each component to a continuous value (for example -1, -1/3, 1/3, 1, or one-hot to 88 dims, which still fits only if `max_action_dim` is raised) and round at inference.
- Inference at about 0.5 s per chunk is fine for a turn-based game.

## 7. Where a discrete 22x4 cross-entropy head would go (not implemented)

All in `lerobot/policies/smolvla/modeling_smolvla.py`, class `VLAFlowMatching`:

- Output projection: `self.action_out_proj = nn.Linear(expert_hidden_size, max_action_dim)` (constructed in `__init__`). A discrete head would add a `Linear(expert_hidden_size, 22*4)` (reshaped to 22x4 logits per action token) in its place or alongside it.
- Loss: `forward` builds noisy actions `x_t`, target `u_t = noise - actions`, and uses `F.mse_loss(u_t, v_t)`. A cross-entropy variant would drop the flow-matching noise/time input (no `x_t` in `embed_suffix`; either feed learned/positional query tokens in `action_in_proj` position, or the previous actions), run the expert once, and compute `F.cross_entropy(logits, targets)` with the padding mask `action_is_pad` already handled in `SmolVLAPolicy.forward`.
- Inference: `sample_actions` / `denoise_step` and `euler_integrate` would be replaced by a single forward pass and argmax. `SmolVLAPolicy._get_action_chunk` slices to `action_feature.shape[0]` and the post-processor un-normalises, so both would need to be bypassed for categorical outputs (targets must stay integers, so also skip the `MEAN_STD` action normalisation in `make_smolvla_pre_post_processors`).
- Padding to `max_action_dim` is then irrelevant; targets are integer class ids (22 components x 4 classes).
- The pretrained expert weights are specialised to continuous velocity prediction; `action_in_proj`, `action_time_mlp_*` and `action_out_proj` would be re-initialised, so interface (b) does not reuse them.

## 8. Blockers

None for feasibility. Open items: (1) the draccus incompatibility with Python 3.14 in `PreTrainedConfig.from_pretrained` (worked around, see section 2); (2) no gradient checkpointing, which is not needed at this scale; (3) quality of SO100-to-Toribash transfer is untested and is the real M6 question.
