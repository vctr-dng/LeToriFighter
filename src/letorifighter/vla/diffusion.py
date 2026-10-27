"""Interface (c): masked-token discrete diffusion over the 22 x 4 action tokens.

Follows Discrete Diffusion VLA (arXiv 2508.20072) at SmolVLA scale. Each of the
`k * 22` entries of an action chunk is one token with vocabulary
{0, 1, 2, 3, MASK} (the Toribash joint states plus a mask symbol; Toribash
actions are natively categorical, so there is no binning):

* the action expert's input is the embedding of every token: `token_emb[token]`
  plus a factorised position embedding `step_emb[j] + comp_emb[c]`. The queries
  attend to the prefix (images, language, state) and bidirectionally to each other
  (the paper replaces the causal action attention by a bidirectional one); the
  pretrained VLM and expert layers are reused, the flow-matching projections are
  removed, and a shared `Linear(hidden, 4)` head gives the 4-way logits;
* training (paper eq. 5): draw a mask ratio from the cosine schedule
  `gamma = cos(pi/2 * r)`, `r ~ U(0,1)` per sample, mask that fraction of the valid
  tokens (at least one), and take the cross-entropy at the masked tokens only.
  Padded chunk steps (beyond the episode end) are always `MASK` and never in the
  loss. Deviation from eq. 5: masked tokens are weighted equally, not by gamma;
* inference: start from an all-`MASK` chunk and run `T` rounds. In round t the
  model scores every still-masked token and commits the most confident ones
  (max-confidence, argmax, no sampling) so that `floor(L * cos(pi/2 * t/T))`
  tokens remain masked; the last round commits everything. The prefix is run once
  and its key/value cache is reused. Not implemented: the paper's tempered Gumbel
  sampling and secondary re-masking (threshold check).

This is not the paper's 7B recipe: the backbone is SmolVLM2-500M (16 layers) with
the 100M action expert, trained on whatever data you give it.
"""
# pyrefly: ignore-errors
# (torch/lerobot/transformers are optional and loosely typed; checked by the tests)

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from lerobot.policies.common.vla_utils import make_att_2d_masks
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from torch import Tensor, nn

from .heads import COMPONENTS, NUM_STATES, ExpertBase

MASK_ID = NUM_STATES  # token 4


def cosine_keep_masked(total: int, step: int, steps: int) -> int:
    """Number of tokens that remain masked after round `step` of `steps`."""
    return math.floor(total * math.cos(math.pi / 2 * step / steps))


def sample_mask(valid: Tensor, generator: torch.Generator | None = None) -> Tensor:
    """Random mask over the valid tokens `(B, k, C)` with a per-sample cosine ratio
    (at least one masked token per sample that has a valid token)."""
    b = valid.shape[0]
    flat_valid = valid.reshape(b, -1)
    ratio = torch.cos(
        math.pi / 2 * torch.rand(b, device=valid.device, generator=generator)
    )
    count = (ratio * flat_valid.sum(dim=1)).ceil().clamp(min=1)
    scores = torch.rand(flat_valid.shape, device=valid.device, generator=generator)
    scores = scores.masked_fill(~flat_valid, 2.0)  # invalid tokens rank last
    rank = scores.argsort(dim=1).argsort(dim=1)
    return ((rank < count[:, None]) & flat_valid).view_as(valid)


class DiffusionExpert(ExpertBase):
    def __init__(
        self,
        config: SmolVLAConfig,
        components: int = COMPONENTS,
        num_states: int = NUM_STATES,
    ) -> None:
        super().__init__(config)
        self.components = components
        self.num_states = num_states
        hidden = self.vlm_with_expert.expert_hidden_size
        k = config.chunk_size
        self.token_emb = nn.Embedding(num_states + 1, hidden)
        nn.init.normal_(self.token_emb.weight, std=0.02)
        self.step_emb = nn.Parameter(torch.randn(k, hidden) * 0.02)
        self.comp_emb = nn.Parameter(torch.randn(components, hidden) * 0.02)
        self.logit_head = nn.Linear(hidden, num_states)

    def _suffix(self, tokens: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        """Embeddings `(B, k*C, H)` and attention markers `(B, k*C)` of a token grid."""
        b, k, c = tokens.shape
        position = self.step_emb[:, None, :] + self.comp_emb[None, :, :]  # (k, C, H)
        embs = (self.token_emb(tokens) + position[None]).reshape(b, k * c, -1)
        # 1 starts a block after the prefix, 0 joins it: bidirectional among tokens.
        att = torch.zeros(b, k * c, device=tokens.device)
        att[:, 0] = 1
        return embs.to(dtype), att

    def _dtype(self) -> torch.dtype:
        return self.vlm_with_expert.get_vlm_model().text_model.embed_tokens.weight.dtype

    def compute_logits(
        self, images, img_masks, lang_tokens, lang_masks, state, tokens: Tensor
    ) -> Tensor:
        """Logits `(B, k, C, 4)` for a partially masked token grid `(B, k, C)`."""
        embs, att = self._suffix(tokens, self._dtype())
        out = self.encode(images, img_masks, lang_tokens, lang_masks, state, embs, att)
        b, k, c = tokens.shape
        return self.logit_head(out).float().view(b, k, c, self.num_states)

    @torch.no_grad()
    def decode(
        self, images, img_masks, lang_tokens, lang_masks, state, steps: int
    ) -> Tensor:
        """Iterative max-confidence unmasking; states `(B, k, C)` long."""
        b = state.shape[0]
        k, c = self.config.chunk_size, self.components
        total = k * c
        prefix_embs, prefix_pad, prefix_att = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_2d = make_att_2d_masks(prefix_pad, prefix_att)
        prefix_pos = torch.cumsum(prefix_pad, dim=1) - 1
        _, past = self.vlm_with_expert.forward(
            attention_mask=prefix_2d,
            position_ids=prefix_pos,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        prefix_len = prefix_pad.shape[1]
        offsets = prefix_pad.sum(dim=-1)[:, None]
        suffix_pad = torch.ones(b, total, dtype=torch.bool, device=state.device)
        tokens = torch.full((b, k, c), MASK_ID, dtype=torch.long, device=state.device)
        for step in range(1, steps + 1):
            embs, att = self._suffix(tokens, self._dtype())
            suffix_2d = make_att_2d_masks(suffix_pad, att)
            full = torch.cat(
                [prefix_pad[:, None, :].expand(b, total, prefix_len), suffix_2d], dim=2
            )
            position_ids = offsets + torch.cumsum(suffix_pad, dim=1) - 1
            outputs, _ = self.vlm_with_expert.forward(
                attention_mask=full,
                position_ids=position_ids,
                past_key_values=past,
                inputs_embeds=[None, embs],
                use_cache=True,
            )
            past.crop(prefix_len)
            logits = self.logit_head(outputs[1][:, -total:].float())
            confidence, guess = logits.softmax(dim=-1).max(dim=-1)  # (B, total)
            flat = tokens.view(b, total)
            masked = flat == MASK_ID
            n_commit = int(masked[0].sum()) - cosine_keep_masked(total, step, steps)
            if n_commit > 0:
                confidence = confidence.masked_fill(~masked, -1.0)
                chosen = confidence.topk(n_commit, dim=1).indices
                flat.scatter_(1, chosen, guess.gather(1, chosen))
        return tokens


class DiffusionHeadPolicy(SmolVLAPolicy):
    """`SmolVLAPolicy` whose model is a `DiffusionExpert` (interface c)."""

    config_class = SmolVLAConfig
    name = "smolvla_dd"
    decode_steps = 4

    def __init__(self, config: SmolVLAConfig, **kwargs) -> None:
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.rtc_processor = None
        self.model = DiffusionExpert(config)
        self.reset()

    def _inputs(self, batch: dict[str, Tensor]):
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        return (
            images,
            img_masks,
            batch[OBS_LANGUAGE_TOKENS],
            batch[OBS_LANGUAGE_ATTENTION_MASK],
            state,
        )

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean"):  # type: ignore[override]
        """Masked-token cross-entropy against `batch["action_states"]` (B, k, 22)."""
        target = batch["action_states"]
        if "action_is_pad" in batch:
            valid = (~batch["action_is_pad"]).unsqueeze(-1).expand_as(target)
        else:
            valid = torch.ones_like(target, dtype=torch.bool)
        mask = sample_mask(valid)
        tokens = torch.where(mask | ~valid, torch.full_like(target, MASK_ID), target)
        logits = self.model.compute_logits(*self._inputs(batch), tokens)
        ce = F.cross_entropy(
            logits.reshape(-1, NUM_STATES), target.reshape(-1), reduction="none"
        ).view(target.shape)
        loss = (ce * mask).sum() / mask.sum().clamp_min(1)
        return loss, {"loss": loss.item()}

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:  # type: ignore[override]
        """Decode `decode_steps` rounds; states `(B, k, 22)` long."""
        self.eval()
        return self.model.decode(*self._inputs(batch), steps=self.decode_steps)
