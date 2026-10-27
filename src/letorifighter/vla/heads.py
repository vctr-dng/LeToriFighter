"""Interface (b): SmolVLA with a parallel 22x4 cross-entropy head.

`DiscreteExpert` subclasses LeRobot's `VLAFlowMatching` (nothing in
site-packages is patched). Compared with the flow-matching model:

* `action_in_proj`, `action_time_mlp_in/out` and `action_out_proj` are removed.
  There is no noisy action and no flow time. The action expert's input tokens are
  `chunk_size` learned query embeddings (one per chunk position);
* the queries attend to each other bidirectionally (the flow model's action tokens
  are causal) and to the prefix (images, language, state); the prefix does not
  attend to them. The pretrained VLM and the pretrained expert transformer layers
  are reused unchanged; only the query table and the logit head are new;
* the expert output at each query goes through `Linear(expert_hidden, 22 * 4)`,
  giving logits `(B, k, 22, 4)`; the loss is cross-entropy against the integer
  state 0..3 of every component, masked by `action_is_pad`;
* inference is one forward pass and an argmax, no denoising loop.

`DiscreteHeadPolicy` is the matching `SmolVLAPolicy` subclass: `forward(batch)`
returns `(loss, metrics)`, `predict_action_chunk(batch)` returns states `(B, k, 22)`
as a long tensor.
"""
# pyrefly: ignore-errors
# (torch/lerobot/transformers are optional and loosely typed; checked by the tests)

from __future__ import annotations

import torch
import torch.nn.functional as F
from lerobot.policies.common.vla_utils import make_att_2d_masks
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, VLAFlowMatching
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from torch import Tensor, nn

COMPONENTS = 22
NUM_STATES = 4


class ExpertBase(VLAFlowMatching):
    """`VLAFlowMatching` without the flow-matching projections, with a generic
    `encode` that runs prefix + suffix tokens through the VLM and the expert."""

    def __init__(self, config: SmolVLAConfig) -> None:
        super().__init__(config)
        del self.action_in_proj, self.action_out_proj
        del self.action_time_mlp_in, self.action_time_mlp_out

    def encode(
        self, images, img_masks, lang_tokens, lang_masks, state, suffix_embs, suffix_att
    ) -> Tensor:
        """Run prefix + suffix through the VLM/expert; return the expert output at
        the last `suffix_embs.shape[1]` positions as float32 `(B, n, hidden)`."""
        prefix_embs, prefix_pad, prefix_att = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        b, n = suffix_embs.shape[:2]
        suffix_pad = torch.ones(b, n, dtype=torch.bool, device=suffix_embs.device)
        pad_masks = torch.cat([prefix_pad, suffix_pad], dim=1)
        att_masks = torch.cat([prefix_att, suffix_att.to(prefix_att.dtype)], dim=1)
        att_2d = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
        )
        return suffix_out[:, -n:].to(torch.float32)


class DiscreteExpert(ExpertBase):
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
        self.queries = nn.Parameter(torch.randn(config.chunk_size, hidden) * 0.02)
        self.logit_head = nn.Linear(hidden, components * num_states)

    def compute_logits(
        self, images, img_masks, lang_tokens, lang_masks, state
    ) -> Tensor:
        """Logits `(B, k, components, num_states)` in float32."""
        b = state.shape[0]
        k = self.config.chunk_size
        dtype = (
            self.vlm_with_expert.get_vlm_model().text_model.embed_tokens.weight.dtype
        )
        suffix_embs = self.queries.to(dtype)[None].expand(b, -1, -1)
        # 1 starts a new attention block after the prefix, 0 joins the previous one:
        # queries see the prefix and each other, the prefix does not see them.
        suffix_att = torch.zeros(b, k, device=state.device)
        suffix_att[:, 0] = 1
        out = self.encode(
            images, img_masks, lang_tokens, lang_masks, state, suffix_embs, suffix_att
        )
        return self.logit_head(out).float().view(b, k, self.components, self.num_states)


class DiscreteHeadPolicy(SmolVLAPolicy):
    """`SmolVLAPolicy` whose model is a `DiscreteExpert` (interface b)."""

    config_class = SmolVLAConfig
    name = "smolvla_ce"

    def __init__(self, config: SmolVLAConfig, **kwargs) -> None:
        # Skip SmolVLAPolicy.__init__: it would build (and load) a flow model.
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.rtc_processor = None
        self.model = DiscreteExpert(config)
        self.reset()

    def _logits(self, batch: dict[str, Tensor]) -> Tensor:
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        return self.model.compute_logits(
            images,
            img_masks,
            batch[OBS_LANGUAGE_TOKENS],
            batch[OBS_LANGUAGE_ATTENTION_MASK],
            state,
        )

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean"):  # type: ignore[override]
        """Cross-entropy loss against `batch["action_states"]` (B, k, 22) long."""
        logits = self._logits(batch)
        target = batch["action_states"]
        ce = F.cross_entropy(
            logits.reshape(-1, NUM_STATES), target.reshape(-1), reduction="none"
        ).view(target.shape)
        valid = torch.ones_like(ce)
        if "action_is_pad" in batch:
            valid = (~batch["action_is_pad"]).unsqueeze(-1).to(ce.dtype).expand_as(ce)
        loss = (ce * valid).sum() / valid.sum().clamp_min(1)
        return loss, {"loss": loss.item()}

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:  # type: ignore[override]
        """Argmax states `(B, k, 22)` (long) in one forward pass."""
        self.eval()
        return self._logits(batch).argmax(dim=-1)
