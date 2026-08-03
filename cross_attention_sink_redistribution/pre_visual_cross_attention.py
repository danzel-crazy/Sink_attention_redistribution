from __future__ import annotations

from typing import Optional

import torch

from cross_attention_sink_redistribution.sink_tokens import sink_token_selector
from cross_attention_sink_redistribution.cross_attention import (
    _apply_sink_mask_and_renormalize,
    _resolve_visual_mask_columns,
)


class pre_visual_cross_attention_importants:
    """Single-head text-to-visual cross-attention keyed on the *pre-decoder* visual
    features (the projected image embeddings before they enter the LLaVA decoder),
    with one visual self-attention pass applied first.

        V_self = Softmax(V V^T / sqrt(d)) V     # visual features self-attend once
        logits = Q_text V_self^T / sqrt(d)      # single head, raw residual vectors
        row_probabilities = Softmax(logits)     # over the visual (key) axis, per text row
        visual_scores = mean_text(row_probabilities)

    where ``V`` are the pre-decoder visual features and ``Q_text`` are the text-token
    hidden states at the current decoder layer.

    Output contract matches ``cross_attention.cross_attention_importants`` so it feeds
    the same faithful redistributor:
      - ``row_probabilities``          : [B, T, N] unmasked per-question-row distribution.
      - ``important_tokens_scores_raw``: [N] unmasked mean over question rows.
      - ``important_tokens_scores``    : [N] sink-masked mean (== raw if masking is off).

    Sink masking is applied as a *second* renormalization on the head-averaged row
    distribution (post-softmax), exactly as in ``cross_attention`` - not the pre-softmax
    ``-inf`` masking the previous implementation used - so the two stay consistent.
    """

    def __init__(self, args, sink_selector: Optional[sink_token_selector] = None):

        # sink tokens: reuse a shared selector if the caller already owns one (e.g. LlamaModel),
        # so sink_tokens_ids/scores don't drift between independently-configured copies.
        self.sink_selector = sink_selector if sink_selector is not None else sink_token_selector(args)
        self.enable_sink_masked = getattr(args, "enable_sink_masked", True)

        self.text_tokens_start_index = getattr(args, "text_tokens_start_index", 0)
        self.text_tokens_length = getattr(args, "text_tokens_length", 0)

        self.visual_tokens_start_index = self.text_tokens_start_index + self.text_tokens_length
        self.visual_tokens_length = getattr(args, "visual_tokens_length", 0)

        self.row_probabilities = None
        self.important_tokens_scores_raw = []
        self.important_tokens_scores = []

    def _extract_text_tokens(self, hidden_states, start=None, length=None):
        start = self.text_tokens_start_index if start is None else start
        length = self.text_tokens_length if length is None else length
        return hidden_states[:, start:start + length, :]

    def _extract_visual_tokens(self, visual_hidden_states):
        return visual_hidden_states[:, self.visual_tokens_start_index:self.visual_tokens_start_index + self.visual_tokens_length, :]

    @staticmethod
    def _visual_self_attention(visual_tokens):
        """V_self = Softmax(V V^T / sqrt(d)) V - one self-attention pass over the visual tokens."""
        scale = visual_tokens.shape[-1] ** 0.5
        self_attn_scores = torch.matmul(visual_tokens, visual_tokens.transpose(0, 1)) / scale
        self_attn_weights = torch.softmax(self_attn_scores, dim=-1)
        return torch.matmul(self_attn_weights, visual_tokens)

    def compute_cross_attention(
        self,
        hidden_states,
        visual_hidden_states,
        text_tokens_start_index=None,
        text_tokens_length=None,
        sink_local_ids=None,
    ):
        """
        hidden_states: current decoder hidden states, used to pull the text-token queries.
        visual_hidden_states: the *pre-decoder* embeddings (batched, full sequence) - the visual
            keys/values are sliced out of this with the static visual-block config, so the visual
            source is the projected image features before any decoder layer processes them, not the
            in-decoder hidden states. Pass the layer-0 ``inputs_embeds`` here.
        text_tokens_start_index/text_tokens_length: override the static system-prompt-prefix
            config for this call - e.g. pass the actual question span, since its length varies
            per example and the static config can't track that.
        sink_local_ids: sink-token indices *local to the visual-token block*. If omitted, sink
            tokens are (re)detected here via self.sink_selector. Applied only to the *masked*
            score; the unmasked ``row_probabilities`` / ``..._raw`` keep the sink mass intact for
            the redistributor to move.
        """
        # text tokens as queries (from the decoder), pre-decoder visual tokens as keys/values.
        # Upcast to float32 before any matmul: V @ V^T (below) sums 4096 products of features whose
        # magnitude can reach ~170, which overflows float16's 65504 limit -> inf -> softmax -> NaN,
        # poisoning the whole score vector (and the redistribution receivers it selects). The offline
        # capture path already recomputes this in float32; matching it here keeps them consistent.
        text_tokens = self._extract_text_tokens(hidden_states, text_tokens_start_index, text_tokens_length)[0].float()
        visual_tokens = self._extract_visual_tokens(visual_hidden_states)[0].float()

        # visual features self-attend once before being used as cross-attention keys
        visual_self = self._visual_self_attention(visual_tokens)

        scale = visual_self.shape[-1] ** 0.5
        logits = torch.matmul(text_tokens, visual_self.transpose(0, 1)) / scale  # [T, N]

        # [1, T, N] to match the cross_attention row-probabilities contract; softmax is already
        # restricted to the visual keys (V_self holds only the visual tokens)
        row_probabilities = torch.softmax(logits.float(), dim=-1).unsqueeze(0)
        self.row_probabilities = row_probabilities
        self.important_tokens_scores_raw = row_probabilities.mean(dim=1)[0]  # [N], unmasked

        visual_importance = self.important_tokens_scores_raw
        if self.enable_sink_masked:
            if sink_local_ids is None:
                sink_ids = self.sink_selector._select_sink_tokens(visual_hidden_states[0])
                visual_sink_mask = (sink_ids >= self.visual_tokens_start_index) & (
                    sink_ids < self.visual_tokens_start_index + self.visual_tokens_length
                )
                sink_local_ids = sink_ids[visual_sink_mask] - self.visual_tokens_start_index

            mask_positions = _resolve_visual_mask_columns(
                sink_local_ids,
                visual_token_num=row_probabilities.shape[-1],
                device=row_probabilities.device,
            )
            if mask_positions is not None and mask_positions.numel() > 0:
                # second, separate renormalization on the row distribution (post-softmax),
                # consistent with cross_attention's sink-mask stage
                masked_rows = _apply_sink_mask_and_renormalize(row_probabilities, mask_positions)
                visual_importance = masked_rows.mean(dim=1)[0]

        self.important_tokens_scores = visual_importance
        return visual_importance


if __name__ == "__main__":
    from types import SimpleNamespace

    torch.manual_seed(0)
    D, N = 32, 8
    L = 5 + N + 3  # text prefix (5) + visual (8) + question (3)
    hidden = torch.randn(1, L, D)
    embeds = torch.randn(1, L, D)

    args = SimpleNamespace(enable_sink_masked=True, text_tokens_start_index=0, text_tokens_length=5,
                           visual_tokens_length=N, sink_dims=[3], sink_score_quantile=0.99)
    c = pre_visual_cross_attention_importants(args)
    sink_local = torch.tensor([1, 4])
    out = c.compute_cross_attention(hidden, embeds, text_tokens_start_index=13, text_tokens_length=3,
                                    sink_local_ids=sink_local)
    assert c.row_probabilities.shape == (1, 3, N)
    assert abs(float(c.important_tokens_scores_raw.sum()) - 1.0) < 1e-4
    assert abs(float(out.sum()) - 1.0) < 1e-4
    assert torch.allclose(out[sink_local], torch.zeros(2), atol=1e-6), "sink columns should be zeroed post-mask"
    print("pre_visual_cross_attention_importants smoke test passed; row_probs", tuple(c.row_probabilities.shape))
