"""Text-to-visual cross-attention scoring (FastV class form).

Ported from cross_attention/cross_attention_compute.py: the pure functions below
are copied verbatim. Instead of re-deriving attention from raw residual-stream
dot products (the previous implementation), `text_to_visual_attention` operates
directly on the model's real post-softmax multi-head attention weights (what
FastV already materializes as `last_layer_attention = layer_outputs[1]`), with a
from-Q/K fallback for flash-attention paths. `cross_attention_importants` is a
thin class wrapper matching the FastV pipeline's construction/state conventions.

Head-averaging note: per head, slicing a full-sequence post-softmax row down
to a fixed column subset (here, the visual span) and renormalizing it to sum
to 1 exactly reconstructs that head's true softmax as if it had only ever
seen those columns -- a standard categorical-conditioning identity
(p_i / sum_V(p_j), for i in V, equals softmax computed with the support
restricted to V from the start, regardless of what logits exist outside V).
That identity holds per head. Averaging these exact per-head
visual-restricted distributions across heads afterwards reproduces exactly
what a from-Q/K computation gives (restrict K to the visual span, softmax
per head, then average over heads) -- so `text_to_visual_attention` needs
only already-materialized attention weights, with no Q/K recapture. Sink
masking is applied as a *second*, separate renormalization on the
head-averaged, per-question-row distribution, not per head before
averaging -- averaging is a mixture, and a mixture does not commute with a
further subset-renormalization, so masking has to happen at the same stage
in both code paths to keep them consistent.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from cross_attention_sink_redistribution.sink_tokens import sink_token_selector


def _assert_batch_one(tensor: torch.Tensor, *, name: str) -> None:
    if tensor.shape[0] != 1:
        raise ValueError(f"{name} must have batch size 1, got {tensor.shape[0]}.")


def _as_long_tensor(positions, *, device: torch.device) -> torch.Tensor:
    if not torch.is_tensor(positions):
        positions = torch.as_tensor(positions, dtype=torch.long)
    positions = positions.to(device=device, dtype=torch.long).flatten()
    if positions.numel() == 0:
        raise ValueError("text_query_positions must not be empty.")
    return positions


def _resolve_visual_mask_columns(
    mask_visual_local_positions,
    *,
    visual_token_num: int,
    device: torch.device,
) -> torch.Tensor | None:
    if mask_visual_local_positions is None:
        return None
    mask_positions = mask_visual_local_positions
    if not torch.is_tensor(mask_positions):
        mask_positions = torch.as_tensor(mask_positions, dtype=torch.long)
    mask_positions = mask_positions.to(device=device, dtype=torch.long).flatten()
    valid = (mask_positions >= 0) & (mask_positions < visual_token_num)
    mask_positions = mask_positions[valid]
    if mask_positions.numel() >= visual_token_num:
        raise ValueError("Masking would remove every visual token.")
    return mask_positions


def _apply_sink_mask_and_renormalize(
    row_probabilities: torch.Tensor,
    mask_positions: torch.Tensor | None,
) -> torch.Tensor:
    if mask_positions is None or mask_positions.numel() == 0:
        return row_probabilities

    row_probabilities = row_probabilities.clone()
    row_probabilities[:, :, mask_positions] = 0.0
    row_sums = row_probabilities.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(row_probabilities.dtype).tiny)
    return row_probabilities / row_sums


def _validate_visual_span(*, visual_token_start: int, visual_token_num: int, key_len: int) -> tuple[int, int]:
    visual_start = int(visual_token_start)
    visual_end = visual_start + int(visual_token_num)
    if visual_start < 0 or visual_end > key_len:
        raise ValueError(f"Visual span [{visual_start}, {visual_end}) exceeds key length {key_len}.")
    return visual_start, visual_end


def text_to_visual_attention(
    self_attn_weights: torch.Tensor,
    *,
    text_query_positions,
    visual_token_start: int,
    visual_token_num: int,
    mask_visual_local_positions=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score visual tokens by how much question text attends to them.

    Args:
        self_attn_weights: [B, H, Lq, Lk], already softmaxed over the full
            key sequence (e.g. FastV's `last_layer_attention = layer_outputs[1]`).
        text_query_positions: absolute query-sequence indices of the
            question "rater" tokens.
        visual_token_start: first key-sequence index of the visual span.
        visual_token_num: number of visual tokens.
        mask_visual_local_positions: visual-local indices (0-based within
            the visual span) to exclude, e.g. detected sink tokens.

    Returns:
        row_probabilities: [B, T, visual_token_num], one distribution over
            the visual span per question row (sink-masked and renormalized
            if a mask was given).
        mean_scores: [B, visual_token_num], row_probabilities averaged over
            the T question rows.
    """
    if self_attn_weights.dim() != 4:
        raise ValueError(f"self_attn_weights must be [B, H, Lq, Lk], got {tuple(self_attn_weights.shape)}.")
    _assert_batch_one(self_attn_weights, name="self_attn_weights")

    device = self_attn_weights.device
    visual_start, visual_end = _validate_visual_span(
        visual_token_start=visual_token_start,
        visual_token_num=visual_token_num,
        key_len=self_attn_weights.shape[-1],
    )
    query_positions = _as_long_tensor(text_query_positions, device=device)
    if int(query_positions.max().item()) >= int(self_attn_weights.shape[-2]):
        raise ValueError(f"text_query_positions exceeds query length {self_attn_weights.shape[-2]}.")

    per_head = self_attn_weights.float().index_select(2, query_positions)  # [B, H, T, Lk]
    per_head_visual = per_head[..., visual_start:visual_end].clone()  # [B, H, T, N]
    head_row_sums = per_head_visual.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(per_head_visual.dtype).tiny)
    per_head_probabilities = per_head_visual / head_row_sums  # exact per-head visual-restricted softmax
    row_probabilities = per_head_probabilities.mean(dim=1)  # [B, T, N], averaged over heads

    mask_positions = _resolve_visual_mask_columns(
        mask_visual_local_positions,
        visual_token_num=visual_token_num,
        device=device,
    )
    row_probabilities = _apply_sink_mask_and_renormalize(row_probabilities, mask_positions)
    mean_scores = row_probabilities.mean(dim=1)  # [B, N]
    return row_probabilities, mean_scores


def text_to_visual_attention_from_qk(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    *,
    text_query_positions,
    visual_token_start: int,
    visual_token_num: int,
    mask_visual_local_positions=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fallback for callers without post-softmax attention weights (e.g.
    flash-attention paths): recomputes logits = Q @ K^T / sqrt(head_dim)
    restricted to the visual keys and softmaxes directly, per head, before
    averaging over heads. Same output contract as `text_to_visual_attention`.

    Args:
        query_states: [B, H, Lq, Dh].
        key_states: [B, H, Lk, Dh].
    """
    if query_states.dim() != 4 or key_states.dim() != 4:
        raise ValueError(
            "query_states/key_states must be [B, H, L, Dh], got "
            f"q={tuple(query_states.shape)}, k={tuple(key_states.shape)}."
        )
    _assert_batch_one(query_states, name="query_states")
    _assert_batch_one(key_states, name="key_states")

    device = query_states.device
    visual_start, visual_end = _validate_visual_span(
        visual_token_start=visual_token_start,
        visual_token_num=visual_token_num,
        key_len=key_states.shape[-2],
    )
    query_positions = _as_long_tensor(text_query_positions, device=device)
    if int(query_positions.max().item()) >= int(query_states.shape[-2]):
        raise ValueError(f"text_query_positions exceeds query length {query_states.shape[-2]}.")

    head_dim = query_states.shape[-1]
    text_queries = query_states.float().index_select(2, query_positions)  # [B, H, T, Dh]
    visual_keys = key_states.float()[:, :, visual_start:visual_end, :]  # [B, H, N, Dh]

    logits = torch.matmul(text_queries, visual_keys.transpose(-1, -2)) / math.sqrt(head_dim)  # [B, H, T, N]
    per_head_probabilities = torch.softmax(logits, dim=-1)  # already restricted to visual keys, per head
    row_probabilities = per_head_probabilities.mean(dim=1)  # [B, T, N], averaged over heads

    mask_positions = _resolve_visual_mask_columns(
        mask_visual_local_positions,
        visual_token_num=visual_token_num,
        device=device,
    )
    row_probabilities = _apply_sink_mask_and_renormalize(row_probabilities, mask_positions)
    mean_scores = row_probabilities.mean(dim=1)  # [B, N]
    return row_probabilities, mean_scores


class cross_attention_importants:
    """Config + per-forward state wrapper around the cross-attention functions above.

    The visual span defaults to the static config (visual_tokens_start_index /
    visual_tokens_length), overridable per call. After `compute_cross_attention`:
      - `row_probabilities`         : [B, T, N] unmasked per-question-row distribution
                                       (fed to sink_attention_redistributor.redistribute).
      - `important_tokens_scores_raw`: [N] unmasked mean over question rows.
      - `important_tokens_scores`    : [N] sink-masked mean (== raw if masking is off).
    """

    def __init__(self, args, sink_selector: Optional[sink_token_selector] = None):
        self.sink_selector = sink_selector if sink_selector is not None else sink_token_selector(args)
        self.enable_sink_masked = getattr(args, "enable_sink_masked", True)

        self.text_tokens_start_index = getattr(args, "text_tokens_start_index", 0)
        self.text_tokens_length = getattr(args, "text_tokens_length", 0)
        self.visual_tokens_start_index = self.text_tokens_start_index + self.text_tokens_length
        self.visual_tokens_length = getattr(args, "visual_tokens_length", 0)

        self.row_probabilities = None
        self.important_tokens_scores_raw = []
        self.important_tokens_scores = []

    def compute_cross_attention(
        self,
        self_attn_weights,
        text_query_positions,
        visual_token_start=None,
        visual_token_num=None,
        sink_local_ids=None,
    ):
        """
        self_attn_weights: [B, H, Lq, Lk] real post-softmax attention (FastV's
            `last_layer_attention`). Use `compute_cross_attention_from_qk` instead
            when only Q/K are available (flash-attention paths).
        text_query_positions: absolute query indices of the question tokens.
        visual_token_start/visual_token_num: default to the static visual-span config.
        sink_local_ids: sink indices local to the visual block, to exclude from the
            *masked* score. The unmasked `row_probabilities` / `important_tokens_scores_raw`
            keep the sink mass intact for the redistributor to move.
        """
        visual_token_start = self.visual_tokens_start_index if visual_token_start is None else visual_token_start
        visual_token_num = self.visual_tokens_length if visual_token_num is None else visual_token_num

        # unmasked - the redistributor needs the sink columns' real mass
        row_probabilities, mean_raw = text_to_visual_attention(
            self_attn_weights,
            text_query_positions=text_query_positions,
            visual_token_start=visual_token_start,
            visual_token_num=visual_token_num,
            mask_visual_local_positions=None,
        )
        self.row_probabilities = row_probabilities
        self.important_tokens_scores_raw = mean_raw[0]

        if self.enable_sink_masked and sink_local_ids is not None and _numel(sink_local_ids) > 0:
            _, mean_masked = text_to_visual_attention(
                self_attn_weights,
                text_query_positions=text_query_positions,
                visual_token_start=visual_token_start,
                visual_token_num=visual_token_num,
                mask_visual_local_positions=sink_local_ids,
            )
            self.important_tokens_scores = mean_masked[0]
        else:
            self.important_tokens_scores = mean_raw[0]

        return self.important_tokens_scores

    def compute_cross_attention_from_qk(
        self,
        query_states,
        key_states,
        text_query_positions,
        visual_token_start=None,
        visual_token_num=None,
        sink_local_ids=None,
    ):
        """Same contract as `compute_cross_attention` but from per-head Q/K states."""
        visual_token_start = self.visual_tokens_start_index if visual_token_start is None else visual_token_start
        visual_token_num = self.visual_tokens_length if visual_token_num is None else visual_token_num

        row_probabilities, mean_raw = text_to_visual_attention_from_qk(
            query_states,
            key_states,
            text_query_positions=text_query_positions,
            visual_token_start=visual_token_start,
            visual_token_num=visual_token_num,
            mask_visual_local_positions=None,
        )
        self.row_probabilities = row_probabilities
        self.important_tokens_scores_raw = mean_raw[0]

        if self.enable_sink_masked and sink_local_ids is not None and _numel(sink_local_ids) > 0:
            _, mean_masked = text_to_visual_attention_from_qk(
                query_states,
                key_states,
                text_query_positions=text_query_positions,
                visual_token_start=visual_token_start,
                visual_token_num=visual_token_num,
                mask_visual_local_positions=sink_local_ids,
            )
            self.important_tokens_scores = mean_masked[0]
        else:
            self.important_tokens_scores = mean_raw[0]

        return self.important_tokens_scores


def _numel(x) -> int:
    return int(x.numel()) if torch.is_tensor(x) else len(x)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, Lq, Dh = 1, 8, 700, 64
    visual_start, visual_num = 35, 576
    Lk = Lq

    query_states = torch.randn(B, H, Lq, Dh)
    key_states = torch.randn(B, H, Lk, Dh)
    logits = torch.matmul(query_states, key_states.transpose(-1, -2)) / math.sqrt(Dh)
    causal_mask = torch.triu(torch.ones(Lq, Lk, dtype=torch.bool), diagonal=1)
    logits = logits.masked_fill(causal_mask, float("-inf"))
    attn_weights = torch.softmax(logits, dim=-1)  # [B, H, Lq, Lk], what a real decoder layer would return

    text_positions = torch.arange(650, 660)

    row_probs_from_weights, mean_from_weights = text_to_visual_attention(
        attn_weights,
        text_query_positions=text_positions,
        visual_token_start=visual_start,
        visual_token_num=visual_num,
    )
    row_probs_from_qk, mean_from_qk = text_to_visual_attention_from_qk(
        query_states,
        key_states,
        text_query_positions=text_positions,
        visual_token_start=visual_start,
        visual_token_num=visual_num,
    )

    print(f"row_probs shape: {tuple(row_probs_from_weights.shape)}, row sums: {row_probs_from_weights.sum(dim=-1).unique()}")
    assert torch.allclose(row_probs_from_weights, row_probs_from_qk, atol=1e-4), (
        "from-attn-weights and from-Q/K paths must agree exactly"
    )
    assert torch.allclose(mean_from_weights, mean_from_qk, atol=1e-4)
    print("from-attn-weights and from-Q/K identity verified")

    from types import SimpleNamespace

    cai = cross_attention_importants(
        SimpleNamespace(enable_sink_masked=True, text_tokens_start_index=0, text_tokens_length=visual_start,
                        visual_tokens_length=visual_num, sink_dims=[2533]),
    )
    scores = cai.compute_cross_attention(attn_weights, text_positions, sink_local_ids=torch.tensor([3, 40, 200]))
    assert cai.row_probabilities.shape == (B, len(text_positions), visual_num)
    assert torch.allclose(scores[torch.tensor([3, 40, 200])], torch.zeros(3), atol=1e-6)
    print("cross_attention_importants wrapper smoke test passed")
