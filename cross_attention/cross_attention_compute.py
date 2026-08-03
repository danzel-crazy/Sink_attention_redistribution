"""Self-contained text-to-visual cross-attention scoring.

Primary entry point (`text_to_visual_attention`) operates directly on
attention weights already produced by a decoder layer's own forward call
(output_attentions=True) -- this is what SparseVLM's `LlamaDynamicvitAttention`
already returns at its pruning-loc layers (`attn_logits = layer_outputs[2]`
in modelling_sparse_llama.py), so no Q/K re-capture/hooking is required. A
from-Q/K fallback (`text_to_visual_attention_from_qk`) is included for
callers that don't have post-softmax attention weights available (e.g. under
flash-attention).

No dependencies beyond torch and math.

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

import torch


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
            key sequence (e.g. SparseVLM's `attn_logits = layer_outputs[2]`).
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

    sink_local_positions = torch.tensor([3, 40, 200])
    masked_row_probs, masked_mean = text_to_visual_attention(
        attn_weights,
        text_query_positions=text_positions,
        visual_token_start=visual_start,
        visual_token_num=visual_num,
        mask_visual_local_positions=sink_local_positions,
    )
    assert torch.allclose(masked_row_probs[:, :, sink_local_positions], torch.zeros(1), atol=1e-6)
    assert torch.allclose(masked_row_probs.sum(dim=-1), torch.ones(1), atol=1e-5)
    print("sink masking + renormalization verified")
