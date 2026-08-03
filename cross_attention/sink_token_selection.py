"""Self-contained sink-token detection for visual tokens.

Pure tensor-in/tensor-out functions with no dependencies beyond torch and the
standard library, so this file can be dropped into another codebase (e.g.
SparseVLM) without pulling in any FastV-specific module.

Shapes are batch-first ([B, ...]) to match how SparseVLM already carries a
batch dimension through its attention/hidden-state tensors, but a batch size
other than 1 is not supported (every VLM inference pipeline this targets
runs one sample at a time, and SparseVLM's own pruning code makes the same
assumption -- see score.py's `mask[0][indices] = 1`).
"""

from __future__ import annotations

from typing import Sequence

import torch

DEFAULT_SINK_TARGET_DIMS = [2533]


def _assert_batch_one(tensor: torch.Tensor, *, name: str) -> None:
    if tensor.shape[0] != 1:
        raise ValueError(f"{name} must have batch size 1, got {tensor.shape[0]}.")


def _resolve_valid_dims(target_dims: Sequence[int], *, hidden_dim: int) -> list[int]:
    valid_dims = [int(d) for d in target_dims if 0 <= int(d) < hidden_dim]
    if not valid_dims:
        raise ValueError(f"No valid target dimensions in {list(target_dims)} for hidden_dim={hidden_dim}.")
    return valid_dims


def compute_hidden_rms_max_sink_scores(
    hidden_states: torch.Tensor,
    target_dims: Sequence[int] = DEFAULT_SINK_TARGET_DIMS,
) -> torch.Tensor:
    """score_i = max_{d in target_dims} |x_i[d] / RMS(x_i)|

    Args:
        hidden_states: [B, N, D] visual-token hidden states.
        target_dims: hidden dimensions treated as sink-related.

    Returns:
        [B, N] sink score per visual token.
    """
    if hidden_states.dim() != 3:
        raise ValueError(f"hidden_states must be [B, N, D], got {tuple(hidden_states.shape)}.")
    _assert_batch_one(hidden_states, name="hidden_states")

    valid_dims = _resolve_valid_dims(target_dims, hidden_dim=hidden_states.shape[-1])

    hidden_states = hidden_states.float()
    mean_square = torch.sum(hidden_states**2, dim=-1) / hidden_states.shape[-1]  # [B, N]
    rms = torch.sqrt(mean_square + 1e-6)
    sink_candidates = torch.abs(hidden_states[..., valid_dims] / rms.unsqueeze(-1))  # [B, N, len(valid_dims)]

    return torch.max(sink_candidates, dim=-1).values  # [B, N]


def compute_question_key_sink_ratio_scores(
    query_state: torch.Tensor,
    key_states: torch.Tensor,
    target_dims: Sequence[int] = DEFAULT_SINK_TARGET_DIMS,
) -> torch.Tensor:
    """SinkRatio_i = (sum_{d in target_dims} q_d * k_i,d) / |q . k_i|

    A question-conditioned alternative to `compute_hidden_rms_max_sink_scores`:
    measures how much of the raw attention logit between the question query
    and a visual key is explained by the sink dimensions alone.

    Args:
        query_state: [B, D] mean question-query vector.
        key_states: [B, N, D] visual keys.
        target_dims: hidden dimensions treated as sink-related.

    Returns:
        [B, N] sink ratio per visual token.
    """
    if query_state.dim() != 2:
        raise ValueError(f"query_state must be [B, D], got {tuple(query_state.shape)}.")
    if key_states.dim() != 3:
        raise ValueError(f"key_states must be [B, N, D], got {tuple(key_states.shape)}.")
    _assert_batch_one(query_state, name="query_state")
    _assert_batch_one(key_states, name="key_states")
    if key_states.shape[-1] != query_state.shape[-1]:
        raise ValueError(
            "query_state and key_states must share the same hidden dimension, got "
            f"{query_state.shape[-1]} and {key_states.shape[-1]}."
        )

    valid_dims = _resolve_valid_dims(target_dims, hidden_dim=query_state.shape[-1])

    query_state = query_state.float()
    key_states = key_states.float()

    sink_logits = torch.sum(
        key_states[..., valid_dims] * query_state[:, valid_dims].unsqueeze(1),
        dim=-1,
    )  # [B, N]
    full_logits = torch.matmul(key_states, query_state.unsqueeze(-1)).squeeze(-1)  # [B, N]
    denom = torch.abs(full_logits).clamp_min(1e-6)
    return sink_logits / denom


def resolve_sink_score_range(
    sink_scores: torch.Tensor,
    *,
    score_min: float | None = None,
    score_max: float | None = None,
    score_quantile: float = 0.95,
) -> tuple[float, float]:
    """Resolve a [score_min, score_max] range for sink-token selection.

    score_max defaults to the observed max; score_min defaults to a quantile
    of the observed scores unless given explicitly.
    """
    flat_scores = sink_scores.detach().float().flatten()
    if flat_scores.numel() == 0:
        raise ValueError("Cannot resolve a sink-score range from an empty tensor.")

    if score_min is None:
        quantile = min(max(float(score_quantile), 0.0), 1.0)
        resolved_min = float(torch.quantile(flat_scores, quantile).item())
    else:
        resolved_min = float(score_min)

    resolved_max = float(torch.max(flat_scores).item()) if score_max is None else float(score_max)

    if resolved_min > resolved_max:
        raise ValueError(f"Invalid sink-score range: min={resolved_min} is greater than max={resolved_max}.")
    return resolved_min, resolved_max


def select_sink_tokens(
    sink_scores: torch.Tensor,
    *,
    score_min: float,
    score_max: float,
) -> torch.Tensor:
    """Boolean mask of visual tokens whose sink score falls in [score_min, score_max].

    Args:
        sink_scores: [B, N].

    Returns:
        [B, N] bool mask.
    """
    if sink_scores.dim() != 2:
        raise ValueError(f"sink_scores must be [B, N], got {tuple(sink_scores.shape)}.")
    _assert_batch_one(sink_scores, name="sink_scores")

    scores = sink_scores.detach().float()
    return (scores >= score_min) & (scores <= score_max)


if __name__ == "__main__":
    torch.manual_seed(0)
    hidden_states = torch.randn(1, 576, 4096)
    hidden_states[:, 12, 2533] = 40.0  # inject an obvious sink token

    scores = compute_hidden_rms_max_sink_scores(hidden_states)
    score_min, score_max = resolve_sink_score_range(scores, score_quantile=0.95)
    mask = select_sink_tokens(scores, score_min=score_min, score_max=score_max)
    print(f"scores shape: {tuple(scores.shape)}, range=({score_min:.3f}, {score_max:.3f}), selected: {int(mask.sum().item())}")
    assert bool(mask[0, 12].item()), "the injected sink token should be selected"

    query_state = torch.randn(1, 4096)
    key_states = torch.randn(1, 576, 4096)
    key_states[:, 12, 2533] = 40.0
    ratio_scores = compute_question_key_sink_ratio_scores(query_state, key_states)
    print(f"question_key_sink_ratio scores shape: {tuple(ratio_scores.shape)}")
    print("sink_token_selection smoke test passed")
