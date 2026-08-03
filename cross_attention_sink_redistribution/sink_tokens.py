"""Sink-token detection for visual tokens (FastV class form).

Ported from cross_attention/sink_token_selection.py: the pure tensor-in/
tensor-out functions below are copied verbatim (batch-first [B, N, D], B == 1),
and `sink_token_selector` is a thin class wrapper that reads config from `args`
and holds per-forward state so it slots into the FastV pipeline
(LlamaModel.fastv_forward) the same way the previous class did.

Two sink-score modes are available (see `sink_score_mode`):
  - "rms_max":            score_i = max_d |x_i[d] / RMS(x_i)|        (hidden-state only)
  - "question_key_ratio": score_i = (sum_d q_d k_i,d) / |q . k_i|   (question-conditioned)
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

# Defaults mirror the offline "sum" sink view in visualization/sink_token_visualizations.py
# so runtime selection matches it: score_i = sum_{d in DEFAULT_SINK_TARGET_DIMS} |x_i[d]/RMS(x_i)|,
# sinks = tokens with score in [quantile, max].
DEFAULT_SINK_TARGET_DIMS = [1415, 2533]
SINK_SCORE_QUANTILE = 0.99
SINK_REDUCTION = "sum"  # "sum" (Σ_d |x/RMS|) or "max" (max_d |x/RMS|)


def _assert_batch_one(tensor: torch.Tensor, *, name: str) -> None:
    if tensor.shape[0] != 1:
        raise ValueError(f"{name} must have batch size 1, got {tensor.shape[0]}.")


def _resolve_valid_dims(target_dims: Sequence[int], *, hidden_dim: int) -> list[int]:
    valid_dims = [int(d) for d in target_dims if 0 <= int(d) < hidden_dim]
    if not valid_dims:
        raise ValueError(f"No valid target dimensions in {list(target_dims)} for hidden_dim={hidden_dim}.")
    return valid_dims


def compute_hidden_rms_sink_scores(
    hidden_states: torch.Tensor,
    target_dims: Sequence[int] = DEFAULT_SINK_TARGET_DIMS,
    reduction: str = SINK_REDUCTION,
) -> torch.Tensor:
    """score_i = reduce_{d in target_dims} |x_i[d] / RMS(x_i)|

    reduction="sum" gives Σ_d |x_i[d]/RMS| (the offline "sum" view); "max" gives max_d |...|.

    Args:
        hidden_states: [B, N, D] visual-token hidden states.
        target_dims: hidden dimensions treated as sink-related.
        reduction: "sum" or "max" over the target dims.

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

    if reduction == "sum":
        return torch.sum(sink_candidates, dim=-1)  # [B, N]
    if reduction == "max":
        return torch.max(sink_candidates, dim=-1).values  # [B, N]
    raise ValueError(f"Unsupported sink reduction: {reduction!r} (expected 'sum' or 'max').")


def compute_hidden_rms_max_sink_scores(
    hidden_states: torch.Tensor,
    target_dims: Sequence[int] = DEFAULT_SINK_TARGET_DIMS,
) -> torch.Tensor:
    """Back-compat alias: max reduction (see compute_hidden_rms_sink_scores)."""
    return compute_hidden_rms_sink_scores(hidden_states, target_dims, reduction="max")


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


class sink_token_selector:
    """Config + per-forward state wrapper around the sink-detection functions above.

    Constructed once from `args` in LlamaModel and reused across generation steps.
    """

    def __init__(self, args):
        # sink score settings
        self.sink_dims = getattr(args, "sink_dims", DEFAULT_SINK_TARGET_DIMS)
        self.sink_min = getattr(args, "sink_score_min", None)
        self.sink_max = getattr(args, "sink_score_max", None)
        self.sink_quantile = getattr(args, "sink_score_quantile", SINK_SCORE_QUANTILE)
        # "rms_max" (hidden-state only) or "question_key_ratio" (question-conditioned)
        self.sink_score_mode = getattr(args, "sink_score_mode", "rms_max")
        # reduction over sink dims for the hidden-state path: "sum" (offline "sum" view) or "max".
        self.sink_reduction = getattr(args, "sink_reduction", SINK_REDUCTION)

        # populated by the most recent selection call
        self.sink_tokens_scores = []
        self.sink_tokens_ids = []
        self.sink_budget = 0.0

    def compute_sink_scores(
        self,
        visual_hidden_states: torch.Tensor,
        query_state: Optional[torch.Tensor] = None,
        key_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """[B, N] sink score per visual token, per self.sink_score_mode.

        visual_hidden_states: [B, N, D] (used by "rms_max").
        query_state/key_states: [B, D] / [B, N, D] (required by "question_key_ratio").
        """
        if self.sink_score_mode == "question_key_ratio":
            if query_state is None or key_states is None:
                raise ValueError("question_key_ratio mode requires query_state and key_states.")
            return compute_question_key_sink_ratio_scores(query_state, key_states, self.sink_dims)
        return compute_hidden_rms_sink_scores(visual_hidden_states, self.sink_dims, self.sink_reduction)

    def select_sink_mask_from_scores(self, sink_scores: torch.Tensor) -> torch.Tensor:
        """[B, N] bool mask from *precomputed* per-visual-token sink scores, using the configured
        quantile/min/max range. Pass the mean in-decoder text-to-visual attention here to select
        sinks by attention (the visual tokens the question text attends to most) instead of by
        hidden-state activation. Also records scores/ids."""
        score_min, score_max = resolve_sink_score_range(
            sink_scores, score_min=self.sink_min, score_max=self.sink_max, score_quantile=self.sink_quantile
        )
        mask = select_sink_tokens(sink_scores, score_min=score_min, score_max=score_max)
        self.sink_tokens_ids = torch.nonzero(mask[0], as_tuple=False).flatten()
        self.sink_tokens_scores = sink_scores.detach().float()[0][self.sink_tokens_ids]
        return mask

    def select_sink_mask(
        self,
        visual_hidden_states: torch.Tensor,
        query_state: Optional[torch.Tensor] = None,
        key_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """[B, N] bool mask over the visual tokens from hidden-state scores; also records scores/ids."""
        sink_scores = self.compute_sink_scores(visual_hidden_states, query_state, key_states)
        return self.select_sink_mask_from_scores(sink_scores)

    def _select_sink_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Back-compat shim: takes an *unbatched* [N, D] visual-token slice (as the
        FastV callers pass `hidden_states[0][visual_slice]`) and returns 1-D sink-token
        indices local to that slice. Uses the rms_max mode."""
        mask = self.select_sink_mask(hidden_states.unsqueeze(0))
        return torch.nonzero(mask[0], as_tuple=False).flatten()


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

    from types import SimpleNamespace

    selector = sink_token_selector(SimpleNamespace(sink_dims=[2533], sink_score_quantile=0.95))
    ids = selector._select_sink_tokens(hidden_states[0])
    assert int(ids.numel()) > 0 and (12 in ids.tolist()), "shim should return the injected sink id"
    print(f"sink_token_selector shim selected {int(ids.numel())} ids; smoke test passed")
