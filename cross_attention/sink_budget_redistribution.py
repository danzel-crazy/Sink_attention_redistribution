"""Self-contained sink-budget attention redistribution + top-k keep-mask.

No dependencies beyond torch. Batch-first shapes ([B, ...]), B asserted == 1
(matching SparseVLM's own convention -- see score.py's `mask[0][indices] = 1`).

`select_topk_keep_mask`'s return value is a drop-in replacement for
`attn_postprocess_topk`'s returned `mask`
(SparseVLMs/llava/model/language_model/score.py:23-46). Composition sketch,
for a `pruning_loc` layer inside `LlamaDynamicvitModel.forward`
(SparseVLMs/llava/model/language_model/modelling_sparse_llama.py:259):

    from sink_token_selection import compute_hidden_rms_max_sink_scores, resolve_sink_score_range, select_sink_tokens
    from cross_attention_compute import text_to_visual_attention

    sink_scores = compute_hidden_rms_max_sink_scores(visual_hidden_states)
    score_min, score_max = resolve_sink_score_range(sink_scores, score_quantile=0.95)
    sink_mask = select_sink_tokens(sink_scores, score_min=score_min, score_max=score_max)

    # IMPORTANT: do NOT pass mask_visual_local_positions here. redistribute_sink_budget
    # needs the sink columns' *real* probability mass to compute a budget to move --
    # if they were already zeroed by text_to_visual_attention's masking, sink_budget
    # below would compute to 0 and redistribution would silently become a no-op.
    row_probs, _ = text_to_visual_attention(
        attn_logits,  # layer_outputs[2]
        text_query_positions=cur_text_token_idx,
        visual_token_start=v_token_start,
        visual_token_num=v_token_num,
    )
    redistributed = redistribute_sink_budget(row_probs, sink_mask=sink_mask, redistribution_ratio=1.0,
                                              strategy="topk", receiver_count=32)
    mask = select_topk_keep_mask(redistributed, keep_count=sparse_token_list[layer_dict[layer_idx]])
"""

from __future__ import annotations

import torch

REDISTRIBUTION_STRATEGIES = ("topk", "random", "all")


def _assert_batch_one(tensor: torch.Tensor, *, name: str) -> None:
    if tensor.shape[0] != 1:
        raise ValueError(f"{name} must have batch size 1, got {tensor.shape[0]}.")


def select_receivers(
    baseline_scores: torch.Tensor,
    *,
    sink_mask: torch.Tensor,
    strategy: str = "topk",
    receiver_count: int = 0,
    receiver_score_power: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick non-sink visual tokens to receive the redistributed sink budget.

    Args:
        baseline_scores: [B, N] per-visual-token relevance score.
        sink_mask: [B, N] bool, True where a token is a detected sink.
        strategy: "topk" (highest-scoring non-sink tokens), "random", or
            "all" (every non-sink token).
        receiver_count: number of receivers for "topk"/"random"; <= 0 means
            use the number of sink tokens. Ignored for "all".
        receiver_score_power: exponent applied to non-sink scores before
            normalizing into receiver weights (only for "topk"/"all").

    Returns:
        receiver_mask: [B, N] bool, True at chosen receiver positions.
        receiver_weights: [B, N] float, sums to 1 over the True positions,
            0 elsewhere.
    """
    if strategy not in REDISTRIBUTION_STRATEGIES:
        raise ValueError(f"Unsupported strategy={strategy!r}; expected one of {REDISTRIBUTION_STRATEGIES}.")
    if baseline_scores.dim() != 2:
        raise ValueError(f"baseline_scores must be [B, N], got {tuple(baseline_scores.shape)}.")
    _assert_batch_one(baseline_scores, name="baseline_scores")
    _assert_batch_one(sink_mask, name="sink_mask")
    if receiver_score_power <= 0:
        raise ValueError(f"receiver_score_power must be positive, got {receiver_score_power}.")

    scores = baseline_scores.detach().float()[0]  # [N]
    sink = sink_mask.detach().bool()[0]  # [N]
    if not bool(sink.any()):
        raise ValueError("sink_mask selects no sink tokens.")
    candidate_mask = ~sink
    if not bool(candidate_mask.any()):
        raise ValueError("No non-sink visual tokens are available as receivers.")

    num_visual = scores.shape[0]
    candidate_scores = scores.clamp_min(0)
    receiver_mask = torch.zeros(num_visual, dtype=torch.bool, device=scores.device)
    receiver_weights = torch.zeros(num_visual, dtype=torch.float32, device=scores.device)

    if strategy == "all":
        receiver_mask = candidate_mask
        weighted = torch.where(candidate_mask, candidate_scores.pow(receiver_score_power), torch.zeros_like(candidate_scores))
        weight_sum = float(weighted.sum().item())
        if weight_sum <= 0.0:
            receiver_weights = candidate_mask.float() / candidate_mask.sum().clamp_min(1)
        else:
            receiver_weights = weighted / weight_sum
        return receiver_mask.unsqueeze(0), receiver_weights.unsqueeze(0)

    resolved_count = int(receiver_count) if int(receiver_count) > 0 else int(sink.sum().item())
    resolved_count = min(max(resolved_count, 1), int(candidate_mask.sum().item()))

    if strategy == "topk":
        masked_scores = torch.where(candidate_mask, candidate_scores, torch.full_like(candidate_scores, float("-inf")))
        top_values, top_indices = torch.topk(masked_scores, k=resolved_count, largest=True)
        receiver_mask[top_indices] = True
        weighted = top_values.clamp_min(0).pow(receiver_score_power)
        weight_sum = float(weighted.sum().item())
        if weight_sum <= 0.0:
            receiver_weights[top_indices] = 1.0 / resolved_count
        else:
            receiver_weights[top_indices] = weighted / weight_sum
    else:  # random
        candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).flatten()
        perm = candidate_indices[torch.randperm(candidate_indices.numel(), device=scores.device)][:resolved_count]
        receiver_mask[perm] = True
        receiver_weights[perm] = 1.0 / resolved_count

    return receiver_mask.unsqueeze(0), receiver_weights.unsqueeze(0)


def redistribute_sink_budget(
    row_probabilities: torch.Tensor,
    *,
    sink_mask: torch.Tensor,
    redistribution_ratio: float = 1.0,
    strategy: str = "topk",
    receiver_count: int = 0,
    receiver_score_power: float = 1.0,
    baseline_scores: torch.Tensor | None = None,
) -> torch.Tensor:
    """Move sink-column probability mass onto receiver columns, per row.

    Per row: move `redistribution_ratio` of the sink columns' mass onto
    receiver columns (weighted by `receiver_weights`), then
    log-softmax-renormalize the row (keeps the result a valid softmax-shaped
    distribution rather than a raw renormalized sum), then mean over rows.

    Args:
        row_probabilities: [B, T, N], one probability distribution over
            visual tokens per question row (e.g. from
            `cross_attention_compute.text_to_visual_attention`). These must be
            the *unmasked* rows: the sink columns' real probability mass is what
            gets moved, so zeroing them beforehand makes the budget 0.
        sink_mask: [B, N] bool.
        baseline_scores: optional [B, N] relevance score used only to *rank*
            receiver tokens. Defaults to `row_probabilities.mean(dim=1)`
            (the mean over the unmasked rows). Pass a sink-masked, renormalized,
            row-averaged score here to match the LLaVA reference pipeline, which
            selects receivers from the masked baseline while still moving the
            unmasked sink mass.

    Returns:
        [B, N] redistributed relevance score per visual token.
    """
    if not 0.0 <= redistribution_ratio <= 1.0:
        raise ValueError(f"redistribution_ratio must be in [0, 1], got {redistribution_ratio}.")
    if row_probabilities.dim() != 3:
        raise ValueError(f"row_probabilities must be [B, T, N], got {tuple(row_probabilities.shape)}.")
    _assert_batch_one(row_probabilities, name="row_probabilities")
    _assert_batch_one(sink_mask, name="sink_mask")

    if baseline_scores is None:
        baseline_scores = row_probabilities.detach().float().mean(dim=1)  # [B, N]
    else:
        if baseline_scores.dim() != 2:
            raise ValueError(f"baseline_scores must be [B, N], got {tuple(baseline_scores.shape)}.")
        _assert_batch_one(baseline_scores, name="baseline_scores")
        if baseline_scores.shape[-1] != row_probabilities.shape[-1]:
            raise ValueError(
                "baseline_scores and row_probabilities must share the visual-token dimension, got "
                f"{baseline_scores.shape[-1]} and {row_probabilities.shape[-1]}."
            )
        baseline_scores = baseline_scores.detach().float()
    _, receiver_weights = select_receivers(
        baseline_scores,
        sink_mask=sink_mask,
        strategy=strategy,
        receiver_count=receiver_count,
        receiver_score_power=receiver_score_power,
    )

    probabilities = row_probabilities.detach().float().clone()  # [B, T, N]
    sink = sink_mask.detach().float()  # [B, N]

    sink_budget = (probabilities * sink.unsqueeze(1)).sum(dim=-1, keepdim=True) * redistribution_ratio  # [B, T, 1]
    probabilities = probabilities * (1.0 - redistribution_ratio * sink.unsqueeze(1))
    probabilities = probabilities + sink_budget * receiver_weights.unsqueeze(1)

    tiny = torch.finfo(probabilities.dtype).tiny
    redistributed = torch.softmax(torch.log(probabilities.clamp_min(tiny)), dim=-1)  # [B, T, N]
    return redistributed.mean(dim=1)  # [B, N]


def select_topk_keep_mask(scores: torch.Tensor, keep_count: int) -> torch.Tensor:
    """Boolean keep-mask for the top `keep_count` visual tokens by score.

    Drop-in replacement for `attn_postprocess_topk`'s returned `mask`
    (SparseVLMs/llava/model/language_model/score.py:23-46).
    """
    if scores.dim() != 2:
        raise ValueError(f"scores must be [B, N], got {tuple(scores.shape)}.")
    _assert_batch_one(scores, name="scores")
    num_visual = scores.shape[-1]
    keep_count = min(max(int(keep_count), 1), num_visual)

    mask = torch.zeros_like(scores, dtype=torch.bool)
    _, indices = torch.topk(scores[0], k=keep_count, largest=True)
    mask[0, indices] = True
    return mask


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, N = 1, 6, 576

    row_probs = torch.rand(B, T, N)
    row_probs = row_probs / row_probs.sum(dim=-1, keepdim=True)

    sink_local_positions = [10, 20, 30]
    sink_mask = torch.zeros(B, N, dtype=torch.bool)
    sink_mask[0, sink_local_positions] = True
    # Make the sink columns dominate several rows, like a real attention sink would.
    row_probs[:, :, sink_local_positions] += 5.0
    row_probs = row_probs / row_probs.sum(dim=-1, keepdim=True)

    baseline_scores = row_probs.mean(dim=1)
    redistributed = redistribute_sink_budget(
        row_probs, sink_mask=sink_mask, redistribution_ratio=1.0, strategy="topk", receiver_count=8,
    )
    print(f"redistributed shape: {tuple(redistributed.shape)}, sums to: {redistributed.sum(dim=-1).item():.4f}")
    assert torch.allclose(redistributed.sum(dim=-1), torch.ones(1), atol=1e-4)
    assert torch.all(redistributed[0, sink_local_positions] < baseline_scores[0, sink_local_positions]), (
        "sink tokens should lose relative score after full redistribution"
    )
    print("full redistribution reduces sink-token score: verified")

    keep_mask = select_topk_keep_mask(redistributed, keep_count=128)
    print(f"kept {int(keep_mask.sum().item())} of {N} visual tokens")
    assert int(keep_mask.sum().item()) == 128
    print("sink_budget_redistribution smoke test passed")
