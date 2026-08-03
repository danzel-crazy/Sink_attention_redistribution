"""Sink-budget attention redistribution (self-attention-mass form).

This file is kept byte-identical to cross_attention_sink_redistribution/attention_redistribution.py
(the reference implementation) apart from the back-compat block at the bottom, which exists only
so PyramidDrop's existing eval scripts keep running unchanged. Port any future change from there.

The redistributed mass is the model's own self-attention over the visual tokens
(`last_layer_attention_avg_last_tok_image`, a 1-D [N] vector). Cross-attention is
used ONLY to (a) select which non-sink visual tokens receive the freed sink budget
and (b) weight how that budget splits among them - it never contributes an
attention magnitude. Every attention value (base, budget, result) comes from the
self-attention, so the output is a drop-in replacement for the raw attention
vector FastV would otherwise `.topk(...)`:

    A'_{v_i} = A_{v_i} + B * w_i ,   B = sum_{j in sink} A_{v_j} ,   sum_i w_i = 1

where A = image_attention (self-attention) and w = normalized cross-attention over
the chosen receivers. Sink positions are zeroed.

Roles (a) and (b) can be driven by *two different* score tensors: `redistribute`
takes a selection score (`important_scores`, which picks the receivers) and an
optional separate `weight_scores` (which splits the budget among them). When
`weight_scores` is omitted the two collapse to a single source - the original
behavior - so a caller can e.g. select receivers by pre-visual cross-attention
while still weighting the split by the in-decoder cross-attention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

REDISTRIBUTION_STRATEGIES = ("topk", "random", "all")

# How the sink budget splits among the chosen receivers: "cross" weights the split by the
# (cross-attention) weight_scores, "uniform" gives every receiver an equal 1/num_receivers share.
# "uniform" is the ablation that keeps cross-attention *selecting* the receivers while removing its
# influence on *how much* each one gets.
RECEIVER_WEIGHT_MODES = ("cross", "uniform")

# accept the previous pipeline's verbose strategy names as aliases
_STRATEGY_ALIASES = {
    "topk_text_visual_tokens": "topk",
    "random_text_visual_tokens": "random",
    "all_text_visual_tokens": "all",
}


def _normalize_strategy(strategy: str) -> str:
    strategy = _STRATEGY_ALIASES.get(strategy, strategy)
    if strategy not in REDISTRIBUTION_STRATEGIES:
        raise ValueError(f"Unsupported strategy={strategy!r}; expected one of {REDISTRIBUTION_STRATEGIES}.")
    return strategy


@dataclass
class AttentionRedistributionResult:
    baseline_scores: torch.Tensor        # [N] self-attention before redistribution
    redistributed_scores: torch.Tensor   # [N] self-attention after redistribution
    receiver_indices: torch.Tensor       # [num_receivers] local visual indices that received budget
    receiver_weights: torch.Tensor       # [num_receivers] normalized split weights (sum to 1)
    sink_local_ids: torch.Tensor         # [num_sink] local visual indices zeroed
    sink_budget: float                   # total mass moved off the sinks
    selection_scores: Optional[torch.Tensor] = None  # [N] score that picked the receivers
    weight_scores: Optional[torch.Tensor] = None     # [N] score that split the budget (== selection_scores if single-source)


class sink_attention_redistributor:
    """Config-only wrapper; per-forward tensors are passed to `redistribute(...)`, so LlamaModel
    can construct this once and reuse it across generation steps."""

    def __init__(self, args):
        self.redistribution_ratio = getattr(args, "redistribution_ratio", 1.0)
        self.redistribution_strategy = _normalize_strategy(
            getattr(args, "redistribution_strategy", "topk")
        )
        self.receiver_token_count = getattr(args, "receiver_token_count", 0)
        self.receiver_score_power = getattr(args, "receiver_score_power", 1.0)
        self.receiver_weight_mode = getattr(args, "receiver_weight_mode", "cross")
        if self.receiver_weight_mode not in RECEIVER_WEIGHT_MODES:
            raise ValueError(
                f"receiver_weight_mode must be one of {RECEIVER_WEIGHT_MODES}, got {self.receiver_weight_mode!r}"
            )

        # populated by the most recent `redistribute(...)` call, for debug/visualization
        self.last_result: Optional[AttentionRedistributionResult] = None

    def _select_receivers(
        self,
        selection_scores: torch.Tensor,
        weight_scores: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pick receiver indices among the non-sink candidates, ranked by `selection_scores`, and
        their normalized split weights, taken from `weight_scores`. The two may be the same tensor
        (single-source, the original behavior) or two different cross-attention scores. Under
        receiver_weight_mode="uniform" `weight_scores` is ignored and the split is 1/num_receivers.
        Returns (indices, weights)."""
        device = selection_scores.device
        num_candidates = int(candidate_mask.sum().item())
        if num_candidates == 0:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return empty, torch.empty(0, device=device)

        num_visual = selection_scores.shape[0]
        if self.redistribution_strategy == "all":
            chosen = torch.nonzero(candidate_mask, as_tuple=False).flatten()
        else:
            count = self.receiver_token_count if self.receiver_token_count > 0 else max(
                1, round(num_visual * self.redistribution_ratio)
            )
            count = min(count, num_candidates)
            if self.redistribution_strategy == "topk":
                masked_scores = selection_scores.masked_fill(~candidate_mask, float("-inf"))
                chosen = torch.topk(masked_scores, count).indices
            else:  # random
                candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).flatten()
                perm = candidate_indices[torch.randperm(candidate_indices.numel(), device=device)]
                chosen = perm[:count]

        # `random` selection carries no ranking signal, so its split has always been uniform.
        if self.redistribution_strategy == "random" or self.receiver_weight_mode == "uniform":
            weights = torch.ones(chosen.numel(), device=device) / chosen.numel()
        else:
            weighted = weight_scores[chosen].clamp_min(0).pow(self.receiver_score_power)
            weight_sum = float(weighted.sum().item())
            weights = weighted / weight_sum if weight_sum > 0.0 else torch.ones_like(weighted) / weighted.numel()
        return chosen, weights

    def redistribute(
        self,
        image_attention: torch.Tensor,
        important_scores: torch.Tensor,
        sink_local_ids: torch.Tensor,
        sink_budget: Optional[float] = None,
        weight_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        image_attention: 1D [N] self-attention over the visual block
            (last_layer_attention_avg_last_tok_image). The actual mass being redistributed.
        important_scores: 1D [N] cross-attention importance used to *select* the receivers (and,
            unless weight_scores is given, to weight the split). Never used as attention mass.
        sink_local_ids: indices into image_attention (local to the visual block).
        sink_budget: total mass to move off the sinks; defaults to image_attention[sinks].sum().
        weight_scores: optional 1D [N] cross-attention importance used to *weight* the split among
            the chosen receivers. Defaults to important_scores (single-source). Pass a different
            tensor to select receivers by one cross-attention and split the budget by another.

        Returns [N] with sinks zeroed and their budget added onto the cross-selected receivers -
        safe to `.topk(...)` exactly like the raw attention vector it replaces.
        """
        device = image_attention.device
        important_scores = important_scores.to(device=device).float()
        weight_scores = (
            important_scores if weight_scores is None else weight_scores.to(device=device).float()
        )
        sink_local_ids = sink_local_ids.to(device=device, dtype=torch.long)

        if sink_budget is None:
            sink_budget = (
                image_attention[sink_local_ids].sum() if sink_local_ids.numel() > 0
                else image_attention.new_zeros(())
            )
        sink_budget = float(sink_budget)

        num_visual = image_attention.shape[0]
        is_sink = torch.zeros(num_visual, dtype=torch.bool, device=device)
        if sink_local_ids.numel() > 0:
            is_sink[sink_local_ids] = True
        candidate_mask = ~is_sink

        new_attention = image_attention.clone()
        new_attention[is_sink] = 0.0

        chosen, weights = self._select_receivers(important_scores, weight_scores, candidate_mask)
        if chosen.numel() > 0 and sink_budget > 0.0:
            redistributed = new_attention[chosen] + sink_budget * weights
            new_attention[chosen] = redistributed.to(dtype=new_attention.dtype)

        self.last_result = AttentionRedistributionResult(
            baseline_scores=image_attention,
            redistributed_scores=new_attention,
            receiver_indices=chosen,
            receiver_weights=weights,
            sink_local_ids=sink_local_ids,
            sink_budget=sink_budget,
            selection_scores=important_scores,
            weight_scores=weight_scores,
        )
        return new_attention


# --- back-compat for PyramidDrop's existing CLI ------------------------------------------
# The reference implementation dropped `redistribution_softmax_mode`: all three modes were
# provably ranking-neutral (softmax and sum-normalization are monotonic, so `.topk(...)`
# returns the same keep-set as the plain additive form this file now always uses). The flag
# is still declared by PyramidDrop/llava/eval/model_vqa_loader*_cross.py and passed by every
# scripts/PyramidDrop/cross_* script, so the name stays importable and is accepted-and-ignored
# rather than breaking those invocations.
REDISTRIBUTION_SOFTMAX_MODES = (
    "post_softmax",
    "resoftmax",
    "post_softmax_resoftmax",
)

# argparse `choices` for --redistribution_strategy: the reference's short names plus the
# verbose aliases the PyramidDrop scripts actually pass ("topk_text_visual_tokens" etc).
REDISTRIBUTION_STRATEGY_CHOICES = tuple(REDISTRIBUTION_STRATEGIES) + tuple(_STRATEGY_ALIASES)


if __name__ == "__main__":
    from types import SimpleNamespace

    torch.manual_seed(0)
    N = 576
    image_attention = torch.rand(N)
    sink_ids = torch.tensor([10, 20, 30])
    image_attention[sink_ids] += 5.0  # sinks dominate the self-attention, like real attention sinks
    cross_scores = torch.rand(N)  # cross-attention importance (receiver select + weight)

    r = sink_attention_redistributor(
        SimpleNamespace(redistribution_ratio=1.0, redistribution_strategy="topk_text_visual_tokens",
                        receiver_token_count=32, receiver_score_power=1.0)
    )
    budget = image_attention[sink_ids].sum()
    out = r.redistribute(image_attention, cross_scores, sink_ids, budget)

    assert out.shape == (N,)
    assert torch.allclose(out[sink_ids], torch.zeros(3), atol=1e-6), "sinks must be zeroed"
    moved = (out - image_attention)
    moved[sink_ids] = 0.0
    assert abs(float(moved.sum().item()) - float(budget.item())) < 1e-3, "budget must be conserved onto receivers"
    assert int(r.last_result.receiver_indices.numel()) == 32
    print(f"redistributed [N]={N}; budget {float(budget):.3f} moved onto {int(r.last_result.receiver_indices.numel())} receivers")

    keep = out.topk(round(N * (1 - 0.77))).indices
    assert not bool((keep.unsqueeze(1) == sink_ids.unsqueeze(0)).any().item()), "zeroed sinks should not survive top-k"
    print("self-attention-mass redistribution smoke test passed")

    # two-source variant: receivers picked by selection_scores, split weighted by weight_scores
    selection_scores = torch.rand(N)
    weight_scores = torch.rand(N)
    out2 = r.redistribute(image_attention, selection_scores, sink_ids, budget, weight_scores=weight_scores)
    chosen2 = r.last_result.receiver_indices
    # receivers must be the top-k of selection_scores (over non-sink candidates), not weight_scores
    sel_masked = selection_scores.clone()
    sel_masked[sink_ids] = float("-inf")
    assert set(chosen2.tolist()) == set(sel_masked.topk(32).indices.tolist()), "receivers must follow selection_scores"
    # split weights must be the normalized weight_scores over the chosen receivers
    expected_w = weight_scores[chosen2].clamp_min(0)
    expected_w = expected_w / expected_w.sum()
    assert torch.allclose(r.last_result.receiver_weights, expected_w, atol=1e-5), "split must follow weight_scores"
    moved2 = out2 - image_attention
    moved2[sink_ids] = 0.0
    assert abs(float(moved2.sum().item()) - float(budget.item())) < 1e-3, "budget must be conserved in two-source mode"
    print("two-source (select vs weight) redistribution smoke test passed")

    # uniform ablation: receivers still picked by the cross-attention, budget split evenly
    r_uniform = sink_attention_redistributor(
        SimpleNamespace(redistribution_ratio=1.0, redistribution_strategy="topk_text_visual_tokens",
                        receiver_token_count=32, receiver_score_power=1.0,
                        receiver_weight_mode="uniform")
    )
    out3 = r_uniform.redistribute(image_attention, selection_scores, sink_ids, budget,
                                  weight_scores=weight_scores)
    chosen3 = r_uniform.last_result.receiver_indices
    assert set(chosen3.tolist()) == set(chosen2.tolist()), "uniform mode must not change receiver selection"
    assert torch.allclose(
        r_uniform.last_result.receiver_weights, torch.full((32,), 1.0 / 32), atol=1e-6
    ), "uniform mode must split the budget evenly"
    moved3 = out3 - image_attention
    moved3[sink_ids] = 0.0
    assert abs(float(moved3.sum().item()) - float(budget.item())) < 1e-3, "budget must be conserved in uniform mode"
    print("uniform-weight ablation smoke test passed")
