from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

import torch

REDISTRIBUTION_STRATEGIES = (
    "all_text_visual_tokens",
    "topk_text_visual_tokens",
    "random_text_visual_tokens",
)

REDISTRIBUTION_SOFTMAX_MODES = (
    "post_softmax",
    "resoftmax",
    "post_softmax_resoftmax",
)


@dataclass
class AttentionRedistributionResult:
    baseline_scores: torch.Tensor
    redistributed_scores: torch.Tensor
    sink_local_positions: List[int]
    sink_abs_positions: List[int]
    receiver_local_positions: List[int]
    receiver_abs_positions: List[int]
    receiver_weights: List[float]


class sink_attention_redistributor:
    def __init__(self, args):
        # This class holds only static config. sink ids / attention scores / budget are only
        # known at inference time (per forward call) and are passed as arguments to
        # `redistribute(...)` instead of stored here - that's what lets LlamaModel construct
        # this once in __init__ and reuse it across every generation step.

        # redistribution settings: all, topk, random
        self.redistribution_ratio = getattr(args, "redistribution_ratio", 1.0)
        self.redistribution_strategy = getattr(
            args, "redistribution_strategy", "topk_text_visual_tokens"
        )
        if self.redistribution_strategy not in REDISTRIBUTION_STRATEGIES:
            raise ValueError(f"Unknown redistribution_strategy: {self.redistribution_strategy}")

        # redistribution softmax mode: post_softmax, resoftmax, post_softmax_resoftmax
        self.redistribution_softmax_mode = getattr(
            args, "redistribution_softmax_mode", "post_softmax"
        )
        if self.redistribution_softmax_mode not in REDISTRIBUTION_SOFTMAX_MODES:
            raise ValueError(f"Unknown redistribution_softmax_mode: {self.redistribution_softmax_mode}")

        # receiver token selection settings
        self.receiver_token_count = getattr(args, "receiver_token_count", 0)
        self.receiver_score_power = getattr(args, "receiver_score_power", 1.0)

        # populated by the most recent `redistribute(...)` call, for debug/visualization
        self.last_result: Optional[AttentionRedistributionResult] = None

    @staticmethod
    def _resolve_sink_local_positions(
        *,
        visual_token_start: int,
        visual_token_length: int,
        sink_token_abs_positions: List[int],
    ) -> List[int]:
        visual_start = int(visual_token_start)
        visual_end = visual_start + int(visual_token_length)
        return sorted(
            {
                int(position) - visual_start
                for position in sink_token_abs_positions
                if visual_start <= int(position) < visual_end
            }
        )

    @staticmethod
    def _resolve_receiver_local_positions(
        *,
        visual_token_length: int,
        sink_local_positions: List[int],
    ) -> List[int]:
        visual_length = int(visual_token_length)
        sink_local_set = set(sink_local_positions)
        return [
            position
            for position in range(visual_length)
            if position not in sink_local_set
        ]

    def _select_receiver_indices(self, receiver_importance_scores: torch.Tensor) -> torch.Tensor:
        """Indices *into* receiver_importance_scores (and receiver_original_scores) that will
        actually receive redistributed mass. Ranked by importance (e.g. cross-attention), not
        by the self-attention mass being redistributed."""
        num_receivers = receiver_importance_scores.shape[0]
        device = receiver_importance_scores.device
        if num_receivers == 0:
            return torch.empty(0, dtype=torch.long, device=device)

        if self.redistribution_strategy == "all_text_visual_tokens":
            return torch.arange(num_receivers, device=device)

        count = self.receiver_token_count if self.receiver_token_count > 0 else max(
            1, round(num_receivers * self.redistribution_ratio)
        )
        count = min(count, num_receivers)

        if self.redistribution_strategy == "topk_text_visual_tokens":
            return torch.topk(receiver_importance_scores, count).indices
        elif self.redistribution_strategy == "random_text_visual_tokens":
            perm = torch.randperm(num_receivers, device=device)
            return perm[:count]
        raise ValueError(f"Unknown redistribution_strategy: {self.redistribution_strategy}")

    def _receiver_weights(
        self, receiver_importance_scores: torch.Tensor, chosen_indices: torch.Tensor
    ) -> torch.Tensor:
        """Normalized weights (sum to 1) describing how sink_budget is split across chosen_indices,
        proportional to importance score (e.g. cross-attention), not to self-attention mass."""
        if chosen_indices.numel() == 0:
            return torch.empty(0, device=receiver_importance_scores.device)

        if self.redistribution_strategy == "random_text_visual_tokens":
            weights = torch.ones(chosen_indices.numel(), device=receiver_importance_scores.device)
        else:
            weights = receiver_importance_scores[chosen_indices].clamp_min(1e-12) ** self.receiver_score_power

        return weights / weights.sum()

    def _redistribute_to_receivers(
        self,
        sink_budget: float,
        receiver_local_positions: List[int],
        receiver_original_scores: torch.Tensor,
        receiver_importance_scores: torch.Tensor,
    ) -> AttentionRedistributionResult:
        """
        Move `sink_budget` probability mass off sink tokens onto the receivers named by
        `receiver_local_positions`. `receiver_original_scores` (self-attention - the mass being
        redistributed) and `receiver_importance_scores` (e.g. cross-attention - decides who
        qualifies as a receiver and how much of the budget each gets) must both already be
        aligned 1:1 with `receiver_local_positions`; `redistributed_scores` preserves that
        alignment (same length, same order) in every strategy/mode combination.
        """
        if receiver_original_scores.shape[0] != len(receiver_local_positions):
            raise ValueError(
                "receiver_original_scores and receiver_local_positions must have the same length: "
                f"{receiver_original_scores.shape[0]} != {len(receiver_local_positions)}"
            )
        if receiver_importance_scores.shape[0] != len(receiver_local_positions):
            raise ValueError(
                "receiver_importance_scores and receiver_local_positions must have the same length: "
                f"{receiver_importance_scores.shape[0]} != {len(receiver_local_positions)}"
            )

        baseline_scores = receiver_original_scores.clone()
        chosen_indices = self._select_receiver_indices(receiver_importance_scores)
        weights = self._receiver_weights(receiver_importance_scores, chosen_indices)

        if self.redistribution_softmax_mode == "post_softmax":
            # additive in probability space; total mass grows by exactly sink_budget
            redistributed_scores = baseline_scores.clone()
            redistributed_scores[chosen_indices] = redistributed_scores[chosen_indices] + sink_budget * weights

        elif self.redistribution_softmax_mode == "resoftmax":
            # additive in probability space like post_softmax, then softmax the whole updated
            # vector (not just renormalize it) to get the final redistributed scores
            redistributed_scores = baseline_scores.clone()
            redistributed_scores[chosen_indices] = redistributed_scores[chosen_indices] + sink_budget * weights
            redistributed_scores = torch.softmax(redistributed_scores, dim=-1)

        elif self.redistribution_softmax_mode == "post_softmax_resoftmax":
            # additive in probability space like post_softmax, but force-renormalized to sum to 1
            redistributed_scores = baseline_scores.clone()
            redistributed_scores[chosen_indices] = redistributed_scores[chosen_indices] + sink_budget * weights
            redistributed_scores = redistributed_scores / redistributed_scores.sum().clamp_min(1e-12)

        else:
            raise ValueError(f"Unknown redistribution_softmax_mode: {self.redistribution_softmax_mode}")

        receiver_weights_full = torch.zeros_like(baseline_scores)
        receiver_weights_full[chosen_indices] = weights

        return AttentionRedistributionResult(
            baseline_scores=baseline_scores,
            redistributed_scores=redistributed_scores,
            sink_local_positions=[],
            sink_abs_positions=[],
            receiver_local_positions=list(receiver_local_positions),
            receiver_abs_positions=[],
            receiver_weights=receiver_weights_full.tolist(),
        )

    def redistribute(
        self,
        image_attention: torch.Tensor,
        important_scores: torch.Tensor,
        sink_local_ids: torch.Tensor,
        sink_budget: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Public entry point called once per forward pass, at inference time.

        image_attention: 1D tensor, per-token self-attention score for the visual-token block
            (e.g. `last_layer_attention_avg_last_tok_image`), length == visual_token_length.
            This is the actual mass being redistributed - sink_budget and the returned tensor's
            values both come from this, never from `important_scores`.
        important_scores: 1D tensor, same length as `image_attention` (e.g. cross-attention
            importance from `cross_attention_importants.compute_cross_attention`). Used only to
            decide which positions qualify as receivers and how much of the budget each gets -
            never used as the mass itself.
        sink_local_ids: indices *into* image_attention (local to the visual-token block,
            i.e. already offset-adjusted - NOT absolute sequence positions like the raw
            output of `sink_token_selector._select_sink_tokens`).
        sink_budget: total probability mass to move off the sinks. If None, it is computed
            as image_attention[sink_local_ids].sum().

        Returns a tensor the same shape as `image_attention`, with sink positions zeroed
        and their freed budget redistributed onto receivers - safe to `.topk(...)` directly,
        same as the raw attention vector it replaces.
        """
        device = image_attention.device
        sink_local_ids = sink_local_ids.to(device=device, dtype=torch.long)
        important_scores = important_scores.to(device=device)

        if sink_budget is None:
            sink_budget = (
                image_attention[sink_local_ids].sum() if sink_local_ids.numel() > 0 else image_attention.new_zeros(())
            )

        receiver_local_positions = self._resolve_receiver_local_positions(
            visual_token_length=image_attention.shape[0],
            sink_local_positions=sink_local_ids.tolist(),
        )
        receiver_local_positions_t = torch.as_tensor(receiver_local_positions, device=device, dtype=torch.long)
        receiver_original_scores = image_attention[receiver_local_positions_t]
        receiver_importance_scores = important_scores[receiver_local_positions_t]

        result = self._redistribute_to_receivers(
            sink_budget=float(sink_budget),
            receiver_local_positions=receiver_local_positions,
            receiver_original_scores=receiver_original_scores,
            receiver_importance_scores=receiver_importance_scores,
        )
        result.sink_local_positions = sink_local_ids.tolist()
        self.last_result = result

        new_image_attention = image_attention.clone()
        new_image_attention[sink_local_ids] = 0.0
        new_image_attention[receiver_local_positions_t] = result.redistributed_scores

        return new_image_attention
