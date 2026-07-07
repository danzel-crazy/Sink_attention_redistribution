import torch
from typing import Sequence, Tuple

class sink_token_selector:
    def __init__(self, args):
        # sink score settings
        self.sink_dims = getattr(args, "sink_dims", [2533])
        self.sink_min = getattr(args, "sink_score_min", None)
        self.sink_max = getattr(args, "sink_score_max", None)
        self.sink_quantile = getattr(args, "sink_score_quantile", 0.99)
        self.sink_tokens_scores = []

        # selected sink tokens ids
        self.sink_tokens_ids = []
        
        # sink budget
        self.sink_budget = 0.0

    @staticmethod
    def _compute_hidden_rms_max_sink_values(
        hidden_states: torch.Tensor,
        valid_dims: Sequence[int],
    ) -> torch.Tensor:
        hidden_dim = hidden_states.shape[1]

        # RMS denominator for each token: shape [num_tokens]
        mean_square_per_token = torch.sum(hidden_states ** 2, dim=1) / hidden_dim
        denom = torch.sqrt(mean_square_per_token + 1e-6)

        # selected sink-dim values: shape [num_tokens, num_valid_dims]
        sink_candidates = torch.abs(hidden_states[:, valid_dims] / denom.unsqueeze(1))

        # max over target dimensions for each token: shape [num_tokens]
        return torch.max(sink_candidates, dim=1).values

    def resolve_sink_score_range(
        self,
        sink_values: torch.Tensor,
    ) -> Tuple[float, float, str]:
        sink_values = sink_values.detach().float().flatten()
        if sink_values.numel() == 0:
            raise ValueError("Cannot resolve sink-score range from an empty tensor.")

        score_min = self.sink_min
        score_max = self.sink_max

        if score_min is None:
            quantile = min(max(float(self.sink_quantile), 0.0), 1.0)
            score_min = float(torch.quantile(sink_values, quantile).item())
            range_source = f"quantile_{quantile:.2f}_to_max"
        else:
            score_min = float(score_min)
            range_source = "manual_range"

        if score_max is None:
            score_max = float(torch.max(sink_values).item())
        else:
            score_max = float(score_max)

        if score_min > score_max:
            raise ValueError(
                f"Invalid sink-score range: min={score_min} is greater than max={score_max}."
            )

        return score_min, score_max, range_source

    def _select_sink_tokens(
        self,
        hidden_states: torch.Tensor,
    ):
        # Compute sink values for the hidden states
        sink_values = self._compute_hidden_rms_max_sink_values(hidden_states, self.sink_dims)

        # Resolve the sink score range
        score_min, score_max, _ = self.resolve_sink_score_range(sink_values)

        # Select tokens based on the resolved range
        selected_tokens_mask = (sink_values >= score_min) & (sink_values <= score_max)
        selected_tokens_ids = torch.nonzero(selected_tokens_mask).squeeze(1)

        self.sink_tokens_ids = selected_tokens_ids
        self.sink_tokens_scores = sink_values[selected_tokens_ids]
        
        return selected_tokens_ids

        