"""Data schema shared by the FastV single-case capture script and the visualization
functions that read its output. Kept separate from the *_visualizations.py topic
modules so those files can stay visualization-only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, List, Optional, Union

import torch


@dataclass
class PrefillDebugSnapshot:
    # --- run identity ---
    model_id: str
    question_id: Any
    image_path: str
    question: str
    prompt: str
    generated_answer: str

    # --- fastv / sink-redistribution config actually used for this run ---
    fastv_k: int
    fastv_r: float
    image_token_start_index: int
    image_token_length: int
    prompt_length: int
    hidden_dim: int

    # --- full-sequence tensors, captured at the layer(fastv_k - 1) -> layer(fastv_k)
    # boundary during prefill (i.e. exactly what fastv_forward's pruning step sees,
    # before it prunes anything) ---
    hidden_states_at_prune_layer: torch.Tensor  # [prompt_length, hidden_dim]
    self_attention_last_token: torch.Tensor  # [prompt_length], heads-averaged

    # --- visual-token-block views (length == image_token_length each) ---
    visual_hidden_states: torch.Tensor
    visual_self_attention: torch.Tensor  # raw self-attention mass (pre-redistribution)
    visual_cross_attention_raw: torch.Tensor  # unmasked cross-attention importance
    visual_cross_attention_masked: torch.Tensor  # sink-masked cross-attention importance
    visual_redistributed_attention: torch.Tensor  # post sink-redistribution, used to rank top-k keeps

    # --- sink tokens (aligned with each other) ---
    sink_local_ids: torch.Tensor  # indices into the visual block, [0, image_token_length)
    sink_abs_ids: torch.Tensor  # sink_local_ids + image_token_start_index
    sink_scores: torch.Tensor

    # --- pruning decision made at layer fastv_k ---
    kept_visual_local_ids: torch.Tensor
    pruned_visual_local_ids: torch.Tensor

    # --- optional metadata for offline plots; absent in older snapshots ---
    text_tokens_start_index: int = 0
    text_tokens_length: int = 0
    prompt_token_ids: Optional[torch.Tensor] = None
    prompt_token_strings: List[str] = field(default_factory=list)
    generated_token_ids: Optional[torch.Tensor] = None
    generated_token_strings: List[str] = field(default_factory=list)
    generated_query_token_ids: Optional[torch.Tensor] = None
    generated_query_token_strings: List[str] = field(default_factory=list)
    generated_query_visual_attentions: Optional[torch.Tensor] = None  # [num_generated_queries, image_token_length]
    prune_layer_input_layernorm_weight: Optional[torch.Tensor] = None
    prune_layer_input_layernorm_eps: Optional[float] = None

    # --- three text->visual cross-attention variants over the visual block, each the mean over
    # question-token rows (row_probabilities.mean(dim=1)), length == image_token_length. Computed
    # at capture time by the real pipeline functions; used by cross_attention_visualize.py. ---
    visual_cross_pre_visual: Optional[torch.Tensor] = None        # pre_visual_cross_attention (V_self)
    visual_cross_text_to_visual: Optional[torch.Tensor] = None    # text_to_visual_attention (real weights)
    visual_cross_from_qk: Optional[torch.Tensor] = None           # text_to_visual_attention_from_qk (raw Q/K)

    # --- unmasked per-question-row probabilities for the same three variants. These let
    # visualization code derive unmasked maps offline with row_probabilities.mean(dim=1).
    # Sink-masked maps should prefer the logit/head-row fields below when present.
    visual_cross_pre_visual_rows: Optional[torch.Tensor] = None        # [B, T, N]
    visual_cross_text_to_visual_rows: Optional[torch.Tensor] = None    # [B, T, N]
    visual_cross_from_qk_rows: Optional[torch.Tensor] = None           # [B, T, N]

    # --- optional pre-head-average / pre-softmax tensors for sink-masked visualization.
    # These support the masked design "mask logits, then softmax" offline. For the
    # post-softmax weights path, storing per-head visual-restricted probabilities is enough:
    # zero+renormalize per head is equivalent to setting masked visual logits to -inf before
    # the visual-restricted softmax.
    visual_cross_pre_visual_logits: Optional[torch.Tensor] = None       # [B, T, N]
    visual_cross_text_to_visual_head_rows: Optional[torch.Tensor] = None  # [B, H, T, N]
    visual_cross_from_qk_logits: Optional[torch.Tensor] = None          # [B, H, T, N]

    # --- exact runtime redistribution bookkeeping from AttentionRedistributionResult.
    # Older snapshots do not have these fields; visualization code can still derive coarse
    # receiver/sink roles from visual_self_attention and visual_redistributed_attention.
    receiver_indices: Optional[torch.Tensor] = None      # [num_receivers], local visual ids
    receiver_weights: Optional[torch.Tensor] = None      # [num_receivers], normalized budget split
    sink_budget: Optional[float] = None                  # sum of baseline_scores over runtime sinks
    receiver_selection_scores: Optional[torch.Tensor] = None  # [N], scores used to select receivers
    receiver_weight_scores: Optional[torch.Tensor] = None      # [N], scores used to weight receivers


def save_snapshot(snapshot: PrefillDebugSnapshot, path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(asdict(snapshot), path)
    return path


def load_snapshot(path: Union[str, Path]) -> PrefillDebugSnapshot:
    # weights_only=False: payload is a plain dict of tensors + python scalars (str/int/float),
    # not a model checkpoint, so the weights_only allowlist restriction doesn't apply here.
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    known = {f.name for f in fields(PrefillDebugSnapshot)}
    return PrefillDebugSnapshot(**{k: v for k, v in payload.items() if k in known})
