"""Cross-attention-on-image visualization.

Overlays the three text->visual cross-attention variants on the input image, side by side:
  1. pre_visual           - pre_visual_cross_attention (V_self over pre-decoder image features)
  2. text_to_visual       - text_to_visual_attention (the real in-decoder post-softmax weights)
  3. text_to_visual_from_qk - text_to_visual_attention_from_qk (recomputed post-RoPE Q/K)

The unmasked plot uses `mean_scores = row_probabilities.mean(dim=1)` [N]. For newly captured
snapshots, the sink-masked plot masks sink columns before softmax for logit-backed variants, or
before the equivalent per-head visual renormalization for the real-weights variant. Older snapshots
that only contain precomputed [N] vectors can still render the unmasked plot, but need to be
recaptured before the logit-style sink-masked plot can be generated.

This module owns all cross-attention image visualization: it registers into the shared registry
(so `render_fastv_visualizations.py` runs it too) AND has a standalone entry point for a quick,
cross-attention-only render:

    python -m visualization.cross_attention_visualize \\
        --snapshot visualization/snapshots/pope_single_case.pt \\
        --out-dir visualization/output
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

from cross_attention_sink_redistribution.cross_attention import (
    _apply_sink_mask_and_renormalize,
    _resolve_visual_mask_columns,
)
from visualization.attention_visualizations import (
    _draw_patch_grid,
    _load_square_display_image,
    _plot_attention_overlay,
    _visual_grid_size,
)
from visualization.fastv_snapshot import PrefillDebugSnapshot, load_snapshot
from visualization.sink_token_visualizations import _offline_sink_views
from visualization.viz_registry import subdir, visualization

# Shares the attention/ output subfolder with the other text->visual plots.
SUBDIR = "attention"
DEFAULT_SINK_MASK_METHOD = "layernorm"
TOP_TOKEN_COUNTS = (8, 16, 32, 64, 128)
# The raw FastV self-attention vector (the image_attention entering redistribute) is the one
# FastV .topk(...)-selects the kept visual tokens from; show a wider range of keep budgets.
SELF_ATTENTION_TOP_TOKEN_COUNTS = (8, 16, 32, 64, 128)
SELF_ATTENTION_FOLDER = "fastv_self_attention"
# The post-sink-redistribution vector FastV .topk(...)-selects the final kept tokens from.
REDISTRIBUTED_TOP_TOKEN_COUNTS = (8, 16, 32, 64, 128)
REDISTRIBUTED_FOLDER = "redistributed_attention"

# (mean-vector attr, row-probability attr, logit/head-row attr, mask source type, panel title,
# output folder) in display order.
_VARIANTS: List[Tuple[str, str, str, str, str, str]] = [
    (
        "visual_cross_pre_visual",
        "visual_cross_pre_visual_rows",
        "visual_cross_pre_visual_logits",
        "logits",
        "pre_visual (V_self)",
        "pre_self",
    ),
    (
        "visual_cross_text_to_visual",
        "visual_cross_text_to_visual_rows",
        "visual_cross_text_to_visual_head_rows",
        "head_rows",
        "text_to_visual (weights)",
        "text_to_visual",
    ),
    (
        "visual_cross_from_qk",
        "visual_cross_from_qk_rows",
        "visual_cross_from_qk_logits",
        "logits",
        "text_to_visual_from_qk",
        "text_to_visual_from_qk",
    ),
]


def _normalize_attention_vector(
    vector: torch.Tensor,
    *,
    zero_positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Min-max normalize one panel's visual-token vector to [0, 1]."""
    vector = vector.detach().float().flatten()
    if vector.numel() == 0:
        return vector
    min_value = vector.min()
    max_value = vector.max()
    span = max_value - min_value
    if float(span.item()) <= 0.0:
        normalized = torch.zeros_like(vector)
    else:
        normalized = (vector - min_value) / span
    if zero_positions is not None and zero_positions.numel() > 0:
        normalized = normalized.clone()
        normalized[zero_positions] = 0.0
    return normalized


def _rows_to_mean_vector(rows: torch.Tensor, *, mask_positions: Optional[torch.Tensor]) -> torch.Tensor:
    row_probabilities = torch.as_tensor(rows).detach().float()
    if row_probabilities.dim() == 2:
        row_probabilities = row_probabilities.unsqueeze(0)
    if row_probabilities.dim() != 3:
        raise ValueError(f"cross-attention rows must be [B, T, N], got {tuple(row_probabilities.shape)}.")

    if mask_positions is not None and mask_positions.numel() > 0:
        row_probabilities = _apply_sink_mask_and_renormalize(row_probabilities, mask_positions)
    mean_scores = row_probabilities.mean(dim=1)  # [B, N]
    vector = mean_scores[0]
    if mask_positions is not None and mask_positions.numel() > 0:
        vector = vector.clone()
        vector[mask_positions] = 0.0
    return vector


def _logits_to_masked_mean_vector(logits: torch.Tensor, *, mask_positions: torch.Tensor) -> torch.Tensor:
    """Apply sink mask on logits, softmax, then average heads/question rows to [N]."""
    logits = torch.as_tensor(logits).detach().float()
    if logits.dim() == 2:
        logits = logits.unsqueeze(0)  # [T, N] -> [B, T, N]
    if logits.dim() not in (3, 4):
        raise ValueError(f"cross-attention logits must be [B, T, N] or [B, H, T, N], got {tuple(logits.shape)}.")

    masked_logits = logits.clone()
    masked_logits[..., mask_positions] = float("-inf")
    probabilities = torch.softmax(masked_logits, dim=-1)
    probabilities = torch.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)
    if probabilities.dim() == 4:
        vector = probabilities.mean(dim=1).mean(dim=1)[0]  # [B, H, T, N] -> [N]
    else:
        vector = probabilities.mean(dim=1)[0]  # [B, T, N] -> [N]
    vector = vector.clone()
    vector[mask_positions] = 0.0
    return vector


def _head_rows_to_masked_mean_vector(head_rows: torch.Tensor, *, mask_positions: torch.Tensor) -> torch.Tensor:
    """Equivalent to visual-logit masking when only per-head post-softmax rows are saved."""
    probabilities = torch.as_tensor(head_rows).detach().float()
    if probabilities.dim() != 4:
        raise ValueError(f"per-head cross-attention rows must be [B, H, T, N], got {tuple(probabilities.shape)}.")

    probabilities = probabilities.clone()
    probabilities[..., mask_positions] = 0.0
    row_sums = probabilities.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(probabilities.dtype).tiny)
    probabilities = probabilities / row_sums
    vector = probabilities.mean(dim=1).mean(dim=1)[0]
    vector = vector.clone()
    vector[mask_positions] = 0.0
    return vector


def _masked_vector_from_variant(
    snapshot: PrefillDebugSnapshot,
    *,
    rows_attr: str,
    mask_attr: str,
    mask_source_type: str,
    mask_positions: torch.Tensor,
) -> Optional[torch.Tensor]:
    mask_source = getattr(snapshot, mask_attr, None)
    if mask_source is not None:
        if mask_source_type == "logits":
            return _logits_to_masked_mean_vector(mask_source, mask_positions=mask_positions)
        if mask_source_type == "head_rows":
            return _head_rows_to_masked_mean_vector(mask_source, mask_positions=mask_positions)
        raise ValueError(f"Unknown mask source type: {mask_source_type}")

    rows = getattr(snapshot, rows_attr, None)
    if rows is None:
        return None
    return _rows_to_mean_vector(rows, mask_positions=mask_positions)


def _offline_sink_mask_positions(
    snapshot: PrefillDebugSnapshot,
    *,
    method: str = DEFAULT_SINK_MASK_METHOD,
) -> Tuple[Optional[torch.Tensor], str]:
    """Return sink-token ids from sink_token_visualizations.py's offline sink views."""
    views = _offline_sink_views(snapshot)
    if not views:
        return None, "none"

    method_by_name = {view.method: view for view in views}
    view = method_by_name.get(method)
    if view is None:
        view = views[0]

    mask_positions = _resolve_visual_mask_columns(
        view.sink_local_ids,
        visual_token_num=int(snapshot.image_token_length),
        device=torch.device("cpu"),
    )
    return mask_positions, view.method


def _unmasked_vector_from_variant(
    snapshot: PrefillDebugSnapshot,
    *,
    mean_attr: str,
    rows_attr: str,
) -> Optional[torch.Tensor]:
    rows = getattr(snapshot, rows_attr, None)
    if rows is not None:
        return _rows_to_mean_vector(rows, mask_positions=None)

    value = getattr(snapshot, mean_attr, None)
    if value is None:
        return None
    return torch.as_tensor(value).detach().float().flatten()


def _collect_variant_maps(
    snapshot: PrefillDebugSnapshot,
    *,
    sink_masked: bool,
    sink_mask_method: str = DEFAULT_SINK_MASK_METHOD,
) -> List[Tuple[str, torch.Tensor]]:
    """The variant [N] vectors present on this snapshot, in display order."""
    maps: List[Tuple[str, torch.Tensor]] = []

    mask_positions = None
    if sink_masked:
        mask_positions, _ = _offline_sink_mask_positions(snapshot, method=sink_mask_method)

    for mean_attr, rows_attr, mask_attr, mask_source_type, title, _folder in _VARIANTS:
        if sink_masked:
            if mask_positions is None or mask_positions.numel() == 0:
                continue
            vector = _masked_vector_from_variant(
                snapshot,
                rows_attr=rows_attr,
                mask_attr=mask_attr,
                mask_source_type=mask_source_type,
                mask_positions=mask_positions,
            )
            if vector is None:
                continue
        else:
            vector = _unmasked_vector_from_variant(snapshot, mean_attr=mean_attr, rows_attr=rows_attr)
            if vector is None:
                continue
        if vector.numel() > 0:
            zero_positions = mask_positions if sink_masked else None
            maps.append((title, _normalize_attention_vector(vector, zero_positions=zero_positions)))
    return maps


def _collect_unmasked_variant_vectors(snapshot: PrefillDebugSnapshot) -> List[Tuple[str, str, torch.Tensor]]:
    """Return raw unmasked [N] vectors for top-token ranking."""
    vectors: List[Tuple[str, str, torch.Tensor]] = []
    for mean_attr, rows_attr, _mask_attr, _mask_source_type, title, folder in _VARIANTS:
        vector = _unmasked_vector_from_variant(snapshot, mean_attr=mean_attr, rows_attr=rows_attr)
        if vector is not None and vector.numel() > 0:
            vectors.append((title, folder, vector.detach().float().flatten()))
    return vectors


def _collect_masked_variant_vectors(
    snapshot: PrefillDebugSnapshot,
    *,
    sink_mask_method: str = DEFAULT_SINK_MASK_METHOD,
) -> Tuple[List[Tuple[str, str, torch.Tensor]], str]:
    """Return raw sink-masked [N] vectors for top-token ranking, plus the resolved sink method."""
    mask_positions, resolved_method = _offline_sink_mask_positions(snapshot, method=sink_mask_method)
    vectors: List[Tuple[str, str, torch.Tensor]] = []
    if mask_positions is None or mask_positions.numel() == 0:
        return vectors, resolved_method

    for mean_attr, rows_attr, mask_attr, mask_source_type, title, folder in _VARIANTS:
        vector = _masked_vector_from_variant(
            snapshot,
            rows_attr=rows_attr,
            mask_attr=mask_attr,
            mask_source_type=mask_source_type,
            mask_positions=mask_positions,
        )
        if vector is not None and vector.numel() > 0:
            vectors.append((title, folder, vector.detach().float().flatten()))
    return vectors, resolved_method


def _plot_cross_attention_maps_on_image(
    snapshot: PrefillDebugSnapshot,
    out_dir: Path,
    *,
    maps: List[Tuple[str, torch.Tensor]],
    filename: str,
    title_prefix: str,
) -> Optional[Path]:
    grid_size = _visual_grid_size(int(snapshot.image_token_length))
    if grid_size is None:
        return None

    side_px = grid_size * 24
    image_array = _load_square_display_image(snapshot.image_path, side_px=side_px)

    fig, axes = plt.subplots(1, len(maps), figsize=(8 * len(maps), 8), dpi=220)
    if len(maps) == 1:
        axes = [axes]

    for ax, (title, vector) in zip(axes, maps):
        _plot_attention_overlay(
            ax,
            image_array=image_array,
            attention_vector=vector,
            title=title,
            grid_size=grid_size,
            vmin=0.0,
            vmax=1.0,
        )

    fig.suptitle(
        f"{title_prefix} on {grid_size}x{grid_size} patches "
        f"(question_id={snapshot.question_id})",
        fontsize=16,
        y=0.98,
    )
    heatmap = plt.cm.ScalarMappable(cmap="inferno", norm=Normalize(vmin=0.0, vmax=1.0))
    cbar = fig.colorbar(heatmap, ax=axes, fraction=0.022, pad=0.02)
    cbar.set_label("per-panel normalized cross-attention", fontsize=12)

    out_path = subdir(out_dir, SUBDIR) / filename
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return out_path


def _plot_top_cross_attention_tokens_on_image(
    snapshot: PrefillDebugSnapshot,
    out_dir: Path,
    *,
    title: str,
    folder: str,
    vector: torch.Tensor,
    top_k: int,
    name_suffix: str = "",
    title_note: str = "",
    kind_label: str = "cross-attention",
) -> Optional[Path]:
    grid_size = _visual_grid_size(int(snapshot.image_token_length))
    if grid_size is None:
        return None

    scores = vector.detach().float().flatten().clone()
    if scores.numel() == 0:
        return None

    k = min(int(top_k), int(scores.numel()))
    if k <= 0:
        return None

    scores[torch.isnan(scores)] = float("-inf")
    top_values, top_local_ids = torch.topk(scores, k=k, largest=True)

    side_px = grid_size * 28
    image_array = _load_square_display_image(snapshot.image_path, side_px=side_px)
    height, width = image_array.shape[:2]
    cell_w = width / grid_size
    cell_h = height / grid_size

    selected_min = float(top_values.min().item())
    selected_max = float(top_values.max().item())
    selected_span = max(selected_max - selected_min, 1e-12)

    rank_fontsize = 9.0 if k <= 16 else (7.5 if k <= 32 else (6.0 if k <= 64 else 4.5))
    table_fontsize = 9.5 if k <= 16 else (8.2 if k <= 32 else (6.5 if k <= 64 else 5.0))

    fig, (ax_image, ax_values) = plt.subplots(
        1,
        2,
        figsize=(13.5, 10),
        dpi=220,
        gridspec_kw={"width_ratios": [1.0, 0.36]},
    )
    ax_image.imshow(image_array)
    _draw_patch_grid(ax_image, grid_size=grid_size, width=width, height=height)

    for rank, (local_id_t, value_t) in enumerate(zip(top_local_ids, top_values), start=1):
        local_id = int(local_id_t.item())
        value = float(value_t.item())
        row, col = divmod(local_id, grid_size)
        x0 = col * cell_w
        y0 = row * cell_h
        intensity = (value - selected_min) / selected_span
        face_alpha = 0.28 + 0.42 * intensity
        edge_color = "#ffdd33" if rank == 1 else "#00d9ff"

        rect = Rectangle(
            (x0, y0),
            cell_w,
            cell_h,
            linewidth=2.0 if rank == 1 else 1.35,
            edgecolor=edge_color,
            facecolor="#00d9ff",
            alpha=face_alpha,
        )
        ax_image.add_patch(rect)
        ax_image.text(
            x0 + cell_w / 2,
            y0 + cell_h / 2,
            str(rank),
            ha="center",
            va="center",
            fontsize=rank_fontsize,
            color="black",
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.14",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.82,
            },
        )

    ax_image.set_title(f"{title}: top {k} visual tokens{title_note}", fontsize=14)
    ax_image.set_xticks([])
    ax_image.set_yticks([])
    ax_image.set_xlim(0, width)
    ax_image.set_ylim(height, 0)

    value_lines = ["rank  token  attention"]
    value_lines.extend(
        f"{rank:>4}  {int(local_id.item()):>5}  {float(value.item()):.6g}"
        for rank, (local_id, value) in enumerate(zip(top_local_ids, top_values), start=1)
    )
    ax_values.axis("off")
    ax_values.text(
        0.0,
        1.0,
        "\n".join(value_lines),
        ha="left",
        va="top",
        family="monospace",
        fontsize=table_fontsize,
    )

    fig.suptitle(
        f"Top {k} raw{title_note} {kind_label} visual tokens "
        f"(question_id={snapshot.question_id})",
        fontsize=16,
        y=0.98,
    )

    variant_dir = subdir(subdir(out_dir, SUBDIR), folder)
    out_path = variant_dir / f"top_{k}_image_tokens{name_suffix}.png"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    return out_path


@visualization
def plot_top_cross_attention_tokens_on_image(snapshot: PrefillDebugSnapshot, out_dir: Path, **_) -> List[Path]:
    """Overlay top-8/16/32 raw cross-attention tokens for each variant, both unmasked and sink-masked."""
    if _visual_grid_size(int(snapshot.image_token_length)) is None:
        return []

    paths: List[Path] = []
    for title, folder, vector in _collect_unmasked_variant_vectors(snapshot):
        for top_k in TOP_TOKEN_COUNTS:
            path = _plot_top_cross_attention_tokens_on_image(
                snapshot,
                out_dir,
                title=title,
                folder=folder,
                vector=vector,
                top_k=top_k,
            )
            if path is not None:
                paths.append(path)

    sink_mask_method = str(_.get("sink_mask_method", DEFAULT_SINK_MASK_METHOD))
    masked_vectors, resolved_sink_method = _collect_masked_variant_vectors(
        snapshot, sink_mask_method=sink_mask_method
    )
    for title, folder, vector in masked_vectors:
        for top_k in TOP_TOKEN_COUNTS:
            path = _plot_top_cross_attention_tokens_on_image(
                snapshot,
                out_dir,
                title=title,
                folder=folder,
                vector=vector,
                top_k=top_k,
                name_suffix="_sink_masked",
                title_note=f" (sink-masked, {resolved_sink_method} sinks)",
            )
            if path is not None:
                paths.append(path)
    return paths


def _fastv_self_attention_vector(snapshot: PrefillDebugSnapshot) -> Optional[torch.Tensor]:
    """Return the raw [N] self-attention vector that FastV feeds into redistribute (pre-redistribution)."""
    vector = getattr(snapshot, "visual_self_attention", None)
    if vector is None:
        return None
    vector = torch.as_tensor(vector).detach().float().flatten()
    return vector if vector.numel() > 0 else None


@visualization
def plot_fastv_self_attention_on_image(snapshot: PrefillDebugSnapshot, out_dir: Path, **_) -> List[Path]:
    """Overlay the raw FastV self-attention (the image_attention entering redistribute, pre-redistribution)
    on the image and mark its top-8/16/32/64/128 tokens - i.e. the visual tokens FastV would originally keep."""
    grid_size = _visual_grid_size(int(snapshot.image_token_length))
    if grid_size is None:
        return []

    vector = _fastv_self_attention_vector(snapshot)
    if vector is None:
        return []

    paths: List[Path] = []

    # Full attention heatmap overlay.
    side_px = grid_size * 24
    image_array = _load_square_display_image(snapshot.image_path, side_px=side_px)
    fig, ax = plt.subplots(figsize=(9, 9), dpi=220)
    _plot_attention_overlay(
        ax,
        image_array=image_array,
        attention_vector=_normalize_attention_vector(vector),
        title="FastV self-attention (pre-redistribution)",
        grid_size=grid_size,
        vmin=0.0,
        vmax=1.0,
    )
    fig.suptitle(
        f"FastV self-attention on {grid_size}x{grid_size} patches (question_id={snapshot.question_id})",
        fontsize=15,
        y=0.98,
    )
    heatmap = plt.cm.ScalarMappable(cmap="inferno", norm=Normalize(vmin=0.0, vmax=1.0))
    cbar = fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("normalized self-attention", fontsize=12)
    overlay_path = subdir(out_dir, SUBDIR) / "fastv_self_attention_on_image.png"
    fig.savefig(overlay_path, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    paths.append(overlay_path)

    # Top-k FastV-selected token positions.
    for top_k in SELF_ATTENTION_TOP_TOKEN_COUNTS:
        path = _plot_top_cross_attention_tokens_on_image(
            snapshot,
            out_dir,
            title="FastV self-attention (pre-redistribution)",
            folder=SELF_ATTENTION_FOLDER,
            vector=vector,
            top_k=top_k,
            kind_label="FastV self-attention",
        )
        if path is not None:
            paths.append(path)
    return paths


def _redistributed_attention_vector(snapshot: PrefillDebugSnapshot) -> Optional[torch.Tensor]:
    """Return the post-sink-redistribution [N] vector FastV .topk()-selects the final keeps from."""
    vector = getattr(snapshot, "visual_redistributed_attention", None)
    if vector is None:
        return None
    vector = torch.as_tensor(vector).detach().float().flatten()
    return vector if vector.numel() > 0 else None


# Token roles inferred offline by comparing the post-redistribution vector against the baseline
# self-attention (both captured verbatim from the runtime redistribute() call).
_ROLE_RECEIVER = "receiver"   # budget added: redistributed > baseline
_ROLE_NATIVE = "native"       # unchanged high self-attention: redistributed == baseline
_ROLE_SINK = "sink"           # zeroed by redistribution: redistributed == 0 < baseline
_ROLE_COLORS = {
    _ROLE_RECEIVER: "#00d9ff",  # cyan - received freed sink budget
    _ROLE_NATIVE: "#ffd23f",    # gold - already-high, kept as-is
    _ROLE_SINK: "#ff3b30",      # red  - dropped sink
}


def _classify_token_roles(
    redistributed: torch.Tensor, baseline: torch.Tensor
) -> torch.Tensor:
    """Per-token role code: 0=native, 1=receiver, 2=sink (as a long tensor aligned with the vector).

    A receiver is a token whose post-redistribution mass exceeds its baseline by more than a small
    threshold. The threshold matters: the stored baseline (visual_self_attention) is not bit-identical
    to the vector inside redistribute() - both are float16 head-means computed at slightly different
    points - so `redistributed > baseline` alone flags hundreds of tokens on ~1e-5 rounding jitter. A
    real receiver is boosted by the per-receiver budget share (~1e-3 here), orders of magnitude above
    the noise, so an adaptive floor (1% of the peak self-attention) cleanly separates the two. Sinks
    are the tokens redistribute() zeroed exactly (the runtime sink set, which may differ slightly from
    the offline-recomputed snapshot.sink_local_ids)."""
    n = int(redistributed.numel())
    roles = torch.zeros(n, dtype=torch.long)  # native
    eps = max(1e-5, 0.01 * float(baseline.max().item()) if baseline.numel() > 0 else 1e-5)
    roles[(redistributed - baseline) > eps] = 1  # receiver (meaningfully boosted)
    roles[redistributed == 0] = 2                # zeroed sink (exact)
    return roles


def _plot_redistributed_top_tokens_on_image(
    snapshot: PrefillDebugSnapshot,
    out_dir: Path,
    *,
    redistributed: torch.Tensor,
    baseline: torch.Tensor,
    roles: torch.Tensor,
    top_k: int,
) -> Optional[Path]:
    """Overlay the top-k post-redistribution tokens, colored by role (receiver vs native), with
    dropped sinks marked for reference."""
    grid_size = _visual_grid_size(int(snapshot.image_token_length))
    if grid_size is None:
        return None

    scores = redistributed.detach().float().flatten().clone()
    if scores.numel() == 0:
        return None
    k = min(int(top_k), int(scores.numel()))
    if k <= 0:
        return None

    scores[torch.isnan(scores)] = float("-inf")
    top_values, top_local_ids = torch.topk(scores, k=k, largest=True)

    side_px = grid_size * 28
    image_array = _load_square_display_image(snapshot.image_path, side_px=side_px)
    height, width = image_array.shape[:2]
    cell_w = width / grid_size
    cell_h = height / grid_size

    selected_min = float(top_values.min().item())
    selected_max = float(top_values.max().item())
    selected_span = max(selected_max - selected_min, 1e-12)

    rank_fontsize = 9.0 if k <= 16 else (7.5 if k <= 32 else (6.0 if k <= 64 else 4.5))
    table_fontsize = 9.5 if k <= 16 else (8.2 if k <= 32 else (6.5 if k <= 64 else 5.0))
    role_names = {0: _ROLE_NATIVE, 1: _ROLE_RECEIVER, 2: _ROLE_SINK}
    role_tag = {0: "N", 1: "R", 2: "S"}

    fig, (ax_image, ax_values) = plt.subplots(
        1,
        2,
        figsize=(13.5, 10),
        dpi=220,
        gridspec_kw={"width_ratios": [1.0, 0.36]},
    )
    ax_image.imshow(image_array)
    _draw_patch_grid(ax_image, grid_size=grid_size, width=width, height=height)

    # Mark all dropped sinks for reference (red dashed outline), even if not in the top-k.
    for local_id in torch.nonzero(roles == 2, as_tuple=False).flatten().tolist():
        row, col = divmod(int(local_id), grid_size)
        ax_image.add_patch(
            Rectangle(
                (col * cell_w, row * cell_h),
                cell_w,
                cell_h,
                linewidth=1.6,
                edgecolor=_ROLE_COLORS[_ROLE_SINK],
                facecolor="none",
                linestyle="--",
            )
        )

    counts = {_ROLE_RECEIVER: 0, _ROLE_NATIVE: 0}
    for rank, (local_id_t, value_t) in enumerate(zip(top_local_ids, top_values), start=1):
        local_id = int(local_id_t.item())
        value = float(value_t.item())
        role = role_names[int(roles[local_id].item())]
        if role in counts:
            counts[role] += 1
        row, col = divmod(local_id, grid_size)
        x0 = col * cell_w
        y0 = row * cell_h
        intensity = (value - selected_min) / selected_span
        face_alpha = 0.28 + 0.42 * intensity
        face_color = _ROLE_COLORS.get(role, _ROLE_COLORS[_ROLE_NATIVE])

        ax_image.add_patch(
            Rectangle(
                (x0, y0),
                cell_w,
                cell_h,
                linewidth=2.4 if rank == 1 else 1.35,
                edgecolor="#ff2d95" if rank == 1 else face_color,
                facecolor=face_color,
                alpha=face_alpha,
            )
        )
        ax_image.text(
            x0 + cell_w / 2,
            y0 + cell_h / 2,
            str(rank),
            ha="center",
            va="center",
            fontsize=rank_fontsize,
            color="black",
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
        )

    ax_image.set_title(
        f"Post-redistribution: top {k} kept tokens "
        f"({counts[_ROLE_RECEIVER]} receivers, {counts[_ROLE_NATIVE]} native)",
        fontsize=14,
    )
    ax_image.set_xticks([])
    ax_image.set_yticks([])
    ax_image.set_xlim(0, width)
    ax_image.set_ylim(height, 0)

    legend_handles = [
        Rectangle((0, 0), 1, 1, facecolor=_ROLE_COLORS[_ROLE_RECEIVER], edgecolor="none", label="receiver (got sink budget)"),
        Rectangle((0, 0), 1, 1, facecolor=_ROLE_COLORS[_ROLE_NATIVE], edgecolor="none", label="native (already high)"),
        Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=_ROLE_COLORS[_ROLE_SINK], linestyle="--", label="sink (zeroed, dropped)"),
    ]
    ax_image.legend(handles=legend_handles, loc="lower right", fontsize=8.5, framealpha=0.9)

    value_lines = ["rank  token  role  attention"]
    value_lines.extend(
        f"{rank:>4}  {int(local_id.item()):>5}  {role_tag[int(roles[int(local_id.item())].item())]:>4}  {float(value.item()):.6g}"
        for rank, (local_id, value) in enumerate(zip(top_local_ids, top_values), start=1)
    )
    ax_values.axis("off")
    ax_values.text(
        0.0,
        1.0,
        "\n".join(value_lines),
        ha="left",
        va="top",
        family="monospace",
        fontsize=table_fontsize,
    )

    fig.suptitle(
        f"Top {k} post-redistribution kept visual tokens "
        f"(question_id={snapshot.question_id})",
        fontsize=16,
        y=0.98,
    )

    variant_dir = subdir(subdir(out_dir, SUBDIR), REDISTRIBUTED_FOLDER)
    out_path = variant_dir / f"top_{k}_image_tokens.png"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    return out_path


@visualization
def plot_redistributed_attention_on_image(snapshot: PrefillDebugSnapshot, out_dir: Path, **_) -> List[Path]:
    """Overlay the post-sink-redistribution attention (the vector FastV .topk()-selects the final keeps
    from) on the image and mark its top-8/16/32/64/128 tokens, colored by role (receiver vs native, with
    dropped sinks flagged). Uses the runtime redistribution captured verbatim in the snapshot."""
    grid_size = _visual_grid_size(int(snapshot.image_token_length))
    if grid_size is None:
        return []

    redistributed = _redistributed_attention_vector(snapshot)
    if redistributed is None:
        return []
    baseline = _fastv_self_attention_vector(snapshot)
    if baseline is None or baseline.numel() != redistributed.numel():
        baseline = torch.zeros_like(redistributed)
    roles = _classify_token_roles(redistributed, baseline)

    paths: List[Path] = []

    # Full post-redistribution heatmap overlay.
    side_px = grid_size * 24
    image_array = _load_square_display_image(snapshot.image_path, side_px=side_px)
    fig, ax = plt.subplots(figsize=(9, 9), dpi=220)
    _plot_attention_overlay(
        ax,
        image_array=image_array,
        attention_vector=_normalize_attention_vector(redistributed),
        title="Post-sink-redistribution attention",
        grid_size=grid_size,
        vmin=0.0,
        vmax=1.0,
    )
    fig.suptitle(
        f"Post-redistribution attention on {grid_size}x{grid_size} patches (question_id={snapshot.question_id})",
        fontsize=15,
        y=0.98,
    )
    heatmap = plt.cm.ScalarMappable(cmap="inferno", norm=Normalize(vmin=0.0, vmax=1.0))
    cbar = fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("normalized redistributed attention", fontsize=12)
    overlay_path = subdir(out_dir, SUBDIR) / "redistributed_attention_on_image.png"
    fig.savefig(overlay_path, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    paths.append(overlay_path)

    # Top-k final kept token positions, colored by role.
    for top_k in REDISTRIBUTED_TOP_TOKEN_COUNTS:
        path = _plot_redistributed_top_tokens_on_image(
            snapshot,
            out_dir,
            redistributed=redistributed,
            baseline=baseline,
            roles=roles,
            top_k=top_k,
        )
        if path is not None:
            paths.append(path)
    return paths


@visualization
def plot_cross_attention_variants_on_image(snapshot: PrefillDebugSnapshot, out_dir: Path, **_) -> List[Path]:
    if _visual_grid_size(int(snapshot.image_token_length)) is None:
        return []

    outputs: List[Path] = []

    unmasked_maps = _collect_variant_maps(snapshot, sink_masked=False)
    if unmasked_maps:
        path = _plot_cross_attention_maps_on_image(
            snapshot,
            out_dir,
            maps=unmasked_maps,
            filename="cross_attention_variants_on_image.png",
            title_prefix="Per-panel normalized text->visual cross-attention variants",
        )
        if path is not None:
            outputs.append(path)

    sink_mask_method = str(_.get("sink_mask_method", DEFAULT_SINK_MASK_METHOD))
    masked_maps = _collect_variant_maps(snapshot, sink_masked=True, sink_mask_method=sink_mask_method)
    if masked_maps:
        _, resolved_sink_method = _offline_sink_mask_positions(snapshot, method=sink_mask_method)
        path = _plot_cross_attention_maps_on_image(
            snapshot,
            out_dir,
            maps=masked_maps,
            filename="cross_attention_variants_sink_masked_on_image.png",
            title_prefix=(
                "Per-panel normalized sink-masked text->visual cross-attention variants "
                f"({resolved_sink_method} offline sinks)"
            ),
        )
        if path is not None:
            outputs.append(path)

    return outputs


def _safe_dirname(value) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return text.strip("_") or "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snapshot", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="visualization/output")
    parser.add_argument(
        "--no-question-subdir",
        dest="question_subdir",
        action="store_false",
        help="Write straight into --out-dir instead of an out-dir/<question_id>/ subfolder.",
    )
    parser.add_argument(
        "--sink-mask-method",
        type=str,
        default=DEFAULT_SINK_MASK_METHOD,
        choices=("max", "sum", "layernorm"),
        help="Offline sink-token method from sink_token_visualizations.py used for the sink-masked cross-attention plot.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    snapshot = load_snapshot(args.snapshot)
    out_dir = Path(args.out_dir)
    if args.question_subdir:
        out_dir = out_dir / _safe_dirname(snapshot.question_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    paths.extend(plot_cross_attention_variants_on_image(snapshot, out_dir, sink_mask_method=args.sink_mask_method))
    paths.extend(plot_top_cross_attention_tokens_on_image(snapshot, out_dir, sink_mask_method=args.sink_mask_method))
    paths.extend(plot_fastv_self_attention_on_image(snapshot, out_dir))
    paths.extend(plot_redistributed_attention_on_image(snapshot, out_dir))
    if not paths:
        print(
            "No cross-attention variants found on the snapshot. Re-run capture_fastv_single_case.py "
            "with the updated capture script to populate them."
        )
        return
    for path in paths:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
