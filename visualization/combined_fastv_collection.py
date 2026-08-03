"""Focused FastV attention-redistribution visualization collection.

This renderer builds the compact set of plots useful for comparing raw FastV
self-attention against post sink-redistribution attention:

  1. baseline_scores top-8/16/32/64/128 image-token overlays
  2. redistributed_scores top-8/16/32/64/128 image-token overlays
  3. receiver_weights image overlay with receiver_indices
  4. sink_budget percentage of baseline_scores
  5. runtime sink-token positions
  6. hidden states of the top-5 redistributed_scores visual tokens

Usage:
    python -m visualization.combined_fastv_collection \\
        --snapshot visualization/snapshots/pope/1.pt \\
        --out-dir visualization/output

Multiple snapshots or a snapshot directory are accepted:
    python -m visualization.combined_fastv_collection \\
        --snapshot visualization/snapshots/pope \\
        --out-dir visualization/output
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - registers the 3d projection
from PIL import Image

from visualization.attention_visualizations import (
    _draw_patch_grid,
    _load_square_display_image,
    _visual_grid_size,
)
from visualization.fastv_snapshot import PrefillDebugSnapshot, load_snapshot

TOP_TOKEN_COUNTS = (8, 16, 32, 64, 128)
SUBDIR = "combined_collection"
# Hidden states of the highest-ranked post-redistribution visual tokens, i.e. the ones FastV
# actually keeps. Written into their own folder inside the collection.
HIDDEN_STATE_TOP_K = 5
HIDDEN_STATE_FOLDER = "redistributed_top_hidden_states"

ROLE_RECEIVER = "receiver"
ROLE_NATIVE = "native"
ROLE_SINK = "sink"
ROLE_COLORS = {
    ROLE_RECEIVER: "#00bcd4",
    ROLE_NATIVE: "#ffd23f",
    ROLE_SINK: "#ff3b30",
}
POSITION_HIGHLIGHT_COLOR = "#a020f0"
POSITION_HIGHLIGHT_ALPHA = 0.60

# keep/drop overlay (the pruning decision recorded in kept_visual_local_ids)
KEPT_COLOR = "#2ecc71"
KEPT_ALPHA = 0.42
DROPPED_COLOR = "#101010"
DROPPED_ALPHA = 0.62


def _safe_dirname(value) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return text.strip("_") or "unknown"


def _expand_snapshot_inputs(inputs: Iterable[str], recursive: bool) -> List[Path]:
    paths: List[Path] = []
    for value in inputs:
        path = Path(value)
        if path.is_dir():
            pattern = "**/*.pt" if recursive else "*.pt"
            paths.extend(sorted(path.glob(pattern)))
        else:
            paths.append(path)
    return paths


def _collection_dir(root: Path, snapshot: PrefillDebugSnapshot, question_subdir: bool) -> Path:
    out_dir = root / _safe_dirname(snapshot.question_id) if question_subdir else root
    out_dir = out_dir / SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _vector(value) -> Optional[torch.Tensor]:
    if value is None:
        return None
    vector = torch.as_tensor(value).detach().float().flatten()
    return vector if vector.numel() > 0 else None


def _valid_ids(ids, n: int) -> torch.Tensor:
    if ids is None:
        return torch.empty(0, dtype=torch.long)
    ids_t = torch.as_tensor(ids).detach().long().flatten()
    if ids_t.numel() == 0:
        return ids_t
    return ids_t[(ids_t >= 0) & (ids_t < int(n))]


def _baseline_scores(snapshot: PrefillDebugSnapshot) -> Optional[torch.Tensor]:
    return _vector(getattr(snapshot, "visual_self_attention", None))


def _redistributed_scores(snapshot: PrefillDebugSnapshot) -> Optional[torch.Tensor]:
    return _vector(getattr(snapshot, "visual_redistributed_attention", None))


def _save_original_image(snapshot: PrefillDebugSnapshot, out_path: Path) -> Optional[Path]:
    try:
        image = Image.open(snapshot.image_path).convert("RGB")
    except FileNotFoundError:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    return out_path


def _save_square_image(snapshot: PrefillDebugSnapshot, out_path: Path) -> Optional[Path]:
    """Save the center-cropped square canvas the overlay plots draw on."""
    grid_size = _visual_grid_size(int(snapshot.image_token_length))
    if grid_size is None or not Path(snapshot.image_path).exists():
        return None
    image_array = _load_square_display_image(snapshot.image_path, side_px=grid_size * 28)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_array).save(out_path)
    return out_path


def _runtime_sink_ids(snapshot: PrefillDebugSnapshot, n: int) -> torch.Tensor:
    return _valid_ids(getattr(snapshot, "sink_local_ids", None), n)


def _classify_redistributed_roles(
    baseline: torch.Tensor,
    redistributed: torch.Tensor,
    *,
    exact_receivers: Optional[torch.Tensor],
    sink_ids: torch.Tensor,
) -> Dict[int, str]:
    roles: Dict[int, str] = {int(local_id.item()): ROLE_SINK for local_id in sink_ids}
    if exact_receivers is not None and exact_receivers.numel() > 0:
        for local_id in exact_receivers:
            roles[int(local_id.item())] = ROLE_RECEIVER
        return roles

    eps = max(1e-5, 0.01 * float(baseline.max().item()) if baseline.numel() > 0 else 1e-5)
    inferred = torch.nonzero((redistributed - baseline) > eps, as_tuple=False).flatten()
    for local_id in inferred:
        roles[int(local_id.item())] = ROLE_RECEIVER
    return roles


def _rank_fontsize(k: int) -> float:
    return 9.0 if k <= 16 else (7.5 if k <= 32 else (6.0 if k <= 64 else 4.5))


def _table_fontsize(k: int) -> float:
    return 9.5 if k <= 16 else (8.2 if k <= 32 else (6.5 if k <= 64 else 5.0))


def _top_token_values_and_ids(vector: torch.Tensor, top_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    scores = vector.detach().float().flatten().clone()
    k = min(int(top_k), int(scores.numel()))
    if k <= 0:
        return torch.empty(0, dtype=torch.float32), torch.empty(0, dtype=torch.long)
    scores[torch.isnan(scores)] = float("-inf")
    return torch.topk(scores, k=k, largest=True)


def _plot_top_tokens_on_image(
    snapshot: PrefillDebugSnapshot,
    out_path: Path,
    *,
    vector: torch.Tensor,
    top_k: int,
    title: str,
    table_score_label: str,
    role_by_token: Optional[Dict[int, str]] = None,
    sink_ids: Optional[torch.Tensor] = None,
) -> Optional[Path]:
    grid_size = _visual_grid_size(int(snapshot.image_token_length))
    if grid_size is None:
        return None

    top_values, top_local_ids = _top_token_values_and_ids(vector, top_k)
    k = int(top_local_ids.numel())
    if k <= 0:
        return None

    side_px = grid_size * 28
    image_array = _load_square_display_image(snapshot.image_path, side_px=side_px)
    height, width = image_array.shape[:2]
    cell_w = width / grid_size
    cell_h = height / grid_size

    selected_min = float(top_values.min().item())
    selected_max = float(top_values.max().item())
    selected_span = max(selected_max - selected_min, 1e-12)

    fig, (ax_image, ax_values) = plt.subplots(
        1,
        2,
        figsize=(13.5, 10),
        dpi=220,
        gridspec_kw={"width_ratios": [1.0, 0.38]},
    )
    ax_image.imshow(image_array)
    _draw_patch_grid(ax_image, grid_size=grid_size, width=width, height=height)

    if sink_ids is not None:
        for local_id in sink_ids.tolist():
            row, col = divmod(int(local_id), grid_size)
            ax_image.add_patch(
                Rectangle(
                    (col * cell_w, row * cell_h),
                    cell_w,
                    cell_h,
                    linewidth=1.45,
                    edgecolor=ROLE_COLORS[ROLE_SINK],
                    facecolor="none",
                    linestyle="--",
                )
            )

    role_counts = {ROLE_RECEIVER: 0, ROLE_NATIVE: 0, ROLE_SINK: 0}
    value_lines = ["rank  token  role  " + table_score_label]
    for rank, (local_id_t, value_t) in enumerate(zip(top_local_ids, top_values), start=1):
        local_id = int(local_id_t.item())
        value = float(value_t.item())
        role = (role_by_token or {}).get(local_id, ROLE_NATIVE if role_by_token else "")
        if role:
            role_counts[role] = role_counts.get(role, 0) + 1

        row, col = divmod(local_id, grid_size)
        x0 = col * cell_w
        y0 = row * cell_h
        intensity = (value - selected_min) / selected_span
        face_alpha = 0.25 + 0.45 * intensity
        face_color = ROLE_COLORS.get(role, "#00d9ff")
        edge_color = "#ff2d95" if rank == 1 else face_color

        ax_image.add_patch(
            Rectangle(
                (x0, y0),
                cell_w,
                cell_h,
                linewidth=2.3 if rank == 1 else 1.35,
                edgecolor=edge_color,
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
            fontsize=_rank_fontsize(k),
            color="black",
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
        )
        role_cell = role[0].upper() if role else "-"
        value_lines.append(f"{rank:>4}  {local_id:>5}  {role_cell:>4}  {value:.6g}")

    if role_by_token:
        ax_image.set_title(
            f"{title}: top {k} ({role_counts[ROLE_RECEIVER]} receivers, "
            f"{role_counts[ROLE_NATIVE]} native)",
            fontsize=14,
        )
        legend_handles = [
            Rectangle((0, 0), 1, 1, facecolor=ROLE_COLORS[ROLE_RECEIVER], edgecolor="none", label="receiver"),
            Rectangle((0, 0), 1, 1, facecolor=ROLE_COLORS[ROLE_NATIVE], edgecolor="none", label="native"),
            Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=ROLE_COLORS[ROLE_SINK], linestyle="--", label="sink"),
        ]
        ax_image.legend(handles=legend_handles, loc="lower right", fontsize=8.5, framealpha=0.9)
    else:
        ax_image.set_title(f"{title}: top {k}", fontsize=14)

    ax_image.set_xticks([])
    ax_image.set_yticks([])
    ax_image.set_xlim(0, width)
    ax_image.set_ylim(height, 0)

    ax_values.axis("off")
    ax_values.text(
        0.0,
        1.0,
        "\n".join(value_lines),
        ha="left",
        va="top",
        family="monospace",
        fontsize=_table_fontsize(k),
    )

    fig.suptitle(f"{title} (question_id={snapshot.question_id})", fontsize=16, y=0.98)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    return out_path


def _plot_top_token_positions_on_image(
    snapshot: PrefillDebugSnapshot,
    out_path: Path,
    *,
    vector: torch.Tensor,
    top_k: int,
    edge_color: str,
    face_color: str,
) -> Optional[Path]:
    grid_size = _visual_grid_size(int(snapshot.image_token_length))
    if grid_size is None:
        return None

    top_values, top_local_ids = _top_token_values_and_ids(vector, top_k)
    k = int(top_local_ids.numel())
    if k <= 0:
        return None

    side_px = grid_size * 28
    image_array = _load_square_display_image(snapshot.image_path, side_px=side_px)
    height, width = image_array.shape[:2]
    cell_w = width / grid_size
    cell_h = height / grid_size

    fig, ax = plt.subplots(figsize=(8.8, 8.8), dpi=220)
    ax.imshow(image_array)

    for local_id_t in top_local_ids:
        local_id = int(local_id_t.item())
        row, col = divmod(local_id, grid_size)
        ax.add_patch(
            Rectangle(
                (col * cell_w, row * cell_h),
                cell_w,
                cell_h,
                linewidth=1.8,
                edgecolor=edge_color,
                facecolor=face_color,
                alpha=POSITION_HIGHLIGHT_ALPHA,
            )
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return out_path


def _plot_kept_tokens_on_image(
    snapshot: PrefillDebugSnapshot,
    out_path: Path,
    *,
    kept_ids: torch.Tensor,
    n: int,
) -> Optional[Path]:
    """Paint the pruning decision itself: which visual tokens survived the prune layer.

    Reads the recorded `kept_visual_local_ids` rather than re-deriving a top-k, so it stays
    correct whatever rounding the pipeline used for the keep count.
    """
    grid_size = _visual_grid_size(int(snapshot.image_token_length))
    if grid_size is None or kept_ids.numel() == 0:
        return None

    side_px = grid_size * 28
    image_array = _load_square_display_image(snapshot.image_path, side_px=side_px)
    height, width = image_array.shape[:2]
    cell_w = width / grid_size
    cell_h = height / grid_size

    keep_mask = torch.zeros(int(n), dtype=torch.bool)
    keep_mask[kept_ids] = True

    fig, ax = plt.subplots(figsize=(8.8, 8.8), dpi=220)
    ax.imshow(image_array)

    for local_id in range(int(n)):
        row, col = divmod(local_id, grid_size)
        kept = bool(keep_mask[local_id].item())
        ax.add_patch(
            Rectangle(
                (col * cell_w, row * cell_h),
                cell_w,
                cell_h,
                linewidth=0.0,
                edgecolor="none",
                facecolor=KEPT_COLOR if kept else DROPPED_COLOR,
                alpha=KEPT_ALPHA if kept else DROPPED_ALPHA,
            )
        )

    kept_count = int(keep_mask.sum().item())
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.legend(
        handles=[
            Patch(facecolor=KEPT_COLOR, alpha=KEPT_ALPHA, label=f"kept ({kept_count})"),
            Patch(facecolor=DROPPED_COLOR, alpha=DROPPED_ALPHA, label=f"dropped ({int(n) - kept_count})"),
        ],
        loc="upper right",
        fontsize=11,
        framealpha=0.9,
    )
    ax.set_title(
        f"visual tokens kept at the prune layer (question_id={snapshot.question_id})",
        fontsize=14,
        pad=12,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    return out_path


def _receiver_data(
    snapshot: PrefillDebugSnapshot,
    baseline: torch.Tensor,
    redistributed: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    exact_indices = _valid_ids(getattr(snapshot, "receiver_indices", None), int(baseline.numel()))
    exact_weights = _vector(getattr(snapshot, "receiver_weights", None))
    if exact_indices.numel() > 0 and exact_weights is not None and exact_weights.numel() == exact_indices.numel():
        return exact_indices, exact_weights, True

    moved = torch.clamp(redistributed - baseline, min=0.0)
    eps = max(1e-5, 0.01 * float(baseline.max().item()) if baseline.numel() > 0 else 1e-5)
    indices = torch.nonzero(moved > eps, as_tuple=False).flatten()
    if indices.numel() == 0:
        return indices.long(), torch.empty(0, dtype=torch.float32), False
    weights = moved[indices].float()
    total = weights.sum()
    if float(total.item()) > 0.0:
        weights = weights / total
    return indices.long(), weights, False


def _plot_receiver_weights_on_image(
    snapshot: PrefillDebugSnapshot,
    out_path: Path,
    *,
    receiver_indices: torch.Tensor,
    receiver_weights: torch.Tensor,
    exact: bool,
) -> Optional[Path]:
    grid_size = _visual_grid_size(int(snapshot.image_token_length))
    if grid_size is None or receiver_indices.numel() == 0:
        return None

    weights = receiver_weights.detach().float().flatten()
    indices = receiver_indices.detach().long().flatten()
    order = torch.argsort(weights, descending=True)
    weights = weights[order]
    indices = indices[order]

    side_px = grid_size * 28
    image_array = _load_square_display_image(snapshot.image_path, side_px=side_px)
    height, width = image_array.shape[:2]
    cell_w = width / grid_size
    cell_h = height / grid_size

    max_weight = max(float(weights.max().item()), 1e-12)
    fig, ax_image = plt.subplots(figsize=(10, 10), dpi=220)
    ax_image.imshow(image_array)
    _draw_patch_grid(ax_image, grid_size=grid_size, width=width, height=height)

    for rank, (local_id_t, weight_t) in enumerate(zip(indices, weights), start=1):
        local_id = int(local_id_t.item())
        weight = float(weight_t.item())
        row, col = divmod(local_id, grid_size)
        x0 = col * cell_w
        y0 = row * cell_h
        alpha = 0.25 + 0.55 * (weight / max_weight)
        ax_image.add_patch(
            Rectangle(
                (x0, y0),
                cell_w,
                cell_h,
                linewidth=1.35,
                edgecolor="#006c7a",
                facecolor=ROLE_COLORS[ROLE_RECEIVER],
                alpha=alpha,
            )
        )
        ax_image.text(
            x0 + cell_w / 2,
            y0 + cell_h / 2,
            str(rank),
            ha="center",
            va="center",
            fontsize=_rank_fontsize(int(indices.numel())),
            color="black",
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
        )

    mode = "exact" if exact else "inferred"
    ax_image.set_title(f"Receiver weights ({mode}, {int(indices.numel())} tokens)", fontsize=14)
    ax_image.set_xticks([])
    ax_image.set_yticks([])
    ax_image.set_xlim(0, width)
    ax_image.set_ylim(height, 0)

    fig.suptitle(f"Receiver budget split (question_id={snapshot.question_id})", fontsize=16, y=0.98)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    return out_path


def _plot_runtime_sink_positions(
    snapshot: PrefillDebugSnapshot,
    out_path: Path,
    *,
    baseline: torch.Tensor,
    sink_ids: torch.Tensor,
) -> Optional[Path]:
    grid_size = _visual_grid_size(int(snapshot.image_token_length))
    if grid_size is None or sink_ids.numel() == 0:
        return None

    sink_budget = float(baseline[sink_ids].sum().item())
    total = float(baseline.sum().item())
    percent = 100.0 * sink_budget / total if total > 0.0 else 0.0

    side_px = grid_size * 28
    image_array = _load_square_display_image(snapshot.image_path, side_px=side_px)
    height, width = image_array.shape[:2]
    cell_w = width / grid_size
    cell_h = height / grid_size

    order = torch.argsort(baseline[sink_ids], descending=True)
    ordered_sink_ids = sink_ids[order]

    fig, (ax_image, ax_values) = plt.subplots(
        1,
        2,
        figsize=(13.5, 10),
        dpi=220,
        gridspec_kw={"width_ratios": [1.0, 0.36]},
    )
    ax_image.imshow(image_array)
    _draw_patch_grid(ax_image, grid_size=grid_size, width=width, height=height)

    for rank, local_id_t in enumerate(ordered_sink_ids, start=1):
        local_id = int(local_id_t.item())
        row, col = divmod(local_id, grid_size)
        ax_image.add_patch(
            Rectangle(
                (col * cell_w, row * cell_h),
                cell_w,
                cell_h,
                linewidth=2.1,
                edgecolor="#5b006b",
                facecolor="#8e24aa",
                alpha=0.72,
            )
        )
        ax_image.text(
            col * cell_w + cell_w / 2,
            row * cell_h + cell_h / 2,
            str(rank),
            ha="center",
            va="center",
            fontsize=_rank_fontsize(int(sink_ids.numel())),
            color="black",
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
        )

    ax_image.set_title(f"Runtime sink tokens ({int(sink_ids.numel())}, {percent:.2f}% baseline mass)", fontsize=14)
    ax_image.set_xticks([])
    ax_image.set_yticks([])
    ax_image.set_xlim(0, width)
    ax_image.set_ylim(height, 0)

    value_lines = ["rank  token  baseline"]
    for rank, local_id_t in enumerate(ordered_sink_ids, start=1):
        local_id = int(local_id_t.item())
        value_lines.append(f"{rank:>4}  {local_id:>5}  {float(baseline[local_id].item()):.6g}")
    ax_values.axis("off")
    ax_values.text(
        0.0,
        1.0,
        "\n".join(value_lines),
        ha="left",
        va="top",
        family="monospace",
        fontsize=_table_fontsize(int(sink_ids.numel())),
    )

    fig.suptitle(f"Sink token positions (question_id={snapshot.question_id})", fontsize=16, y=0.98)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    return out_path


def _plot_sink_budget_summary(
    snapshot: PrefillDebugSnapshot,
    out_path: Path,
    *,
    baseline: torch.Tensor,
    redistributed: Optional[torch.Tensor],
    sink_ids: torch.Tensor,
    receiver_count: int,
    exact_receiver_weights: bool,
) -> Dict[str, float]:
    baseline_total = float(baseline.sum().item())
    computed_sink_budget = float(baseline[sink_ids].sum().item()) if sink_ids.numel() > 0 else 0.0
    stored_sink_budget = getattr(snapshot, "sink_budget", None)
    sink_budget = float(stored_sink_budget) if stored_sink_budget is not None else computed_sink_budget
    sink_budget_percent = 100.0 * sink_budget / baseline_total if baseline_total > 0.0 else 0.0
    redistributed_total = float(redistributed.sum().item()) if redistributed is not None else 0.0

    fig, (ax_bar, ax_text) = plt.subplots(
        1,
        2,
        figsize=(11.5, 4.8),
        dpi=220,
        gridspec_kw={"width_ratios": [0.95, 1.05]},
    )
    non_sink = max(baseline_total - sink_budget, 0.0)
    ax_bar.bar(["sink budget", "remaining baseline"], [sink_budget, non_sink], color=["#8e24aa", "#607d8b"])
    ax_bar.set_ylabel("baseline_scores mass")
    ax_bar.set_title(f"Sink budget = {sink_budget_percent:.2f}%")
    ax_bar.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.5)

    lines = [
        f"question_id: {snapshot.question_id}",
        f"sink tokens: {int(sink_ids.numel())}",
        f"receiver tokens: {receiver_count}",
        f"receiver weights: {'exact' if exact_receiver_weights else 'inferred/unavailable'}",
        f"baseline total: {baseline_total:.8g}",
        f"sink budget: {sink_budget:.8g}",
        f"sink budget %: {sink_budget_percent:.4f}",
        f"redistributed total: {redistributed_total:.8g}",
    ]
    if stored_sink_budget is not None:
        lines.append(f"computed sink budget: {computed_sink_budget:.8g}")

    ax_text.axis("off")
    ax_text.text(0.0, 1.0, "\n".join(lines), ha="left", va="top", family="monospace", fontsize=10)
    fig.suptitle("Sink budget accounting", fontsize=15, y=0.98)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)

    return {
        "baseline_total": baseline_total,
        "redistributed_total": redistributed_total,
        "sink_budget": sink_budget,
        "computed_sink_budget": computed_sink_budget,
        "sink_budget_percent": sink_budget_percent,
        "sink_count": float(int(sink_ids.numel())),
        "receiver_count": float(receiver_count),
    }


def _visual_hidden_states(snapshot: PrefillDebugSnapshot, expected_tokens: int) -> Optional[torch.Tensor]:
    """Return the [N, D] visual-token hidden states at the prune layer, local-id indexed.

    Prefers the precomputed visual view; older snapshots without it are served by slicing the
    full-sequence tensor at the image token block.
    """
    states = getattr(snapshot, "visual_hidden_states", None)
    if states is not None:
        states = torch.as_tensor(states).detach().float()
        if states.dim() == 2 and states.shape[0] == expected_tokens:
            return states

    full = getattr(snapshot, "hidden_states_at_prune_layer", None)
    if full is None:
        return None
    full = torch.as_tensor(full).detach().float()
    start = int(snapshot.image_token_start_index)
    end = start + expected_tokens
    if full.dim() != 2 or end > full.shape[0]:
        return None
    return full[start:end]


def _hidden_state_token_label(rank: int, local_id: int, score: float, role: str) -> str:
    return f"#{rank} tok={local_id} {role[0].upper() if role else '-'} s={score:.4g}"


def _top_hidden_state_rows(
    redistributed: torch.Tensor,
    hidden_states: torch.Tensor,
    role_by_token: Dict[int, str],
) -> List[Tuple[int, int, float, str, torch.Tensor]]:
    """(rank, local_id, score, role, hidden_vector) for the top-k redistributed tokens."""
    top_values, top_local_ids = _top_token_values_and_ids(redistributed, HIDDEN_STATE_TOP_K)
    rows: List[Tuple[int, int, float, str, torch.Tensor]] = []
    for rank, (local_id_t, value_t) in enumerate(zip(top_local_ids, top_values), start=1):
        local_id = int(local_id_t.item())
        role = role_by_token.get(local_id, ROLE_NATIVE)
        rows.append((rank, local_id, float(value_t.item()), role, hidden_states[local_id]))
    return rows


def _plot_top_hidden_states_3d(
    snapshot: PrefillDebugSnapshot,
    out_path: Path,
    *,
    rows: List[Tuple[int, int, float, str, torch.Tensor]],
) -> Optional[Path]:
    """Stack each top token's hidden-state vector along y, highlighting its two largest |dims|."""
    if not rows:
        return None

    hidden_dim = int(rows[0][4].numel())
    xs_full = np.arange(hidden_dim)
    top1_c, top2_c = "#d62728", "#ff9d3a"

    fig = plt.figure(figsize=(28, 12), dpi=220)
    ax = fig.add_subplot(projection="3d")
    fig.subplots_adjust(left=0.04, right=0.92, bottom=0.08, top=0.98)

    z_min = 0.0
    z_max = 0.0
    top1_dims: set = set()
    top2_dims: set = set()
    for index, (_rank, _local_id, _score, role, hidden_vector) in enumerate(rows):
        z_values = hidden_vector.numpy()
        z_min = min(z_min, float(z_values.min()))
        z_max = max(z_max, float(z_values.max()))

        ax.plot(
            xs_full,
            np.full(hidden_dim, index),
            z_values,
            color=ROLE_COLORS.get(role, ROLE_COLORS[ROLE_NATIVE]),
            linewidth=0.6,
            alpha=0.85,
            zorder=1,
        )

        for j, dim in enumerate(np.argsort(np.abs(z_values))[-2:][::-1].tolist()):
            (top1_dims if j == 0 else top2_dims).add(int(dim))
            ax.bar3d(
                dim - hidden_dim * 0.003,
                index - 0.16,
                0,
                hidden_dim * 0.006,
                0.32,
                float(z_values[dim]),
                color=top1_c if j == 0 else top2_c,
                shade=True,
                alpha=0.97,
                zorder=5,
            )

    ax.set_title(
        f"Hidden states of the top {len(rows)} tokens by redistributed_scores "
        f"(question_id={snapshot.question_id})",
        fontsize=17,
        pad=18,
    )
    ax.set_xlabel("hidden dimension", fontsize=15, labelpad=18)
    # No y axis label: the rank tick labels are long and self-describing, and any label placed
    # clear of them ends up off the canvas.

    xtick_pairs = sorted(
        {(d, top1_c) for d in top1_dims} | {(d, top2_c) for d in top2_dims if d not in top1_dims},
        key=lambda pair: pair[0],
    )
    ax.set_xticks([dim for dim, _ in xtick_pairs])
    ax.set_xticklabels([str(dim) for dim, _ in xtick_pairs], fontsize=13, fontweight="bold")
    for tick, (_dim, color) in zip(ax.get_xticklabels(), xtick_pairs):
        tick.set_color(color)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(
        [_hidden_state_token_label(rank, local_id, score, role) for rank, local_id, score, role, _ in rows],
        fontsize=13,
    )
    ax.tick_params(axis="z", labelsize=12)

    ax.set_ylim(-0.5, len(rows) - 0.5)
    z_span = max(z_max - z_min, 1e-6)
    ax.set_zlim(z_min - z_span * 0.06, z_max + z_span * 0.06)
    ax.view_init(elev=20, azim=-58)
    ax.set_box_aspect((2.2, 1.0, 0.85))

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor((0.85, 0.85, 0.85, 1.0))
        axis._axinfo["grid"].update(color=(0.9, 0.9, 0.9, 1.0), linewidth=0.6)

    ax.legend(
        handles=[
            Line2D([0], [0], color=top1_c, lw=6, label="top-1 dim"),
            Line2D([0], [0], color=top2_c, lw=6, label="top-2 dim"),
            Line2D([0], [0], color=ROLE_COLORS[ROLE_RECEIVER], lw=6, label="receiver"),
            Line2D([0], [0], color=ROLE_COLORS[ROLE_NATIVE], lw=6, label="native"),
        ],
        loc="upper left",
        fontsize=13,
        framealpha=0.9,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.4)
    plt.close(fig)
    return out_path


def _plot_top_hidden_states_lines(
    snapshot: PrefillDebugSnapshot,
    out_path: Path,
    *,
    rows: List[Tuple[int, int, float, str, torch.Tensor]],
) -> Optional[Path]:
    """Flat per-token line plots of the same vectors, readable down to individual dimensions."""
    if not rows:
        return None

    hidden_dim = int(rows[0][4].numel())
    xs = np.arange(hidden_dim)

    fig, axes = plt.subplots(
        len(rows),
        1,
        figsize=(16, 2.6 * len(rows)),
        dpi=200,
        sharex=True,
    )
    if len(rows) == 1:
        axes = [axes]

    for ax, (rank, local_id, score, role, hidden_vector) in zip(axes, rows):
        z_values = hidden_vector.numpy()
        ax.plot(xs, z_values, color=ROLE_COLORS.get(role, ROLE_COLORS[ROLE_NATIVE]), linewidth=0.7)
        ax.margins(y=0.16)  # headroom so the peak annotations are not clipped
        ax.axhline(0.0, color="#9e9e9e", linewidth=0.6, linestyle=":")
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
        ax.set_ylabel(
            _hidden_state_token_label(rank, local_id, score, role),
            fontsize=9,
            rotation=0,
            ha="right",
            va="center",
            labelpad=58,
        )
        for dim in np.argsort(np.abs(z_values))[-2:][::-1].tolist():
            value = float(z_values[dim])
            ax.plot([dim], [value], marker="o", markersize=4, color="#d62728")
            ax.annotate(
                f"dim {int(dim)}: {value:.3g}",
                xy=(dim, value),
                xytext=(6, 4),
                textcoords="offset points",
                fontsize=8,
                color="#d62728",
            )

    axes[-1].set_xlabel("hidden dimension", fontsize=11)
    fig.suptitle(
        f"Top {len(rows)} redistributed visual tokens: hidden states "
        f"(question_id={snapshot.question_id})",
        fontsize=14,
        y=0.995,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return out_path


def render_collection(snapshot_path: Path, out_root: Path, question_subdir: bool) -> List[Path]:
    snapshot = load_snapshot(snapshot_path)
    out_dir = _collection_dir(out_root, snapshot, question_subdir)
    paths: List[Path] = []
    index = {
        "snapshot": str(snapshot_path),
        "question_id": snapshot.question_id,
        "plots": {},
        "notes": [],
    }

    path = _save_original_image(snapshot, out_dir / "original_image.png")
    if path is not None:
        paths.append(path)
        index["plots"]["original_image"] = str(path)
    else:
        index["notes"].append(f"original image missing at snapshot.image_path={snapshot.image_path!r}; original image export skipped.")

    path = _save_square_image(snapshot, out_dir / "original_image_square.png")
    if path is not None:
        paths.append(path)
        index["plots"]["original_image_square"] = str(path)

    baseline = _baseline_scores(snapshot)
    redistributed = _redistributed_scores(snapshot)
    if baseline is None:
        raise ValueError(f"{snapshot_path} has no visual_self_attention/baseline_scores vector.")

    n = int(baseline.numel())
    sink_ids = _runtime_sink_ids(snapshot, n)
    receiver_indices = torch.empty(0, dtype=torch.long)
    receiver_weights = torch.empty(0, dtype=torch.float32)
    exact_receivers = False
    if redistributed is not None and redistributed.numel() == baseline.numel():
        receiver_indices, receiver_weights, exact_receivers = _receiver_data(snapshot, baseline, redistributed)
    else:
        index["notes"].append("redistributed_scores missing or length mismatch; redistributed and receiver plots skipped.")

    for top_k in TOP_TOKEN_COUNTS:
        path = _plot_top_tokens_on_image(
            snapshot,
            out_dir / f"baseline_scores_top_{top_k}.png",
            vector=baseline,
            top_k=top_k,
            title="baseline_scores",
            table_score_label="score",
        )
        if path is not None:
            paths.append(path)
            index["plots"][f"baseline_scores_top_{top_k}"] = str(path)
        path = _plot_top_token_positions_on_image(
            snapshot,
            out_dir / f"baseline_scores_positions_top_{top_k}.png",
            vector=baseline,
            top_k=top_k,
            edge_color=POSITION_HIGHLIGHT_COLOR,
            face_color=POSITION_HIGHLIGHT_COLOR,
        )
        if path is not None:
            paths.append(path)
            index["plots"][f"baseline_scores_positions_top_{top_k}"] = str(path)

    if redistributed is not None and redistributed.numel() == baseline.numel():
        roles = _classify_redistributed_roles(
            baseline,
            redistributed,
            exact_receivers=receiver_indices if exact_receivers else None,
            sink_ids=sink_ids,
        )
        for top_k in TOP_TOKEN_COUNTS:
            path = _plot_top_tokens_on_image(
                snapshot,
                out_dir / f"redistributed_scores_top_{top_k}.png",
                vector=redistributed,
                top_k=top_k,
                title="redistributed_scores",
                table_score_label="score",
                role_by_token=roles,
                sink_ids=sink_ids,
            )
            if path is not None:
                paths.append(path)
                index["plots"][f"redistributed_scores_top_{top_k}"] = str(path)
            path = _plot_top_token_positions_on_image(
                snapshot,
                out_dir / f"redistributed_scores_positions_top_{top_k}.png",
                vector=redistributed,
                top_k=top_k,
                edge_color=POSITION_HIGHLIGHT_COLOR,
                face_color=POSITION_HIGHLIGHT_COLOR,
            )
            if path is not None:
                paths.append(path)
                index["plots"][f"redistributed_scores_positions_top_{top_k}"] = str(path)

        receiver_name = "receiver_weights_on_image" if exact_receivers else "receiver_weights_inferred_on_image"
        path = _plot_receiver_weights_on_image(
            snapshot,
            out_dir / f"{receiver_name}.png",
            receiver_indices=receiver_indices,
            receiver_weights=receiver_weights,
            exact=exact_receivers,
        )
        if path is not None:
            paths.append(path)
            index["plots"][receiver_name] = str(path)
        if not exact_receivers:
            index["notes"].append(
                "Exact receiver_indices/receiver_weights are missing; receiver plot is inferred from redistributed-baseline. "
                "Recapture with the updated capture script for exact runtime weights."
            )

        hidden_states = _visual_hidden_states(snapshot, n)
        if hidden_states is None:
            index["notes"].append(
                "visual_hidden_states unavailable (and no usable hidden_states_at_prune_layer slice); "
                f"{HIDDEN_STATE_FOLDER} plots skipped."
            )
        else:
            hidden_dir = out_dir / HIDDEN_STATE_FOLDER
            hidden_rows = _top_hidden_state_rows(redistributed, hidden_states, roles)
            path = _plot_top_hidden_states_3d(
                snapshot,
                hidden_dir / f"top_{HIDDEN_STATE_TOP_K}_redistributed_hidden_states_3d.png",
                rows=hidden_rows,
            )
            if path is not None:
                paths.append(path)
                index["plots"][f"redistributed_top_{HIDDEN_STATE_TOP_K}_hidden_states_3d"] = str(path)
            path = _plot_top_hidden_states_lines(
                snapshot,
                hidden_dir / f"top_{HIDDEN_STATE_TOP_K}_redistributed_hidden_states_lines.png",
                rows=hidden_rows,
            )
            if path is not None:
                paths.append(path)
                index["plots"][f"redistributed_top_{HIDDEN_STATE_TOP_K}_hidden_states_lines"] = str(path)

    kept_ids = _valid_ids(getattr(snapshot, "kept_visual_local_ids", None), n)
    if kept_ids.numel() > 0:
        path = _plot_kept_tokens_on_image(
            snapshot,
            out_dir / "kept_tokens.png",
            kept_ids=kept_ids,
            n=n,
        )
        if path is not None:
            paths.append(path)
            index["plots"]["kept_tokens"] = str(path)
    else:
        index["notes"].append("kept_visual_local_ids missing or empty; keep/drop overlay skipped.")

    path = _plot_runtime_sink_positions(
        snapshot,
        out_dir / "runtime_sink_positions.png",
        baseline=baseline,
        sink_ids=sink_ids,
    )
    if path is not None:
        paths.append(path)
        index["plots"]["runtime_sink_positions"] = str(path)

    metrics = _plot_sink_budget_summary(
        snapshot,
        out_dir / "sink_budget_summary.png",
        baseline=baseline,
        redistributed=redistributed,
        sink_ids=sink_ids,
        receiver_count=int(receiver_indices.numel()),
        exact_receiver_weights=exact_receivers,
    )
    paths.append(out_dir / "sink_budget_summary.png")
    index["plots"]["sink_budget_summary"] = str(out_dir / "sink_budget_summary.png")
    index["metrics"] = metrics

    index_path = out_dir / "collection_index.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    paths.append(index_path)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snapshot", nargs="+", required=True, help="Snapshot .pt file(s), or directory/directories of .pt snapshots.")
    parser.add_argument("--out-dir", type=str, default="visualization/output")
    parser.add_argument("--recursive", action="store_true", help="When --snapshot includes a directory, search recursively for .pt files.")
    parser.add_argument(
        "--no-question-subdir",
        dest="question_subdir",
        action="store_false",
        help="Write straight into --out-dir/combined_collection instead of --out-dir/<question_id>/combined_collection.",
    )
    parser.set_defaults(question_subdir=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshots = _expand_snapshot_inputs(args.snapshot, recursive=args.recursive)
    if not snapshots:
        raise SystemExit("No snapshots found.")

    out_root = Path(args.out_dir)
    for snapshot_path in snapshots:
        paths = render_collection(snapshot_path, out_root, question_subdir=args.question_subdir)
        print(f"[{snapshot_path}] wrote {len(paths)} files")
        for path in paths:
            print(f"  {path}")


if __name__ == "__main__":
    main()
