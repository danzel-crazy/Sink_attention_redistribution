"""Sink-token visualizations.

Every plot here concerns the sink tokens' hidden states and writes into the
`sink_tokens/` subfolder of the run's output directory. Nothing executes on import
beyond registering the tagged plots, so importing this file has no side effects and it
is never imported by the benchmark entrypoints.

Each function takes `(snapshot: PrefillDebugSnapshot, out_dir: Path, **_) -> List[Path]`
and is driven by `visualization/render_fastv_visualizations.py`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - registers the 3d projection
from PIL import Image, ImageOps

from visualization.fastv_snapshot import PrefillDebugSnapshot
from visualization.viz_registry import subdir, visualization

# Every plot in this module lands here, under the run's output directory.
SUBDIR = "sink_tokens"

# These settings are used only by the offline visualization pipeline. They do not affect
# FastV inference, sink membership during benchmarking, or runtime redistribution logic.
OFFLINE_SINK_DIMS = (1415, 2533)
OFFLINE_SINK_METHODS = ("max", "sum", "layernorm")
OFFLINE_SINK_SCORE_QUANTILE = 0.99
RANDOM_NON_SINK_DETAIL_COUNT = 6
RANDOM_NON_SINK_DETAIL_SEED = 2533


@dataclass(frozen=True)
class OfflineSinkView:
    """One visualization-only sink-token configuration derived from a snapshot."""

    method: str
    sink_dims: Tuple[int, ...]
    sink_values: torch.Tensor
    sink_local_ids: torch.Tensor
    sink_abs_ids: torch.Tensor
    sink_scores: torch.Tensor


def _phi_profile(hidden_vector: torch.Tensor) -> torch.Tensor:
    """Return the RMS-normalized absolute activation profile for one token."""
    hidden_vector = hidden_vector.detach().float()
    rms = torch.sqrt(torch.mean(hidden_vector ** 2) + 1e-6)
    return torch.abs(hidden_vector) / rms


def _compute_sink_values(
    hidden_states: torch.Tensor,
    valid_dims: Tuple[int, ...],
    method: str,
) -> torch.Tensor:
    """Compute one offline sink score per visual token using the requested reduction."""
    hidden_states = hidden_states.detach().float()
    hidden_dim = hidden_states.shape[1]
    mean_square_per_token = torch.sum(hidden_states ** 2, dim=1) / hidden_dim
    denom = torch.sqrt(mean_square_per_token + 1e-6)
    sink_candidates = torch.abs(hidden_states[:, list(valid_dims)] / denom.unsqueeze(1))

    if method == "max":
        return torch.max(sink_candidates, dim=1).values
    if method == "sum":
        return torch.sum(sink_candidates, dim=1)
    raise ValueError(f"Unsupported offline sink method: {method}")


def _has_layernorm_snapshot(snapshot: PrefillDebugSnapshot) -> bool:
    """Return whether the snapshot carries the prune-layer RMSNorm data needed for exact layernorm scoring."""
    weight = getattr(snapshot, "prune_layer_input_layernorm_weight", None)
    eps = getattr(snapshot, "prune_layer_input_layernorm_eps", None)
    return isinstance(weight, torch.Tensor) and weight.numel() > 0 and eps is not None


def _compute_layernorm_sink_values(snapshot: PrefillDebugSnapshot, valid_dims: Tuple[int, ...]) -> torch.Tensor:
    """Compute sink scores from the exact prune-layer input RMSNorm saved in the snapshot."""
    if not _has_layernorm_snapshot(snapshot):
        raise ValueError("Snapshot is missing prune-layer input_layernorm data required for layernorm sink scores.")

    hidden_states = snapshot.visual_hidden_states.detach().float()
    weight = snapshot.prune_layer_input_layernorm_weight.detach().float()
    eps = float(snapshot.prune_layer_input_layernorm_eps)

    variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
    normalized = hidden_states * torch.rsqrt(variance + eps)
    normalized = normalized * weight.unsqueeze(0)
    sink_candidates = normalized[:, list(valid_dims)].abs()
    return torch.max(sink_candidates, dim=-1).values


def _resolve_valid_sink_dims(snapshot: PrefillDebugSnapshot) -> Tuple[int, ...]:
    """Filter the requested sink dims against the hidden size saved in the snapshot."""
    hidden_dim = int(snapshot.visual_hidden_states.shape[1])
    valid_dims = tuple(dim for dim in OFFLINE_SINK_DIMS if 0 <= int(dim) < hidden_dim)
    if not valid_dims:
        raise ValueError(
            f"No valid offline sink dims in {list(OFFLINE_SINK_DIMS)} for hidden_dim={hidden_dim}."
        )
    return valid_dims


def _select_sink_tokens(sink_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply the same quantile-to-max selection rule used by the runtime sink selector."""
    sink_values = sink_values.detach().float().flatten()
    if sink_values.numel() == 0:
        empty_ids = torch.empty(0, dtype=torch.long)
        empty_scores = torch.empty(0, dtype=torch.float32)
        return empty_ids, empty_scores

    score_min = float(torch.quantile(sink_values, OFFLINE_SINK_SCORE_QUANTILE).item())
    score_max = float(torch.max(sink_values).item())
    selected = torch.nonzero(
        (sink_values >= score_min) & (sink_values <= score_max),
        as_tuple=False,
    ).flatten()
    if selected.numel() == 0:
        empty_ids = torch.empty(0, dtype=torch.long)
        empty_scores = torch.empty(0, dtype=torch.float32)
        return empty_ids, empty_scores

    selected_scores = sink_values[selected]
    order = torch.argsort(selected_scores, descending=True)
    return selected[order], selected_scores[order]


def _build_offline_sink_view(snapshot: PrefillDebugSnapshot, method: str) -> OfflineSinkView:
    """Recompute sink scores/tokens from the saved snapshot for one offline method."""
    valid_dims = _resolve_valid_sink_dims(snapshot)
    if method == "layernorm":
        sink_values = _compute_layernorm_sink_values(snapshot, valid_dims)
    else:
        sink_values = _compute_sink_values(snapshot.visual_hidden_states, valid_dims, method)
    sink_local_ids, sink_scores = _select_sink_tokens(sink_values)
    sink_abs_ids = sink_local_ids + int(snapshot.image_token_start_index)
    return OfflineSinkView(
        method=method,
        sink_dims=valid_dims,
        sink_values=sink_values.detach().float(),
        sink_local_ids=sink_local_ids.detach().long(),
        sink_abs_ids=sink_abs_ids.detach().long(),
        sink_scores=sink_scores.detach().float(),
    )


def _offline_sink_views(snapshot: PrefillDebugSnapshot) -> List[OfflineSinkView]:
    """Return the visualization-only sink views for max, sum, and exact layernorm scoring."""
    methods = list(OFFLINE_SINK_METHODS)
    if not _has_layernorm_snapshot(snapshot):
        methods = [method for method in methods if method != "layernorm"]
    return [_build_offline_sink_view(snapshot, method) for method in methods]


def _method_out_dir(out_dir: Path, sink_view: OfflineSinkView) -> Path:
    """Return the output folder for one sink-score method under sink_tokens/."""
    return subdir(out_dir, f"{SUBDIR}/{sink_view.method}")


def _resolve_plot_attention(snapshot: PrefillDebugSnapshot) -> Tuple[torch.Tensor, str]:
    """Choose the offline attention vector paired with sink-token plots."""
    return snapshot.visual_self_attention.detach().float().flatten(), "Visual self-attention"


def _sorted_sink_order(sink_view: OfflineSinkView) -> List[int]:
    """Return selected sink-token indices sorted by descending offline sink score."""
    if sink_view.sink_scores.numel() == 0:
        return []
    return list(range(int(sink_view.sink_scores.numel())))


def _random_non_sink_local_ids(sink_view: OfflineSinkView, image_token_length: int) -> List[int]:
    """Return a deterministic random sample of visual-token ids excluded from this sink set."""
    if image_token_length <= 0:
        return []

    sink_ids = {int(local_id.item()) for local_id in sink_view.sink_local_ids}
    candidates = torch.tensor(
        [local_id for local_id in range(image_token_length) if local_id not in sink_ids],
        dtype=torch.long,
    )
    if candidates.numel() == 0:
        return []

    method_offsets = {"max": 0, "sum": 10_000, "layernorm": 20_000}
    generator = torch.Generator().manual_seed(
        RANDOM_NON_SINK_DETAIL_SEED + method_offsets.get(sink_view.method, 30_000)
    )
    sample_count = min(
        RANDOM_NON_SINK_DETAIL_COUNT,
        int(sink_view.sink_scores.numel()),
        int(candidates.numel()),
    )
    order = torch.randperm(int(candidates.numel()), generator=generator)[:sample_count]
    return [int(local_id.item()) for local_id in candidates[order]]


def _visual_grid_size(image_token_length: int) -> Optional[int]:
    """Return the patch-grid width/height for square visual-token layouts."""
    grid = int(round(math.sqrt(image_token_length)))
    return grid if grid * grid == image_token_length else None


def _load_square_display_image(image_path: str, side_px: int) -> np.ndarray:
    """Load the source image and crop it to a square display canvas for overlays."""
    resampling = getattr(Image, "Resampling", Image)
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        return np.full((side_px, side_px, 3), 255, dtype=np.uint8)
    square = ImageOps.fit(image, (side_px, side_px), method=resampling.BICUBIC)
    return np.asarray(square)


def _draw_patch_grid(ax, grid_size: int, width: int, height: int) -> None:
    """Draw the visual-token patch lattice on top of an image."""
    xs = np.linspace(0, width, grid_size + 1)
    ys = np.linspace(0, height, grid_size + 1)
    for x in xs:
        ax.axvline(x=x, color="white", linewidth=0.45, alpha=0.38)
    for y in ys:
        ax.axhline(y=y, color="white", linewidth=0.45, alpha=0.38)


def _sink_plot_layer(snapshot: PrefillDebugSnapshot) -> int:
    """Return the prune-boundary layer index used by FastV."""
    return int(snapshot.fastv_k) - 1


def _method_title_suffix(sink_view: OfflineSinkView) -> str:
    """Human-readable label for the current offline sink-score setting."""
    dims = ", ".join(str(dim) for dim in sink_view.sink_dims)
    return f"{sink_view.method} score, dims=[{dims}]"


def _sink_dimension_profile(
    hidden_vector: torch.Tensor,
    sink_view: OfflineSinkView,
    snapshot: PrefillDebugSnapshot,
) -> torch.Tensor:
    """Return the per-dimension contribution to the active offline sink score."""
    if sink_view.method == "layernorm" and _has_layernorm_snapshot(snapshot):
        hidden_vector = hidden_vector.detach().float()
        weight = snapshot.prune_layer_input_layernorm_weight.detach().float()
        eps = float(snapshot.prune_layer_input_layernorm_eps)
        variance = hidden_vector.pow(2).mean()
        normalized = hidden_vector * torch.rsqrt(variance + eps)
        normalized = normalized * weight
        profile = torch.zeros_like(normalized)
        profile[list(sink_view.sink_dims)] = normalized[list(sink_view.sink_dims)].abs()
        return profile

    phi = _phi_profile(hidden_vector)
    profile = torch.zeros_like(phi)
    profile[list(sink_view.sink_dims)] = phi[list(sink_view.sink_dims)]
    return profile


def _plot_sink_token_hidden_states(
    snapshot: PrefillDebugSnapshot,
    out_dir: Path,
    sink_view: OfflineSinkView,
    order: List[int],
    out_name: str,
    y_label: str,
    token_labels: List[str],
    *,
    z_value_mode: str = "hidden_state",
) -> List[Path]:
    """Render a 3D summary of multiple sink-token hidden-state vectors."""
    if sink_view.sink_scores.numel() == 0:
        return []

    num_tokens = len(order)
    hidden_dim = snapshot.hidden_states_at_prune_layer.shape[-1]
    xs_full = np.arange(hidden_dim)

    fig = plt.figure(figsize=(28, 12), dpi=220)
    ax = fig.add_subplot(projection="3d")
    fig.subplots_adjust(left=0.04, right=0.92, bottom=0.08, top=0.98)

    line_colors = plt.cm.viridis(np.linspace(0.12, 0.82, num_tokens))
    top1_c, top2_c = "#d62728", "#ff9d3a"

    z_min = 0.0
    z_max = 0.0
    top1_dims: set = set()
    top2_dims: set = set()
    for rank, sink_idx_in_order in enumerate(order):
        abs_pos = int(sink_view.sink_abs_ids[sink_idx_in_order].item())
        hidden_vector_t = snapshot.hidden_states_at_prune_layer[abs_pos].detach().float()
        if z_value_mode == "sink_score":
            z_values_t = _sink_dimension_profile(hidden_vector_t, sink_view, snapshot)
            top_dims = [
                int(dim)
                for dim in torch.argsort(z_values_t, descending=True).tolist()
                if float(z_values_t[dim].item()) > 0.0
            ][:2]
            plot_title = f"Layer {_sink_plot_layer(snapshot)}: Sink-score dimensions ({_method_title_suffix(sink_view)})"
            # z_label = "sink score"
        else:
            z_values_t = hidden_vector_t
            top_dims = np.argsort(np.abs(z_values_t.numpy()))[-2:][::-1].tolist()
            plot_title = f"Layer {_sink_plot_layer(snapshot)}: Sink token hidden states ({_method_title_suffix(sink_view)})"
            # z_label = ""

        z_values = z_values_t.numpy()
        z_min = min(z_min, float(z_values.min()))
        z_max = max(z_max, float(z_values.max()))

        ax.plot(
            xs_full,
            np.full(hidden_dim, rank),
            z_values,
            color=line_colors[rank],
            linewidth=0.6,
            alpha=0.85,
            zorder=1,
        )

        for j, dim in enumerate(top_dims):
            if not (0 <= dim < hidden_dim):
                continue
            h = float(z_values[dim])
            c = top1_c if j == 0 else top2_c
            (top1_dims if j == 0 else top2_dims).add(int(dim))
            ax.bar3d(
                dim - hidden_dim * 0.003,
                rank - 0.16,
                0,
                hidden_dim * 0.006,
                0.32,
                h,
                color=c,
                shade=True,
                alpha=0.97,
                zorder=5,
            )

    ax.set_title(
        plot_title,
        fontsize=17,
        pad=18,
    )
    ax.set_xlabel("hidden dimension", fontsize=15, labelpad=18)
    ax.set_ylabel(y_label, fontsize=15, labelpad=18)
    # ax.set_zlabel(z_label, fontsize=15, labelpad=10)

    xtick_pairs = sorted(
        {(d, top1_c) for d in top1_dims} | {(d, top2_c) for d in top2_dims if d not in top1_dims},
        key=lambda p: p[0],
    )
    ax.set_xticks([d for d, _ in xtick_pairs])
    ax.set_xticklabels([str(d) for d, _ in xtick_pairs], fontsize=13, fontweight="bold")
    for tick, (_, c) in zip(ax.get_xticklabels(), xtick_pairs):
        tick.set_color(c)
    ax.set_yticks(np.arange(num_tokens))
    ax.set_yticklabels(token_labels, fontsize=13)
    ax.tick_params(axis="z", labelsize=12)

    ax.set_ylim(-0.5, num_tokens - 0.5)
    z_span = max(z_max - z_min, 1e-6)
    ax.set_zlim(z_min - z_span * 0.06, z_max + z_span * 0.06)
    ax.view_init(elev=20, azim=-58)
    ax.set_box_aspect((2.2, 1.0, 0.85))

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor((0.85, 0.85, 0.85, 1.0))
        axis._axinfo["grid"].update(color=(0.9, 0.9, 0.9, 1.0), linewidth=0.6)

    from matplotlib.lines import Line2D

    ax.legend(
        handles=[
            Line2D([0], [0], color=top1_c, lw=6, label="top-1 dim"),
            Line2D([0], [0], color=top2_c, lw=6, label="top-2 dim"),
        ],
        loc="upper left",
        fontsize=13,
        framealpha=0.9,
    )

    out_path = _method_out_dir(out_dir, sink_view) / out_name
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.4)
    plt.close(fig)
    return [out_path]


@visualization
def plot_sink_tokens_pure_hidden_states(snapshot: PrefillDebugSnapshot, out_dir: Path, **_) -> List[Path]:
    """Plot each offline sink set in local-id order as a 3D hidden-state summary."""
    paths: List[Path] = []
    for sink_view in _offline_sink_views(snapshot):
        if sink_view.sink_scores.numel() == 0:
            continue
        order = list(range(int(sink_view.sink_scores.numel())))
        token_labels = [str(int(sink_view.sink_local_ids[idx].item())) for idx in order]
        paths.extend(
            _plot_sink_token_hidden_states(
                snapshot,
                out_dir,
                sink_view=sink_view,
                order=order,
                out_name="sink_tokens_pure_hidden_states_3d.png",
                y_label="sink tokens",
                token_labels=token_labels,
                z_value_mode="hidden_state",
            )
        )
    return paths


@visualization
def plot_sink_tokens_hidden_states_by_score(snapshot: PrefillDebugSnapshot, out_dir: Path, **_) -> List[Path]:
    """Plot each offline sink set in descending sink-score order as a 3D summary, including exact layernorm when the snapshot contains prune-layer RMSNorm data."""
    paths: List[Path] = []
    for sink_view in _offline_sink_views(snapshot):
        if sink_view.sink_scores.numel() == 0:
            continue
        order = _sorted_sink_order(sink_view)
        token_labels = [str(int(sink_view.sink_local_ids[idx].item())) for idx in order]
        paths.extend(
            _plot_sink_token_hidden_states(
                snapshot,
                out_dir,
                sink_view=sink_view,
                order=order,
                out_name="sink_tokens_hidden_states_by_score_3d.png",
                y_label="sink tokens",
                token_labels=token_labels,
                z_value_mode="sink_score",
            )
        )
    return paths


@visualization
def plot_sink_value_vs_visual_self_attention_profile(
    snapshot: PrefillDebugSnapshot,
    out_dir: Path,
    **_,
) -> List[Path]:
    """Plot all visual-token attention vs sink value for both offline sink-score methods."""
    if int(snapshot.image_token_length) <= 0:
        return []

    attention_scores, attention_label = _resolve_plot_attention(snapshot)
    paths: List[Path] = []
    for sink_view in _offline_sink_views(snapshot):
        n = min(int(snapshot.image_token_length), int(sink_view.sink_values.numel()), int(attention_scores.numel()))
        if n <= 0:
            continue

        sink_values = sink_view.sink_values[:n]
        plot_attention = attention_scores[:n]
        xs = np.arange(n)
        highlight_pairs = [
            (int(sink_view.sink_local_ids[idx].item()), idx)
            for idx in _sorted_sink_order(sink_view)
            if 0 <= int(sink_view.sink_local_ids[idx].item()) < n
        ]

        fig, ax = plt.subplots(figsize=(15, 5.2), dpi=220)
        # ax.plot(xs, plot_attention.numpy(), color="#1f77b4", linewidth=1.9, label=attention_label, zorder=2)
        ax.plot(
            xs,
            sink_values.numpy(),
            color="#d62728",
            linewidth=1.7,
            label=f"Sink value ({_method_title_suffix(sink_view)})",
            zorder=3,
        )

        if highlight_pairs:
            highlight_local_ids = [local_id for local_id, _ in highlight_pairs]
            ax.scatter(
                highlight_local_ids,
                plot_attention[highlight_local_ids].numpy(),
                color="#111111",
                s=42,
                marker="o",
                label="Selected sink tokens",
                zorder=5,
            )
            for local_id, sink_idx in highlight_pairs[:10]:
                ax.annotate(
                    f"idx={local_id}\nscore={float(sink_view.sink_scores[sink_idx].item()):.2f}",
                    xy=(local_id, float(plot_attention[local_id].item())),
                    xytext=(6, 8),
                    textcoords="offset points",
                    fontsize=8,
                    color="black",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.85),
                )
                ax.scatter(
                    [local_id],
                    [float(sink_values[local_id].item())],
                    color="#d62728",
                    s=30,
                    marker="x",
                    zorder=6,
                )

        # ax.text(
        #     0.01,
        #     0.98,
        #     f"offline sink view\n{_method_title_suffix(sink_view)}\nquantile={OFFLINE_SINK_SCORE_QUANTILE:.2f}",
        #     transform=ax.transAxes,
        #     ha="left",
        #     va="top",
        #     fontsize=10,
        #     bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.92, "edgecolor": "#dddddd"},
        # )

        # ax.set_title(
        #     f"Layer {_sink_plot_layer(snapshot)}: Visual self-attention and sink value per token",
        #     fontsize=16,
        # )
        ax.set_xlabel("visual tokens id", fontsize=13)
        ax.set_ylabel("sink score", fontsize=13)
        ax.set_xlim(0, max(n - 1, 0))
        ax.grid(True, linestyle=":", linewidth=0.65, alpha=0.45)
        ax.legend(loc="best", fontsize=11, framealpha=0.95)

        out_path = _method_out_dir(out_dir, sink_view) / "sink_value_vs_visual_self_attention_profile.png"
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.25)
        plt.close(fig)
        paths.append(out_path)
    return paths


def _plot_visual_token_hidden_state_detail(
    snapshot: PrefillDebugSnapshot,
    sink_view: OfflineSinkView,
    output_dir: Path,
    *,
    local_id: int,
    token_kind: str,
    out_name: str,
    score: Optional[float] = None,
    mark_top_dims: bool = True,
) -> Path:
    """Render one visual token's hidden activations and phi profile."""
    layer = _sink_plot_layer(snapshot)
    abs_id = int(snapshot.image_token_start_index) + local_id
    hidden_vector = snapshot.hidden_states_at_prune_layer[abs_id].detach().float()
    top_dims = torch.argsort(torch.abs(hidden_vector), descending=True)[:2].tolist() if mark_top_dims else []
    xs = np.arange(int(hidden_vector.numel()))

    fig, ax_hidden = plt.subplots(
        figsize=(16, 7.2),
        dpi=220,
    )

    ax_hidden.plot(xs, hidden_vector.numpy(), color="#4568dc", linewidth=1.0, alpha=0.95)
    ax_hidden.axhline(0.0, color="#888888", linewidth=0.8, alpha=0.75)
    for rank, dim in enumerate(top_dims):
        color = "#d62728" if rank == 0 else "#ff9d3a"
        value = float(hidden_vector[dim].item())
        y_offset = -14 if value >= 0 else 8
        vertical_align = "top" if value >= 0 else "bottom"
        ax_hidden.scatter([dim], [value], color=color, s=48, zorder=4)
        ax_hidden.annotate(
            f"dim {dim}",
            xy=(dim, value),
            xytext=(5, y_offset),
            textcoords="offset points",
            ha="left",
            va=vertical_align,
            fontsize=9,
            color=color,
            fontweight="bold",
            annotation_clip=True,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.9),
        )

    title_score = "" if score is None else f", score={score:.3f}"
    ax_hidden.margins(y=0.12)
    ax_hidden.set_ylabel("hidden state value", fontsize=12)
    ax_hidden.grid(True, linestyle=":", linewidth=0.55, alpha=0.45)
    ax_hidden.set_title(
        f"Layer {layer}: {token_kind} {local_id} hidden-state activations ({_method_title_suffix(sink_view)}{title_score})",
        fontsize=15,
    )
    ax_hidden.set_xlabel("hidden dimension", fontsize=12)

    # Phi-profile subplot intentionally disabled for hidden-state-only detail plots.
    # phi = _phi_profile(hidden_vector)
    # ax_phi.plot(xs, phi.numpy(), color="#2ca02c", linewidth=1.0, alpha=0.95, label="phi profile")
    # for rank, dim in enumerate(top_dims):
    #     color = "#d62728" if rank == 0 else "#ff9d3a"
    #     ax_phi.scatter([dim], [float(phi[dim].item())], color=color, s=40, zorder=4)
    # ax_phi.set_xlabel("hidden dimension", fontsize=12)
    # ax_phi.set_ylabel("phi", fontsize=12)
    # ax_phi.grid(True, linestyle=":", linewidth=0.55, alpha=0.45)
    # ax_phi.legend(loc="upper right", fontsize=10, framealpha=0.95)

    out_path = output_dir / out_name
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)
    return out_path


@visualization
def plot_sink_token_hidden_state_details(snapshot: PrefillDebugSnapshot, out_dir: Path, **_) -> List[Path]:
    """Render sink-token details plus random non-sink visual-token controls for each offline sink set."""
    paths: List[Path] = []
    for sink_view in _offline_sink_views(snapshot):
        if sink_view.sink_scores.numel() == 0:
            continue

        layer = _sink_plot_layer(snapshot)
        output_dir = _method_out_dir(out_dir, sink_view) / "sink_hidden_state_details"
        output_dir.mkdir(parents=True, exist_ok=True)

        for sink_idx in _sorted_sink_order(sink_view):
            local_id = int(sink_view.sink_local_ids[sink_idx].item())
            paths.append(
                _plot_visual_token_hidden_state_detail(
                    snapshot,
                    sink_view,
                    output_dir,
                    local_id=local_id,
                    token_kind="sink token",
                    out_name=f"layer_{layer:02d}_sink_token_{local_id}_hidden_state.png",
                    score=float(sink_view.sink_scores[sink_idx].item()),
                )
            )

        for local_id in _random_non_sink_local_ids(sink_view, int(snapshot.image_token_length)):
            paths.append(
                _plot_visual_token_hidden_state_detail(
                    snapshot,
                    sink_view,
                    output_dir,
                    local_id=local_id,
                    token_kind="random non-sink visual token",
                    out_name=f"layer_{layer:02d}_random_non_sink_visual_token_{local_id}_hidden_state.png",
                    score=float(sink_view.sink_values[local_id].item()),
                    mark_top_dims=False,
                )
            )

    return paths


@visualization
def plot_sink_token_positions_on_image(snapshot: PrefillDebugSnapshot, out_dir: Path, **_) -> List[Path]:
    """Overlay selected sink tokens on the image patch grid for both offline sink views."""
    grid_size = _visual_grid_size(int(snapshot.image_token_length))
    if grid_size is None:
        return []

    side_px = grid_size * 28
    image_array = _load_square_display_image(snapshot.image_path, side_px=side_px)
    height, width = image_array.shape[:2]
    cell_w = width / grid_size
    cell_h = height / grid_size
    paths: List[Path] = []

    for sink_view in _offline_sink_views(snapshot):
        if sink_view.sink_scores.numel() == 0:
            continue

        fig, ax = plt.subplots(figsize=(10, 10), dpi=220)
        ax.imshow(image_array)

        for sink_idx in _sorted_sink_order(sink_view):
            local_id = int(sink_view.sink_local_ids[sink_idx].item())
            row, col = divmod(local_id, grid_size)
            x0 = col * cell_w
            y0 = row * cell_h
            # rect = Rectangle((x0, y0), cell_w, cell_h, linewidth=2.2, edgecolor="orange", facecolor="orange", alpha=0.5)
            rect = Rectangle((x0, y0), cell_w, cell_h, linewidth=2.2, edgecolor="purple", facecolor="purple", alpha=0.85)
            ax.add_patch(rect)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)

        out_path = _method_out_dir(out_dir, sink_view) / "sink_token_positions_on_image.png"
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
        paths.append(out_path)
    return paths


@visualization
def plot_sink_value_vs_visual_self_attention_cloud(
    snapshot: PrefillDebugSnapshot,
    out_dir: Path,
    **_,
) -> List[Path]:
    """Plot one sink-value/attention scatter cloud for each offline sink-score method."""
    if int(snapshot.image_token_length) <= 0:
        return []

    attention_scores, attention_label = _resolve_plot_attention(snapshot)
    paths: List[Path] = []
    for sink_view in _offline_sink_views(snapshot):
        n = min(int(snapshot.image_token_length), int(sink_view.sink_values.numel()), int(attention_scores.numel()))
        if n <= 0:
            continue

        sink_values = sink_view.sink_values[:n]
        plot_attention = torch.clamp(attention_scores[:n], min=1e-12)
        highlight_local_ids = [
            int(sink_view.sink_local_ids[idx].item())
            for idx in _sorted_sink_order(sink_view)
            if 0 <= int(sink_view.sink_local_ids[idx].item()) < n
        ]

        fig, ax = plt.subplots(figsize=(7.6, 5.8), dpi=220)
        ax.scatter(
            sink_values.numpy(),
            plot_attention.numpy(),
            s=10,
            alpha=0.22,
            c="#a64521",
            edgecolors="none",
            label=f"All visual tokens ({attention_label.lower()})",
        )

        if highlight_local_ids:
            ax.scatter(
                sink_values[highlight_local_ids].numpy(),
                plot_attention[highlight_local_ids].numpy(),
                s=42,
                c="black",
                marker="o",
                label="Selected sink tokens",
                zorder=3,
            )
            for local_id in highlight_local_ids[:10]:
                ax.annotate(
                    f"idx={local_id}",
                    xy=(float(sink_values[local_id].item()), float(plot_attention[local_id].item())),
                    xytext=(5, 6),
                    textcoords="offset points",
                    fontsize=8,
                    color="black",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.85),
                )

        # ax.text(
        #     0.01,
        #     0.99,
        #     f"offline sink view\n{_method_title_suffix(sink_view)}\nquantile={OFFLINE_SINK_SCORE_QUANTILE:.2f}",
        #     transform=ax.transAxes,
        #     ha="left",
        #     va="top",
        #     fontsize=9.5,
        #     bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.92, "edgecolor": "#dddddd"},
        # )

        ax.set_yscale("log")
        ax.set_xlabel(f"Sink score", fontsize=12)
        ax.set_ylabel(F"Attention", fontsize=12)
        # ax.set_title(f"Sink score vs attention", fontsize=15)
        ax.grid(True, linestyle=":", linewidth=0.55, alpha=0.4)
        ax.legend(loc="best", fontsize=10, framealpha=0.95)

        out_path = _method_out_dir(out_dir, sink_view) / "sink_value_vs_visual_self_attention_cloud.png"
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)
        paths.append(out_path)
    return paths
