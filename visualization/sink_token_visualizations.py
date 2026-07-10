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


def _phi_profile(hidden_vector: torch.Tensor) -> torch.Tensor:
    """Return the RMS-normalized absolute activation profile for one token."""
    # Same normalization sink_token_selector applies before reading sink dims, but
    # evaluated across every hidden dimension so it can drive offline plots too.
    hidden_vector = hidden_vector.detach().float()
    rms = torch.sqrt(torch.mean(hidden_vector ** 2) + 1e-6)
    return torch.abs(hidden_vector) / rms


def _compute_visual_phi_profiles(snapshot: PrefillDebugSnapshot) -> torch.Tensor:
    """Return the normalized absolute activation profile for every visual token."""
    visual_hidden = snapshot.visual_hidden_states.detach().float()
    rms = torch.sqrt(torch.mean(visual_hidden ** 2, dim=1, keepdim=True) + 1e-6)
    return torch.abs(visual_hidden) / rms


def _infer_single_sink_dim(snapshot: PrefillDebugSnapshot, phi_profiles: torch.Tensor) -> Optional[int]:
    """Infer a single sink dimension when selected sink scores are consistent with one dim."""
    if snapshot.sink_scores.numel() == 0:
        return None

    sink_local_ids = snapshot.sink_local_ids.detach().long()
    sink_scores = snapshot.sink_scores.detach().float()
    if sink_local_ids.numel() == 0 or phi_profiles.numel() == 0:
        return None
    if int(torch.max(sink_local_ids).item()) >= phi_profiles.shape[0]:
        return None

    selected_phi = phi_profiles[sink_local_ids]
    errors = torch.max(torch.abs(selected_phi - sink_scores.unsqueeze(1)), dim=0).values
    best_dim = int(torch.argmin(errors).item())
    best_error = float(errors[best_dim].item())

    if best_error <= 1e-4:
        return best_dim
    score_scale = max(float(torch.max(torch.abs(sink_scores)).item()), 1.0)
    if best_error <= 1e-3 * score_scale:
        return best_dim
    return None


def _resolve_visual_sink_values(snapshot: PrefillDebugSnapshot) -> Tuple[torch.Tensor, str, Optional[str]]:
    """Resolve the sink-value series used by the reference-style all-token plots."""
    visual_sink_values = getattr(snapshot, "visual_sink_values", None)
    if isinstance(visual_sink_values, torch.Tensor) and visual_sink_values.numel() == int(snapshot.image_token_length):
        return (
            visual_sink_values.detach().float().flatten(),
            "Sink value",
            None,
        )

    phi_profiles = _compute_visual_phi_profiles(snapshot)
    inferred_dim = _infer_single_sink_dim(snapshot, phi_profiles)
    if inferred_dim is not None:
        return (
            phi_profiles[:, inferred_dim].detach().float(),
            f"Sink value (inferred dim {inferred_dim})",
            "The snapshot does not store the full sink-value vector, so the sink dimension was inferred from the selected sink scores.",
        )

    return (
        torch.max(phi_profiles, dim=1).values.detach().float(),
        "Sink proxy (max phi over all dims)",
        "The snapshot does not store the full sink-value vector or sink dims, so the red series uses a full-dimension proxy.",
    )


def _resolve_plot_attention(snapshot: PrefillDebugSnapshot) -> Tuple[torch.Tensor, str]:
    """Choose the offline attention vector paired with sink-token plots."""
    # The snapshot keeps several attention views but not the original `exp` switch from
    # sink_value_fastv, so use the raw visual self-attention captured at the prune layer.
    return snapshot.visual_self_attention.detach().float().flatten(), "Visual self-attention"


def _sorted_sink_order(snapshot: PrefillDebugSnapshot) -> List[int]:
    """Return sink-token indices sorted by descending sink score."""
    if snapshot.sink_scores.numel() == 0:
        return []
    return torch.argsort(snapshot.sink_scores.detach().float(), descending=True).tolist()


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


def _plot_sink_token_hidden_states(
    snapshot: PrefillDebugSnapshot,
    out_dir: Path,
    order: List[int],
    out_name: str,
    y_label: str,
    token_labels: List[str],
) -> List[Path]:
    """Render a 3D summary of multiple sink-token hidden-state vectors."""
    if snapshot.sink_scores.numel() == 0:
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
        abs_pos = int(snapshot.sink_abs_ids[sink_idx_in_order].item())
        hidden_vector = snapshot.hidden_states_at_prune_layer[abs_pos].detach().float().numpy()
        top_dims = np.argsort(np.abs(hidden_vector))[-2:][::-1]
        z_min = min(z_min, float(hidden_vector.min()))
        z_max = max(z_max, float(hidden_vector.max()))

        ax.plot(
            xs_full,
            np.full(hidden_dim, rank),
            hidden_vector,
            color=line_colors[rank],
            linewidth=0.6,
            alpha=0.85,
            zorder=1,
        )

        for j, dim in enumerate(top_dims):
            if not (0 <= dim < hidden_dim):
                continue
            h = float(hidden_vector[dim])
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
            text_offset = max(z_max - z_min, 1e-6) * 0.02
            ax.text(
                dim,
                rank,
                h + (text_offset if h >= 0 else -text_offset),
                f"{int(dim)}",
                color=c,
                fontsize=12,
                fontweight="bold",
                ha="center",
                va="bottom" if h >= 0 else "top",
                zorder=10,
            )

    ax.set_xlabel("hidden dimension", fontsize=15, labelpad=18)
    ax.set_ylabel(y_label, fontsize=15, labelpad=18)
    ax.set_zlabel("hidden state value", fontsize=15, labelpad=10)

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

    out_path = subdir(out_dir, SUBDIR) / out_name
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.4)
    plt.close(fig)
    return [out_path]


@visualization
def plot_sink_tokens_pure_hidden_states(snapshot: PrefillDebugSnapshot, out_dir: Path, **_) -> List[Path]:
    """Plot every selected sink token in local-id order as a 3D hidden-state summary."""
    if snapshot.sink_scores.numel() == 0:
        return []

    order = list(range(int(snapshot.sink_scores.numel())))
    token_labels = [str(int(snapshot.sink_local_ids[idx].item())) for idx in order]
    return _plot_sink_token_hidden_states(
        snapshot,
        out_dir,
        order=order,
        out_name="sink_tokens_pure_hidden_states_3d.png",
        y_label="sink token local id",
        token_labels=token_labels,
    )


@visualization
def plot_sink_tokens_hidden_states_by_score(snapshot: PrefillDebugSnapshot, out_dir: Path, **_) -> List[Path]:
    """Plot every selected sink token in descending sink-score order as a 3D summary."""
    if snapshot.sink_scores.numel() == 0:
        return []

    order = torch.argsort(snapshot.sink_scores, descending=True).tolist()
    token_labels = [
        f"{int(snapshot.sink_local_ids[idx].item())} ({float(snapshot.sink_scores[idx].item()):.3f})"
        for idx in order
    ]
    return _plot_sink_token_hidden_states(
        snapshot,
        out_dir,
        order=order,
        out_name="sink_tokens_hidden_states_by_score_3d.png",
        y_label="sink token local id (score)",
        token_labels=token_labels,
    )


@visualization
def plot_sink_value_vs_visual_self_attention_profile(
    snapshot: PrefillDebugSnapshot,
    out_dir: Path,
    **_,
) -> List[Path]:
    """Plot one reference-style profile: all visual-token attention vs sink value."""
    if int(snapshot.image_token_length) <= 0:
        return []

    sink_values, sink_label, sink_note = _resolve_visual_sink_values(snapshot)
    attention_scores, attention_label = _resolve_plot_attention(snapshot)

    n = min(int(snapshot.image_token_length), int(sink_values.numel()), int(attention_scores.numel()))
    if n <= 0:
        return []

    sink_values = sink_values[:n]
    attention_scores = attention_scores[:n]
    xs = np.arange(n)
    highlight_order = _sorted_sink_order(snapshot)
    highlight_pairs = [
        (int(snapshot.sink_local_ids[idx].item()), idx)
        for idx in highlight_order
        if 0 <= int(snapshot.sink_local_ids[idx].item()) < n
    ]

    fig, ax = plt.subplots(figsize=(15, 5.2), dpi=220)
    ax.plot(xs, attention_scores.numpy(), color="#1f77b4", linewidth=1.9, label=attention_label, zorder=2)
    ax.plot(xs, sink_values.numpy(), color="#d62728", linewidth=1.7, label=sink_label, zorder=3)

    if highlight_pairs:
        highlight_local_ids = [local_id for local_id, _ in highlight_pairs]
        ax.scatter(
            highlight_local_ids,
            attention_scores[highlight_local_ids].numpy(),
            color="#111111",
            s=42,
            marker="o",
            label="Selected sink tokens",
            zorder=5,
        )
        for local_id, sink_idx in highlight_pairs[:10]:
            ax.annotate(
                f"idx={local_id}\nscore={float(snapshot.sink_scores[sink_idx].item()):.2f}",
                xy=(local_id, float(attention_scores[local_id].item())),
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

    if sink_note:
        ax.text(
            0.01,
            0.98,
            sink_note,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.92, "edgecolor": "#dddddd"},
        )

    ax.set_title(f"Layer {_sink_plot_layer(snapshot)}: Visual self-attention and sink value per token", fontsize=16)
    ax.set_xlabel("visual token local id", fontsize=13)
    ax.set_ylabel("value", fontsize=13)
    ax.set_xlim(0, max(n - 1, 0))
    ax.grid(True, linestyle=":", linewidth=0.65, alpha=0.45)
    ax.legend(loc="best", fontsize=11, framealpha=0.95)

    out_path = subdir(out_dir, SUBDIR) / "sink_value_vs_visual_self_attention_profile.png"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return [out_path]


@visualization
def plot_sink_token_hidden_state_details(snapshot: PrefillDebugSnapshot, out_dir: Path, **_) -> List[Path]:
    """Render one per-token hidden-dimension activation plot for each selected sink token."""
    if snapshot.sink_scores.numel() == 0:
        return []

    layer = _sink_plot_layer(snapshot)
    output_dir = subdir(out_dir, SUBDIR) / "sink_hidden_state_details"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []

    for sink_idx in _sorted_sink_order(snapshot):
        local_id = int(snapshot.sink_local_ids[sink_idx].item())
        abs_id = int(snapshot.sink_abs_ids[sink_idx].item())
        hidden_vector = snapshot.hidden_states_at_prune_layer[abs_id].detach().float()
        phi = _phi_profile(hidden_vector)
        top_dims = torch.argsort(torch.abs(hidden_vector), descending=True)[:2].tolist()
        xs = np.arange(int(hidden_vector.numel()))

        fig, (ax_hidden, ax_phi) = plt.subplots(
            2,
            1,
            figsize=(16, 7.2),
            dpi=220,
            sharex=True,
            gridspec_kw={"height_ratios": [2.3, 1.2]},
        )

        ax_hidden.plot(xs, hidden_vector.numpy(), color="#4568dc", linewidth=1.0, alpha=0.95)
        ax_hidden.axhline(0.0, color="#888888", linewidth=0.8, alpha=0.75)
        for rank, dim in enumerate(top_dims):
            color = "#d62728" if rank == 0 else "#ff9d3a"
            value = float(hidden_vector[dim].item())
            ax_hidden.scatter([dim], [value], color=color, s=48, zorder=4)
            ax_hidden.annotate(
                f"dim {dim}",
                xy=(dim, value),
                xytext=(5, 7 if value >= 0 else -16),
                textcoords="offset points",
                fontsize=9,
                color=color,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.9),
            )

        ax_hidden.set_ylabel("hidden state value", fontsize=12)
        ax_hidden.grid(True, linestyle=":", linewidth=0.55, alpha=0.45)
        ax_hidden.set_title(
            f"Layer {layer}: sink token {local_id} hidden-state activations (score={float(snapshot.sink_scores[sink_idx].item()):.3f})",
            fontsize=15,
        )

        ax_phi.plot(xs, phi.numpy(), color="#2ca02c", linewidth=1.0, alpha=0.95, label="phi profile")
        for rank, dim in enumerate(top_dims):
            color = "#d62728" if rank == 0 else "#ff9d3a"
            ax_phi.scatter([dim], [float(phi[dim].item())], color=color, s=40, zorder=4)
        ax_phi.set_xlabel("hidden dimension", fontsize=12)
        ax_phi.set_ylabel("phi", fontsize=12)
        ax_phi.grid(True, linestyle=":", linewidth=0.55, alpha=0.45)
        ax_phi.legend(loc="upper right", fontsize=10, framealpha=0.95)

        info_lines = [
            f"local id: {local_id}",
            f"absolute prompt id: {abs_id}",
            f"top-1 |activation| dim: {top_dims[0]}" if top_dims else "top-1 |activation| dim: n/a",
            f"top-2 |activation| dim: {top_dims[1]}" if len(top_dims) > 1 else "top-2 |activation| dim: n/a",
        ]
        ax_hidden.text(
            0.995,
            0.98,
            "\n".join(info_lines),
            transform=ax_hidden.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.92, "edgecolor": "#dddddd"},
        )

        out_path = output_dir / f"layer_{layer:02d}_sink_token_{local_id}_hidden_state.png"
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.22)
        plt.close(fig)
        paths.append(out_path)

    return paths


@visualization
def plot_sink_token_positions_on_image(snapshot: PrefillDebugSnapshot, out_dir: Path, **_) -> List[Path]:
    """Overlay selected sink tokens on the image patch grid with score labels."""
    if snapshot.sink_scores.numel() == 0:
        return []

    grid_size = _visual_grid_size(int(snapshot.image_token_length))
    if grid_size is None:
        return []

    side_px = grid_size * 28
    image_array = _load_square_display_image(snapshot.image_path, side_px=side_px)
    height, width = image_array.shape[:2]
    cell_w = width / grid_size
    cell_h = height / grid_size
    layer = _sink_plot_layer(snapshot)

    fig, ax = plt.subplots(figsize=(10, 10), dpi=220)
    ax.imshow(image_array)
    _draw_patch_grid(ax, grid_size=grid_size, width=width, height=height)

    order = _sorted_sink_order(snapshot)
    colors = plt.cm.plasma(np.linspace(0.15, 0.92, max(len(order), 1)))
    for color, sink_idx in zip(colors, order):
        local_id = int(snapshot.sink_local_ids[sink_idx].item())
        row, col = divmod(local_id, grid_size)
        x0 = col * cell_w
        y0 = row * cell_h
        rect = Rectangle((x0, y0), cell_w, cell_h, linewidth=2.2, edgecolor=color, facecolor=(*color[:3], 0.18))
        ax.add_patch(rect)
        ax.text(
            x0 + cell_w * 0.06,
            y0 + cell_h * 0.14,
            f"{local_id}\n{float(snapshot.sink_scores[sink_idx].item()):.2f}",
            color="white",
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="top",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": (0.05, 0.05, 0.05, 0.72), "edgecolor": "none"},
        )

    ax.set_title(f"Layer {layer}: sink token positions on the {grid_size}x{grid_size} visual grid", fontsize=16)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)

    out_path = subdir(out_dir, SUBDIR) / "sink_token_positions_on_image.png"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return [out_path]


@visualization
def plot_sink_value_vs_visual_self_attention_cloud(
    snapshot: PrefillDebugSnapshot,
    out_dir: Path,
    **_,
) -> List[Path]:
    """Plot the reference-style sink-value/attention scatter cloud for one snapshot."""
    if int(snapshot.image_token_length) <= 0:
        return []

    sink_values, sink_label, sink_note = _resolve_visual_sink_values(snapshot)
    attention_scores, attention_label = _resolve_plot_attention(snapshot)

    n = min(int(snapshot.image_token_length), int(sink_values.numel()), int(attention_scores.numel()))
    if n <= 0:
        return []

    sink_values = sink_values[:n]
    attention_scores = torch.clamp(attention_scores[:n], min=1e-12)
    highlight_order = _sorted_sink_order(snapshot)
    highlight_local_ids = [
        int(snapshot.sink_local_ids[idx].item())
        for idx in highlight_order
        if 0 <= int(snapshot.sink_local_ids[idx].item()) < n
    ]

    fig, ax = plt.subplots(figsize=(7.6, 5.8), dpi=220)
    ax.scatter(
        sink_values.numpy(),
        attention_scores.numpy(),
        s=10,
        alpha=0.22,
        c="#a64521",
        edgecolors="none",
        label=f"All visual tokens ({attention_label.lower()})",
    )

    if highlight_local_ids:
        ax.scatter(
            sink_values[highlight_local_ids].numpy(),
            attention_scores[highlight_local_ids].numpy(),
            s=42,
            c="black",
            marker="o",
            label="Selected sink tokens",
            zorder=3,
        )
        for local_id in highlight_local_ids[:10]:
            ax.annotate(
                f"idx={local_id}",
                xy=(float(sink_values[local_id].item()), float(attention_scores[local_id].item())),
                xytext=(5, 6),
                textcoords="offset points",
                fontsize=8,
                color="black",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.85),
            )

    if sink_note:
        ax.text(
            0.01,
            0.99,
            sink_note,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.5,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.92, "edgecolor": "#dddddd"},
        )

    ax.set_yscale("log")
    ax.set_xlabel(sink_label.lower(), fontsize=12)
    ax.set_ylabel(attention_label.lower(), fontsize=12)
    ax.set_title(f"Layer {_sink_plot_layer(snapshot)}: Sink value vs visual self-attention", fontsize=15)
    ax.grid(True, linestyle=":", linewidth=0.55, alpha=0.4)
    ax.legend(loc="best", fontsize=10, framealpha=0.95)

    out_path = subdir(out_dir, SUBDIR) / "sink_value_vs_visual_self_attention_cloud.png"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return [out_path]
