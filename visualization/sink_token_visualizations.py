"""Sink-token visualizations.

Every plot here concerns the sink tokens' hidden states and writes into the
`sink_tokens/` subfolder of the run's output directory. Nothing executes on import
beyond registering the tagged plots, so importing this file has no side effects and it
is never imported by the benchmark entrypoints.

Each function takes `(snapshot: PrefillDebugSnapshot, out_dir: Path, **_) -> List[Path]`
and is driven by `visualization/render_fastv_visualizations.py`.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - registers the 3d projection

from visualization.fastv_snapshot import PrefillDebugSnapshot
from visualization.viz_registry import subdir, visualization

# Every plot in this module lands here, under the run's output directory.
SUBDIR = "sink_tokens"


def _phi_profile(hidden_vector: torch.Tensor) -> torch.Tensor:
    # Same RMS-normalized-abs profile sink_token_selector uses over `sink_dims`
    # (cross_attention_sink_redistribution/sink_tokens.py::_compute_hidden_rms_max_sink_values),
    # just evaluated over every hidden dimension instead of only the configured sink dims.
    hidden_vector = hidden_vector.detach().float()
    rms = torch.sqrt(torch.mean(hidden_vector ** 2) + 1e-6)
    return torch.abs(hidden_vector) / rms


def _plot_sink_token_hidden_states(
    snapshot: PrefillDebugSnapshot,
    out_dir: Path,
    order: List[int],
    out_name: str,
    y_label: str,
    token_labels: List[str],
) -> List[Path]:
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
