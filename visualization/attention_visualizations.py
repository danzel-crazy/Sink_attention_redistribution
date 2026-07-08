"""Visual-attention visualizations.

Every plot here concerns text/generated -> visual-token attention and writes into the
`attention/` subfolder of the run's output directory. Nothing executes on import beyond
registering the tagged plots, so importing this file has no side effects and it is never
imported by the benchmark entrypoints.

Each function takes `(snapshot: PrefillDebugSnapshot, out_dir: Path, **_) -> List[Path]`
and is driven by `visualization/render_fastv_visualizations.py`.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from PIL import Image, ImageOps

from visualization.fastv_snapshot import PrefillDebugSnapshot
from visualization.viz_registry import subdir, visualization

# Every plot in this module lands here, under the run's output directory.
SUBDIR = "attention"


def _resolve_text_span(snapshot: PrefillDebugSnapshot) -> Tuple[int, int]:
    start = int(getattr(snapshot, "text_tokens_start_index", 0) or 0)
    length = int(getattr(snapshot, "text_tokens_length", 0) or 0)
    if length <= 0:
        length = int(snapshot.image_token_start_index)
    length = max(0, min(length, int(snapshot.prompt_length) - start))
    return start, length


def _compute_text_to_visual_attention(
    snapshot: PrefillDebugSnapshot,
    *,
    sink_masked: bool,
) -> torch.Tensor:
    text_start, text_len = _resolve_text_span(snapshot)
    image_start = int(snapshot.image_token_start_index)
    image_len = int(snapshot.image_token_length)

    if text_len <= 0 or image_len <= 0:
        return torch.empty(0, image_len, dtype=torch.float32)

    hidden_states = snapshot.hidden_states_at_prune_layer.detach().float()
    text_tokens = hidden_states[text_start:text_start + text_len]
    visual_tokens = hidden_states[image_start:image_start + image_len]
    scale = visual_tokens.shape[-1] ** 0.5
    scores = torch.matmul(text_tokens, visual_tokens.transpose(0, 1)) / scale

    if sink_masked and snapshot.sink_local_ids.numel() > 0:
        scores = scores.clone()
        scores[:, snapshot.sink_local_ids.long()] = float("-inf")
    return torch.softmax(scores, dim=-1)


def _token_tensor_is_present(value: Optional[torch.Tensor]) -> bool:
    return value is not None and value.numel() > 0


def _clean_token_label(token: str) -> str:
    token = token.replace("Ġ", " ").replace("▁", " ").replace("Ċ", "\\n")
    return token.strip() or "<ws>"


def _summarize_tokens(tokens: Sequence[str], max_tokens: int = 8) -> str:
    cleaned = [_clean_token_label(token) for token in tokens if token is not None]
    if not cleaned:
        return "none"
    if len(cleaned) <= max_tokens:
        return ", ".join(cleaned)
    head = ", ".join(cleaned[:max_tokens])
    return f"{head}, ... ({len(cleaned)} total)"


def _visual_grid_size(image_token_length: int) -> Optional[int]:
    grid = int(round(math.sqrt(image_token_length)))
    return grid if grid * grid == image_token_length else None


def _load_square_display_image(image_path: str, side_px: int) -> np.ndarray:
    resampling = getattr(Image, "Resampling", Image)
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        return np.full((side_px, side_px, 3), 255, dtype=np.uint8)
    square = ImageOps.fit(image, (side_px, side_px), method=resampling.BICUBIC)
    return np.asarray(square)


def _draw_patch_grid(ax, grid_size: int, width: int, height: int) -> None:
    xs = np.linspace(0, width, grid_size + 1)
    ys = np.linspace(0, height, grid_size + 1)
    for x in xs:
        ax.axvline(x=x, color="white", linewidth=0.35, alpha=0.35)
    for y in ys:
        ax.axhline(y=y, color="white", linewidth=0.35, alpha=0.35)


def _plot_attention_overlay(
    ax,
    image_array: np.ndarray,
    attention_vector: torch.Tensor,
    title: str,
    grid_size: int,
    vmin: float,
    vmax: float,
) -> None:
    attention_map = attention_vector.detach().float().reshape(grid_size, grid_size).numpy()
    height, width = image_array.shape[:2]
    ax.imshow(image_array)
    ax.imshow(
        attention_map,
        cmap="inferno",
        alpha=0.48,
        interpolation="nearest",
        extent=(0, width, height, 0),
        vmin=vmin,
        vmax=vmax,
    )
    _draw_patch_grid(ax, grid_size=grid_size, width=width, height=height)
    ax.set_title(title, fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)


@visualization
def plot_visual_token_attention_profiles(snapshot: PrefillDebugSnapshot, out_dir: Path, **_) -> List[Path]:
    image_len = int(snapshot.image_token_length)
    xs = np.arange(image_len)
    text_raw = _compute_text_to_visual_attention(snapshot, sink_masked=False)
    text_masked = _compute_text_to_visual_attention(snapshot, sink_masked=True)
    generated = snapshot.generated_query_visual_attentions
    has_generated = _token_tensor_is_present(generated)

    fig, ax = plt.subplots(figsize=(22, 8), dpi=220)

    if text_raw.numel() > 0:
        for row in text_raw.numpy():
            ax.plot(xs, row, color="#7fb3ff", linewidth=0.7, alpha=0.12, zorder=1)
        ax.plot(
            xs,
            text_raw.mean(dim=0).numpy(),
            color="#1f77b4",
            linewidth=2.6,
            label=f"text -> visual raw mean ({text_raw.shape[0]} tokens)",
            zorder=3,
        )
        ax.plot(
            xs,
            text_masked.mean(dim=0).numpy(),
            color="#2ca02c",
            linewidth=2.4,
            linestyle="--",
            label=f"text -> visual sink-masked mean ({text_masked.shape[0]} tokens)",
            zorder=4,
        )

    if has_generated:
        generated = generated.detach().float()
        for row in generated.numpy():
            ax.plot(xs, row, color="#f4a261", linewidth=0.7, alpha=0.18, zorder=1)
        ax.plot(
            xs,
            generated.mean(dim=0).numpy(),
            color="#d62728",
            linewidth=2.5,
            label=f"generated -> visual mean ({generated.shape[0]} query tokens)",
            zorder=5,
        )
        last_label = "last generated query"
        if getattr(snapshot, "generated_query_token_strings", None):
            last_label = f"generated -> visual last query [{_clean_token_label(snapshot.generated_query_token_strings[-1])}]"
        ax.plot(
            xs,
            generated[-1].numpy(),
            color="#ff9d3a",
            linewidth=1.8,
            label=last_label,
            zorder=6,
        )

    if snapshot.sink_local_ids.numel() > 0:
        ax.scatter(
            snapshot.sink_local_ids.numpy(),
            np.zeros(snapshot.sink_local_ids.numel()),
            color="#111111",
            marker="|",
            s=120,
            alpha=0.65,
            label="sink token local ids",
            zorder=7,
        )

    text_start, text_len = _resolve_text_span(snapshot)
    prompt_tokens = snapshot.prompt_token_strings[text_start:text_start + text_len] if snapshot.prompt_token_strings else []
    info_lines = [
        f"text span [{text_start}:{text_start + text_len}]: {_summarize_tokens(prompt_tokens)}",
        f"generated query tokens: {_summarize_tokens(snapshot.generated_query_token_strings)}",
    ]
    ax.text(
        0.01,
        0.99,
        "\n".join(info_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9, "edgecolor": "#dddddd"},
    )

    ax.set_title(f"Visual-token attention profiles (question_id={snapshot.question_id})", fontsize=16)
    ax.set_xlabel("visual token local id", fontsize=13)
    ax.set_ylabel("attention score", fontsize=13)
    ax.set_xlim(0, max(image_len - 1, 0))
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)
    ax.legend(loc="upper right", fontsize=11, framealpha=0.95)

    out_path = subdir(out_dir, SUBDIR) / "visual_token_attention_profiles.png"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return [out_path]


@visualization
def plot_visual_attention_heatmaps_on_image(snapshot: PrefillDebugSnapshot, out_dir: Path, **_) -> List[Path]:
    grid_size = _visual_grid_size(int(snapshot.image_token_length))
    if grid_size is None:
        return []

    text_raw = _compute_text_to_visual_attention(snapshot, sink_masked=False)
    text_masked = _compute_text_to_visual_attention(snapshot, sink_masked=True)
    generated = snapshot.generated_query_visual_attentions
    has_generated = _token_tensor_is_present(generated)

    if text_masked.numel() == 0 and not has_generated:
        return []

    maps: List[Tuple[str, torch.Tensor]] = []
    if text_masked.numel() > 0:
        maps.append(("text -> visual (sink-masked mean)", text_masked.mean(dim=0)))
    if has_generated:
        maps.append(("generated -> visual (mean query)", generated.detach().float().mean(dim=0)))
    elif text_raw.numel() > 0:
        maps.append(("text -> visual (raw mean)", text_raw.mean(dim=0)))

    side_px = grid_size * 24
    image_array = _load_square_display_image(snapshot.image_path, side_px=side_px)
    vmax = max(float(attn.max().item()) for _, attn in maps)
    vmin = min(float(attn.min().item()) for _, attn in maps)

    fig, axes = plt.subplots(1, len(maps), figsize=(8 * len(maps), 8), dpi=220)
    if len(maps) == 1:
        axes = [axes]

    for ax, (title, attention_vector) in zip(axes, maps):
        _plot_attention_overlay(
            ax,
            image_array=image_array,
            attention_vector=attention_vector,
            title=title,
            grid_size=grid_size,
            vmin=vmin,
            vmax=vmax,
        )

    fig.suptitle(
        f"Visual attention heatmaps on {grid_size}x{grid_size} patches (question_id={snapshot.question_id})",
        fontsize=16,
        y=0.96,
    )
    heatmap = plt.cm.ScalarMappable(cmap="inferno", norm=Normalize(vmin=vmin, vmax=vmax))
    cbar = fig.colorbar(heatmap, ax=axes, fraction=0.022, pad=0.02)
    cbar.set_label("attention score", fontsize=12)

    out_path = subdir(out_dir, SUBDIR) / "visual_attention_heatmaps_on_image.png"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return [out_path]
