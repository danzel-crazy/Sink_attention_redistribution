import json
import os
from typing import Optional, Sequence, Tuple, List

import matplotlib.pyplot as plt
import torch

from attention_experiment import get_image_attention, get_text_attentinon
from highlight_tokens import plot_token_positions_on_image
from visualize import plot_attention_dimensions


DEFAULT_SINK_SCORE_METHOD = "hidden_rms_max"
QUERY_KEY_SINK_RATIO_METHOD = "question_key_sink_ratio"
SINK_SCORE_METHODS = (
    DEFAULT_SINK_SCORE_METHOD,
    QUERY_KEY_SINK_RATIO_METHOD,
)


def append_json_record(path: str, record: dict) -> None:
    records = []
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "r") as f:
            try:
                loaded = json.load(f)
                records = loaded if isinstance(loaded, list) else [loaded]
            except json.JSONDecodeError:
                records = []
    records.append(record)
    with open(path, "w") as f:
        json.dump(records, f, indent=4)


def flatten_indices(indices, device=None) -> torch.Tensor:
    if indices is None:
        return torch.empty(0, dtype=torch.long, device=device)
    if not torch.is_tensor(indices):
        indices = torch.tensor(indices, dtype=torch.long, device=device)
    else:
        indices = indices.to(device=device, dtype=torch.long)
    return indices.flatten()


def resolve_image_indices(img_attn: torch.Tensor, fastv_info=None) -> torch.Tensor:
    device = img_attn.device
    img_attn_len = int(img_attn.numel())

    fastv_indices = flatten_indices(
        fastv_info.get("selected_indices") if fastv_info is not None else None,
        device=device,
    )
    if fastv_indices.numel() == img_attn_len:
        return fastv_indices

    total_patches = None
    if fastv_info is not None and fastv_info.get("image_token_length") is not None:
        total_patches = int(fastv_info["image_token_length"])

    if total_patches == img_attn_len:
        return torch.arange(img_attn_len, device=device, dtype=torch.long)

    if fastv_indices.numel() > 0:
        return fastv_indices
    return torch.arange(img_attn_len, device=device, dtype=torch.long)


def resolve_sink_score_method(sink_score_method: Optional[str]) -> str:
    method = str(sink_score_method or DEFAULT_SINK_SCORE_METHOD).strip().lower()
    if method not in SINK_SCORE_METHODS:
        raise ValueError(
            f"Unsupported sink score method {sink_score_method!r}; "
            f"expected one of {', '.join(SINK_SCORE_METHODS)}."
        )
    return method


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


def _compute_question_key_sink_ratio_values(
    *,
    query_state: torch.Tensor,
    key_states: torch.Tensor,
    valid_dims: Sequence[int],
) -> torch.Tensor:
    if query_state is None or key_states is None:
        raise ValueError(
            "question_key_sink_ratio requires both query_state and key_states."
        )
    if query_state.dim() != 1:
        raise ValueError(
            f"query_state must be 1-D [hidden_dim], got {tuple(query_state.shape)}."
        )
    if key_states.dim() != 2:
        raise ValueError(
            f"key_states must be 2-D [tokens, hidden_dim], got {tuple(key_states.shape)}."
        )
    if key_states.shape[1] != query_state.shape[0]:
        raise ValueError(
            "query_state and key_states must share the same hidden dimension, got "
            f"{query_state.shape[0]} and {key_states.shape[1]}."
        )

    query_state = query_state.detach().float()
    key_states = key_states.detach().float()

    sink_logits = torch.sum(
        key_states[:, valid_dims] * query_state[valid_dims].unsqueeze(0),
        dim=1,
    )
    full_logits = torch.matmul(key_states, query_state)
    denom = torch.abs(full_logits).clamp_min(1e-6)
    return sink_logits / denom


def compute_sink_values(
    hidden_states: torch.Tensor,
    target_dim: Sequence[int],
    *,
    sink_score_method: Optional[str] = None,
    query_state: Optional[torch.Tensor] = None,
    key_states: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute a sink score for each token.

    Supported methods:
      - hidden_rms_max:
          score_i = max_{d in target_dim} | x_i[d] / RMS(x_i) |
      - question_key_sink_ratio:
          SinkLogit_i = sum_{d in target_dim} q_d * k_i,d
          SinkRatio_i = SinkLogit_i / abs(q · k_i)

    Args:
        hidden_states: Tensor of shape [num_tokens, hidden_dim].
            Hidden states for all tokens. For the default method this is the
            tensor being scored. For query-conditioned methods it is used only
            for shape validation and backwards-compatible call signatures.
        target_dim: Iterable of int.
            Hidden dimensions regarded as sink-related dimensions.
        sink_score_method: Sink-score method name. Defaults to hidden_rms_max.
        query_state: Query vector used by query-conditioned methods.
        key_states: Per-token key vectors used by query-conditioned methods.

    Returns:
        Tensor of shape [num_tokens].
            One sink score per token.
    """

    if hidden_states.dim() != 2:
        raise ValueError(
            f"hidden_states must be 2-D [tokens, hidden_dim], got {tuple(hidden_states.shape)}"
        )

    sink_score_method = resolve_sink_score_method(sink_score_method)
    hidden_dim = hidden_states.shape[1]

    valid_dims = [int(d) for d in target_dim if 0 <= int(d) < hidden_dim]
    if not valid_dims:
        raise ValueError(
            f"No valid target dimensions in {list(target_dim)} for hidden_dim={hidden_dim}."
        )

    if sink_score_method == DEFAULT_SINK_SCORE_METHOD:
        return _compute_hidden_rms_max_sink_values(hidden_states, valid_dims)

    if sink_score_method == QUERY_KEY_SINK_RATIO_METHOD:
        return _compute_question_key_sink_ratio_values(
            query_state=query_state,
            key_states=key_states,
            valid_dims=valid_dims,
        )

    raise ValueError(f"Unhandled sink score method: {sink_score_method}")


def plot_sink_vs_attention(
    layer: int,
    attention_scores: torch.Tensor,
    sink_values: torch.Tensor,
    highlight_indices: list[int],
    output_dir: str,
    highlight_label: str = "Selected sink tokens",
) -> str:
    os.makedirs(output_dir, exist_ok=True)

    attn = attention_scores.detach().float().cpu()
    sink = sink_values.detach().float().cpu()
    x = torch.arange(attn.numel()).numpy()

    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=120)
    ax.plot(x, attn.numpy(), color="#1f77b4", linewidth=1.8, label="Attention score")
    ax.plot(x, sink.numpy(), color="#d62728", linewidth=1.6, label="Sink value")

    if highlight_indices:
        topk_tensor = torch.tensor(highlight_indices, dtype=torch.long)
        ax.scatter(
            topk_tensor.numpy(),
            attn[topk_tensor].numpy(),
            color="black",
            s=40,
            marker="o",
            label=highlight_label,
            zorder=3,
        )
        for idx in highlight_indices:
            idx_i = int(idx)
            if idx_i < 0 or idx_i >= attn.numel() or idx_i >= sink.numel():
                continue
            ax.annotate(
                f"idx={idx_i}\nsink={sink[idx_i].item():.2f}",
                xy=(idx_i, attn[idx_i].item()),
                xytext=(6, 8),
                textcoords="offset points",
                fontsize=8,
                color="black",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.85),
            )
            ax.scatter(
                [idx_i],
                [sink[idx_i].item()],
                color="#d62728",
                s=30,
                marker="x",
                zorder=4,
            )

    ax.set_title(f"Layer {layer}: Attention and Sink Value per Token")
    ax.set_xlabel("Image token index (in current layer view)")
    ax.set_ylabel("Value")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()

    output_path = os.path.join(output_dir, f"layer_{layer:02d}_sink_attention.png")
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _load_all_points_from_records(path: str) -> Tuple[List[float], List[float]]:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return [], []
    try:
        with open(path, "r") as f:
            records = json.load(f)
    except (json.JSONDecodeError, OSError):
        return [], []

    if not isinstance(records, list):
        records = [records]

    all_sink = []
    all_attn = []
    for rec in records:
        sink_vals = rec.get("sink_values", [])
        attn_vals = rec.get("attention_scores", [])
        if not isinstance(sink_vals, list) or not isinstance(attn_vals, list):
            continue
        n = min(len(sink_vals), len(attn_vals))
        if n <= 0:
            continue
        all_sink.extend(float(v) for v in sink_vals[:n])
        all_attn.extend(float(v) for v in attn_vals[:n])
    return all_sink, all_attn


def plot_sink_attention_cloud(
    layer: int,
    sink_values: torch.Tensor,
    attention_scores: torch.Tensor,
    highlight_indices: List[int],
    output_dir: str,
    global_sink: Optional[List[float]] = None,
    global_attn: Optional[List[float]] = None,
    highlight_label: str = "Selected sink tokens",
) -> Tuple[str, Optional[str]]:
    os.makedirs(output_dir, exist_ok=True)

    sink = sink_values.detach().float().cpu()
    attn = attention_scores.detach().float().cpu()
    eps = 1e-12
    attn = torch.clamp(attn, min=eps)

    # Per-layer cloud
    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=160)
    ax.scatter(
        sink.numpy(),
        attn.numpy(),
        s=8,
        alpha=0.20,
        c="#a64521",
        edgecolors="none",
    )
    if highlight_indices:
        valid_topk = [int(i) for i in highlight_indices if 0 <= int(i) < sink.numel() and int(i) < attn.numel()]
        if valid_topk:
            topk_t = torch.tensor(valid_topk, dtype=torch.long)
            ax.scatter(
                sink[topk_t].numpy(),
                attn[topk_t].numpy(),
                s=38,
                c="black",
                marker="o",
                label=highlight_label,
                zorder=3,
            )
            for idx_i in valid_topk:
                ax.annotate(
                    f"idx={idx_i}",
                    xy=(sink[idx_i].item(), attn[idx_i].item()),
                    xytext=(5, 6),
                    textcoords="offset points",
                    fontsize=8,
                    color="black",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.85),
                )
    ax.set_yscale("log")
    ax.set_xlabel("Sink dimension value")
    ax.set_ylabel("Attention weights")
    ax.set_title(f"Layer {layer}: Sink Value vs Attention")
    ax.grid(alpha=0.15)
    if highlight_indices:
        ax.legend(loc="best")
    fig.tight_layout()
    layer_path = os.path.join(output_dir, f"layer_{layer:02d}_sink_attention_cloud.png")
    fig.savefig(layer_path)
    plt.close(fig)

    global_path = None
    if global_sink and global_attn:
        g_sink = torch.tensor(global_sink, dtype=torch.float32)
        g_attn = torch.clamp(torch.tensor(global_attn, dtype=torch.float32), min=eps)

        fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=160)
        ax.scatter(
            g_sink.numpy(),
            g_attn.numpy(),
            s=5,
            alpha=0.10,
            c="#7f78b2",
            edgecolors="none",
        )
        ax.set_yscale("log")
        ax.set_xlabel("Sink dimension value")
        ax.set_ylabel("Attention weights")
        ax.set_title("All Layers: Sink Value vs Attention")
        ax.grid(alpha=0.15)
        fig.tight_layout()
        global_path = os.path.join(output_dir, "all_layers_sink_attention_cloud.png")
        fig.savefig(global_path)
        plt.close(fig)

    return layer_path, global_path


def resolve_sink_score_range(
    sink_values: torch.Tensor,
    args,
) -> Tuple[float, float, str]:
    sink_values = sink_values.detach().float().flatten()
    if sink_values.numel() == 0:
        raise ValueError("Cannot resolve sink-score range from an empty tensor.")

    score_min = getattr(args, "sink_score_min", None)
    score_max = getattr(args, "sink_score_max", None)

    if score_min is None:
        quantile = float(getattr(args, "sink_score_quantile", 0.95))
        quantile = min(max(quantile, 0.0), 1.0)
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


def select_sink_tokens_by_range(
    sink_values: torch.Tensor,
    score_min: float,
    score_max: float,
) -> torch.Tensor:
    sink_values = sink_values.detach().float().flatten()
    selected = torch.nonzero(
        (sink_values >= score_min) & (sink_values <= score_max),
        as_tuple=False,
    ).flatten()
    if selected.numel() == 0:
        return selected

    selected_scores = sink_values[selected]
    order = torch.argsort(selected_scores, descending=True)
    return selected[order]


def sink_value_fastv(j, fastv_info, attention, hidden_states, original_image, spans, args):
    layer = int(j)
    target_layer = int(args.fastv_k) - 1
    if layer != target_layer:
        return

    if spans is None:
        raise ValueError("Missing spans for attention/image alignment.")
    if attention is None:
        raise ValueError("Missing attention matrix.")

    if not torch.is_tensor(attention):
        attention = torch.tensor(attention)
    attention = attention.float()

    if args.exp == 1:
        img_attn, img_hidden_states = get_image_attention(attention, hidden_states, spans, layer, args.fastv_k)
    elif args.exp == 2:
        img_attn, img_hidden_states = get_text_attentinon(attention, hidden_states, spans, layer, args.fastv_k)
    else:
        raise ValueError(f"Unsupported exp: {args.exp}")

    img_attn = img_attn.detach().float().flatten()
    if img_hidden_states.dim() == 3 and img_hidden_states.shape[0] == 1:
        img_hidden_states = img_hidden_states.squeeze(0)
    img_hidden_states = img_hidden_states.detach().float()

    if img_hidden_states.shape[0] != img_attn.numel():
        n = min(int(img_hidden_states.shape[0]), int(img_attn.numel()))
        img_hidden_states = img_hidden_states[:n]
        img_attn = img_attn[:n]

    sink_values = compute_sink_values(
        img_hidden_states,
        target_dim=getattr(args, "sink_target_dims", [2533]),
    )

    score_min, score_max, range_source = resolve_sink_score_range(sink_values, args)
    selected_local_indices = select_sink_tokens_by_range(
        sink_values=sink_values,
        score_min=score_min,
        score_max=score_max,
    )
    selected_local_indices_list = selected_local_indices.detach().cpu().tolist()

    selected_indices = resolve_image_indices(img_attn, fastv_info=fastv_info)
    selected_image_indices = [
        int(selected_indices[idx].item())
        for idx in selected_local_indices_list
        if idx < selected_indices.numel()
    ]

    highlighted_plot_indices = selected_local_indices_list[: int(getattr(args, "sink_plot_max_tokens", 10))]

    sink_plot_dir = os.path.join(args.output_dir, "sink_plots")
    plot_path = plot_sink_vs_attention(
        layer=layer,
        attention_scores=img_attn,
        sink_values=sink_values,
        highlight_indices=highlighted_plot_indices,
        output_dir=sink_plot_dir,
        highlight_label="Selected sink tokens",
    )

    if not hasattr(args, "sink_value_path"):
        args.sink_value_path = os.path.join(args.output_dir, "sink_values.json")

    total_patches = int(fastv_info["image_token_length"])
    grid_size = int(total_patches ** 0.5)
    if grid_size * grid_size != total_patches:
        raise ValueError(
            f"Unexpected number of image patches ({total_patches}). Cannot form a square grid."
        )

    hidden_plot_dir = os.path.join(args.output_dir, "sink_hidden_states")
    sink_records = []
    for token_rank, local_idx in enumerate(selected_local_indices_list):
        if local_idx >= img_hidden_states.shape[0]:
            continue
        image_idx = selected_image_indices[token_rank] if token_rank < len(selected_image_indices) else None
        peak_dim = plot_attention_dimensions(
            layer_idx=layer,
            hidden_states=img_hidden_states,
            token_id=local_idx,
            token_rank=token_rank,
            output_path=hidden_plot_dir,
        )
        hidden_plot_path = os.path.join(
            hidden_plot_dir,
            f"layer_{layer}",
            f"dim_act_layer_{layer}_token_{local_idx}.png",
        )
        sink_records.append(
            {
                "token_rank": token_rank,
                "token_index": int(local_idx),
                "image_index": image_idx,
                "sink_score": float(sink_values[local_idx].item()),
                "attention_score": float(img_attn[local_idx].item()),
                "peak_dim": peak_dim,
                "hidden_state_plot_path": hidden_plot_path,
            }
        )

    position_labels = [
        f"{int(image_idx)}:{float(sink_values[local_idx].item()):.2f}"
        for local_idx, image_idx in zip(selected_local_indices_list, selected_image_indices)
    ]
    sink_position_dir = os.path.join(args.output_dir, "sink_positions")
    position_plot_path = plot_token_positions_on_image(
        original_image=original_image,
        token_indices=selected_image_indices,
        grid_size=grid_size,
        layer=layer,
        output_path=sink_position_dir,
        alpha=args.alpha,
        labels=position_labels,
        file_name=f"layer_{layer:02d}_sink_tokens.png",
    )

    append_json_record(
        args.sink_value_path,
        {
            "layer": layer,
            "target_dims": list(getattr(args, "sink_target_dims", [1513, 2533])),
            "sink_score_range": {
                "min": score_min,
                "max": score_max,
                "source": range_source,
            },
            "attention_scores": img_attn.detach().cpu().numpy().tolist(),
            "sink_values": sink_values.detach().cpu().numpy().tolist(),
            "selected_sink_token_count": len(sink_records),
            "selected_sink_tokens": sink_records,
            "plot_path": plot_path,
            "position_plot_path": position_plot_path,
        },
    )

    # Create an additional cloud-style plot similar to the reference figure.
    all_sink, all_attn = _load_all_points_from_records(args.sink_value_path)
    cloud_plot_dir = os.path.join(args.output_dir, "sink_cloud_plots")
    layer_cloud_path, global_cloud_path = plot_sink_attention_cloud(
        layer=layer,
        sink_values=sink_values,
        attention_scores=img_attn,
        highlight_indices=highlighted_plot_indices,
        output_dir=cloud_plot_dir,
        global_sink=all_sink,
        global_attn=all_attn,
        highlight_label="Selected sink tokens",
    )

    print(f"Layer {layer} sink plot saved: {plot_path}")
    print(f"Layer {layer} sink token positions saved: {position_plot_path}")
    print(f"Layer {layer} sink cloud plot saved: {layer_cloud_path}")
    if global_cloud_path is not None:
        print(f"All-layers sink cloud plot saved: {global_cloud_path}")
