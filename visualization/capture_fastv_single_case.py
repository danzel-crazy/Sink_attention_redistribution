"""Run ONE POPE example through the current FastV + sink-redistribution pipeline
(FastV/src/FastV/inference/eval/inference.py) and dump everything needed to analyze
it offline into a single .pt file: hidden states of visual tokens, attention of visual
tokens (self-attention, cross-attention, and post-redistribution), sink scores and sink
token ids.

This file does NOT modify any FastV/cross_attention_sink_redistribution source. It
reuses the real model classes and the real `sink_token_selector` /
`cross_attention_importants` / `sink_attention_redistributor` instances that
LlamaModel.fastv_forward builds for itself, and reads their state back after
`generate()` finishes. The one piece of information those instances don't expose
(the raw self-attention weights for the layer right before pruning, since
`fastv_forward` never appends to `all_self_attns`) is captured with a plain
`nn.Module.register_forward_hook` on that one decoder layer - an external, read-only
attachment, not a code change.

Usage:
    python -m visualization.capture_fastv_single_case \\
        --model-id llava-hf/llava-1.5-7b-hf \\
        --question-file /path/to/llava_pope_test.jsonl \\
        --image-folder /path/to/val2014 \\
        --line-index 0 \\
        --out visualization/snapshots/pope_single_case.pt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from cross_attention_sink_redistribution.attention_redistribution import (
    REDISTRIBUTION_STRATEGIES,
    _STRATEGY_ALIASES,
)

_REDISTRIBUTION_STRATEGY_CHOICES = tuple(REDISTRIBUTION_STRATEGIES) + tuple(_STRATEGY_ALIASES)
from FastV.src.FastV.inference.eval.inference import (
    build_fastv_config,
    build_prompt,
    prepare_inputs,
    resolve_dtype,
    resolve_image_path,
    strip_answer_suffix,
)
from visualization.fastv_snapshot import PrefillDebugSnapshot, save_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", type=str, default="llava-hf/llava-1.5-7b-hf")
    # Pin the same commit the benchmark uses (FastV/src/FastV/inference/eval/inference.py
    # defaults --revision to a272c74). A commit SHA is immutable, so transformers loads it
    # straight from the HF cache; leaving this None loads the `main` branch, which forces a
    # live Hub round-trip every run to resolve/verify the branch (slow, and stalls if the Hub
    # is unreachable) and can trigger a fresh multi-GB download into a cache that only has the
    # pinned snapshot. Also guarantees the visualization reflects the same weights as the run.
    parser.add_argument("--revision", type=str, default="a272c74")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float16", choices=["auto", "float16", "bfloat16", "float32"])

    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-id", type=str, default=None, help="Select the case by question_id.")
    parser.add_argument("--line-index", type=int, default=0, help="Select the case by 0-based line number (used when --question-id is not set).")

    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--min-new-tokens", type=int, default=0)

    parser.add_argument("--out", type=str, default="visualization/snapshots/pope_single_case.pt")

    # fastv_config knobs - same names/defaults as scripts/FastV/cross/pope_hf.sh
    parser.add_argument("--visual_token_num", type=int, default=576)
    parser.add_argument("--fastv_k", type=int, default=5)
    parser.add_argument("--fastv_r", type=float, default=0.77)
    parser.add_argument("--image_token_start_index", type=int, default=5)

    # sink_token_selector (cross_attention_sink_redistribution/sink_tokens.py)
    parser.add_argument("--sink-dims", type=int, nargs="+", default=[2533])
    parser.add_argument("--sink-score-min", type=float, default=None)
    parser.add_argument("--sink-score-max", type=float, default=None)
    parser.add_argument("--sink-score-quantile", type=float, default=0.99)

    # cross_attention_importants (cross_attention_sink_redistribution/cross_attention.py)
    parser.add_argument("--disable-sink-masked", action="store_false", dest="enable_sink_masked")
    parser.add_argument(
        "--text-tokens-length",
        type=int,
        default=None,
        help="Defaults to --image_token_start_index (system-prompt prefix length) if unset.",
    )

    # sink_attention_redistributor (cross_attention_sink_redistribution/attention_redistribution.py)
    parser.add_argument("--redistribution-ratio", type=float, default=1.0)
    parser.add_argument("--redistribution-strategy", type=str, default="topk_text_visual_tokens", choices=_REDISTRIBUTION_STRATEGY_CHOICES)
    # deprecated / ignored by the faithful redistributor (kept so old invocations don't error)
    parser.add_argument("--redistribution-softmax-mode", type=str, default="post_softmax_resoftmax")
    parser.add_argument("--receiver-token-count", type=int, default=32)
    parser.add_argument("--receiver-score-power", type=float, default=1.0)
    parser.add_argument("--receiver-importance-source", type=str, default="pre_visual", choices=("cross", "pre_visual"))
    parser.set_defaults(enable_sink_masked=True)

    args = parser.parse_args()
    args.use_fastv = True  # this script only exists to capture the fastv+sink path
    return args


def load_single_question(question_file: str, question_id: Optional[str], line_index: int) -> Dict[str, Any]:
    with open(os.path.expanduser(question_file), "r") as file_handle:
        questions = [json.loads(line) for line in file_handle]
    if question_id is not None:
        for question in questions:
            if str(question["question_id"]) == str(question_id):
                return question
        raise ValueError(f"question_id {question_id!r} not found in {question_file}")
    return questions[line_index]


def make_prune_layer_hook(state: Dict[str, Any]):
    """Forward hook for decoder layer `fastv_k - 1`. Fires once per generation step
    (fastv_forward forces output_attentions=True for this one layer every time), but
    we only want the prefill call - identified by seq_len > 1, since every decode step
    feeds the model exactly one token.
    """

    def hook(module, inputs, output):
        hidden_states_out = output[0]
        if hidden_states_out.shape[1] <= 1:
            if len(output) > 1 and output[1] is not None:
                decode_attn = output[1][0].detach().to("cpu", torch.float32).clone()
                state.setdefault("decode_query_attentions", []).append(decode_attn.mean(dim=0)[-1])
            return
        if state["done"]:
            return
        state["hidden_states_at_prune_layer"] = hidden_states_out[0].detach().to("cpu", torch.float32).clone()
        if len(output) > 1 and output[1] is not None:
            state["last_layer_attention"] = output[1][0].detach().to("cpu", torch.float32).clone()
        state["done"] = True

    return hook


def make_inputs_embeds_hook(state: Dict[str, Any]):
    """Pre-hook on decoder layer 0: capture the pre-decoder embeddings (its input hidden_states)
    once, at prefill (seq_len > 1). Needed as the visual source for the pre_visual cross-attention."""

    def hook(module, args, kwargs):
        hidden_states = kwargs.get("hidden_states", args[0] if args else None)
        if hidden_states is None or hidden_states.shape[1] <= 1 or "inputs_embeds" in state:
            return
        state["inputs_embeds"] = hidden_states[0].detach().to("cpu", torch.float32).clone()

    return hook


def make_self_attn_qk_hook(state: Dict[str, Any]):
    """Pre-hook on the prune layer's self_attn: capture its (post-layernorm) input hidden_states
    and position_ids at prefill, so we can recompute the post-RoPE Q/K offline for the
    text_to_visual_attention_from_qk variant."""

    def hook(module, args, kwargs):
        hidden_states = kwargs.get("hidden_states", args[0] if args else None)
        if hidden_states is None or hidden_states.shape[1] <= 1 or "self_attn_input" in state:
            return
        position_ids = kwargs.get("position_ids")
        if position_ids is None:
            for a in args:
                if torch.is_tensor(a) and a.dtype == torch.long and a.dim() == 2:
                    position_ids = a
                    break
        state["self_attn_input"] = hidden_states.detach().clone()
        state["self_attn_position_ids"] = position_ids.detach().clone() if position_ids is not None else None

    return hook


def _recompute_prune_layer_qk(attn_module, hidden_states, position_ids):
    """Reproduce the prune layer's post-RoPE query/key states (read-only) from its captured input,
    matching LlamaAttention.forward. Returns (q, k) as [1, num_heads, seq, head_dim] on CPU."""
    bsz, q_len, _ = hidden_states.shape
    query_states = attn_module.q_proj(hidden_states).view(bsz, q_len, attn_module.num_heads, attn_module.head_dim).transpose(1, 2)
    key_states = attn_module.k_proj(hidden_states).view(bsz, q_len, attn_module.num_key_value_heads, attn_module.head_dim).transpose(1, 2)
    if position_ids is None:
        position_ids = torch.arange(q_len, device=hidden_states.device).unsqueeze(0)
    cos, sin = attn_module.rotary_emb(key_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    if attn_module.num_key_value_heads != attn_module.num_heads:
        key_states = repeat_kv(key_states, attn_module.num_heads // attn_module.num_key_value_heads)
    return (
        query_states.detach().to("cpu", torch.float32),
        key_states.detach().to("cpu", torch.float32),
    )


def _visual_restricted_per_head_rows(
    self_attn_weights: torch.Tensor,
    *,
    text_query_positions: torch.Tensor,
    visual_token_start: int,
    visual_token_num: int,
) -> torch.Tensor:
    """Return per-head visual-restricted probabilities [B, H, T, N] from full attention weights."""
    if self_attn_weights.dim() == 3:
        self_attn_weights = self_attn_weights.unsqueeze(0)
    query_positions = text_query_positions.to(device=self_attn_weights.device, dtype=torch.long).flatten()
    visual_start = int(visual_token_start)
    visual_end = visual_start + int(visual_token_num)
    per_head = self_attn_weights.float().index_select(2, query_positions)  # [B, H, T, L]
    per_head_visual = per_head[..., visual_start:visual_end].clone()
    row_sums = per_head_visual.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(per_head_visual.dtype).tiny)
    return per_head_visual / row_sums


def _visual_qk_logits(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    *,
    text_query_positions: torch.Tensor,
    visual_token_start: int,
    visual_token_num: int,
) -> torch.Tensor:
    """Return visual-key logits [B, H, T, N] from post-RoPE query/key states."""
    query_positions = text_query_positions.to(device=query_states.device, dtype=torch.long).flatten()
    text_queries = query_states.float().index_select(2, query_positions)
    visual_start = int(visual_token_start)
    visual_end = visual_start + int(visual_token_num)
    visual_keys = key_states.float()[:, :, visual_start:visual_end, :]
    return torch.matmul(text_queries, visual_keys.transpose(-1, -2)) / math.sqrt(query_states.shape[-1])


def _pre_visual_logits(
    pre_visual_attn,
    hidden_states_at_prune_layer: torch.Tensor,
    inputs_embeds: torch.Tensor,
    *,
    text_tokens_start_index: int,
    text_tokens_length: int,
    visual_token_start: int,
    visual_token_num: int,
) -> torch.Tensor:
    """Return pre_visual logits [B, T, N] using the same V_self computation as the pipeline class."""
    text_tokens = hidden_states_at_prune_layer[text_tokens_start_index:text_tokens_start_index + text_tokens_length]
    visual_tokens = inputs_embeds[visual_token_start:visual_token_start + visual_token_num]
    visual_self = pre_visual_attn._visual_self_attention(visual_tokens)
    logits = torch.matmul(text_tokens, visual_self.transpose(0, 1)) / math.sqrt(visual_self.shape[-1])
    return logits.float().unsqueeze(0)


def resolve_device(args: argparse.Namespace) -> torch.device:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = torch.device("cpu")
    return device


def load_model_and_processor(args: argparse.Namespace, device: torch.device, dtype: torch.dtype, fastv_config: Dict[str, Any]):
    """Load the FastV-configured LLaVA model + processor once so callers can reuse them
    across many questions."""
    revision = args.revision if args.revision else None
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_id,
        revision=revision,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        fastv_config=fastv_config,
    ).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_id, revision=revision)
    return model, processor


@torch.no_grad()
def capture_question(
    args: argparse.Namespace,
    model: LlavaForConditionalGeneration,
    processor: AutoProcessor,
    question: Dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    fastv_config: Dict[str, Any],
) -> PrefillDebugSnapshot:
    """Run one question through the already-loaded model and build its snapshot."""
    cur_prompt = question["text"]
    image_path = resolve_image_path(args.image_folder, question["image"])
    prompt = build_prompt(cur_prompt)
    image = Image.open(image_path).convert("RGB")
    inputs = prepare_inputs(processor, prompt, image, device, dtype)
    prompt_token_ids = inputs["input_ids"][0].detach().to("cpu").clone()
    prompt_token_strings = processor.tokenizer.convert_ids_to_tokens(prompt_token_ids.tolist())

    llama_model = model.language_model.model
    hook_layer_idx = args.fastv_k - 1
    if not (0 <= hook_layer_idx < len(llama_model.layers)):
        raise ValueError(f"fastv_k={args.fastv_k} leaves no valid layer to hook (model has {len(llama_model.layers)} layers).")
    state: Dict[str, Any] = {"done": False}
    handles = [
        llama_model.layers[hook_layer_idx].register_forward_hook(make_prune_layer_hook(state)),
        llama_model.layers[0].register_forward_pre_hook(make_inputs_embeds_hook(state), with_kwargs=True),
        llama_model.layers[hook_layer_idx].self_attn.register_forward_pre_hook(make_self_attn_qk_hook(state), with_kwargs=True),
    ]
    try:
        output = model.generate(
            **inputs,
            min_new_tokens=args.min_new_tokens,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
        )
    finally:
        for handle in handles:
            handle.remove()

    if not state["done"]:
        raise RuntimeError("The prune-layer hook never fired - is fastv_config actually being applied?")
    if "last_layer_attention" not in state:
        raise RuntimeError("Decoder layer did not return attention weights; expected output_attentions to be forced for this layer.")

    decoded = processor.batch_decode(output.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    generated_answer = decoded.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in decoded else decoded.strip()
    generated_token_ids = output.sequences[0, prompt_token_ids.shape[0]:].detach().to("cpu").clone()
    generated_token_strings = processor.tokenizer.convert_ids_to_tokens(generated_token_ids.tolist())

    image_start = args.image_token_start_index
    image_len = args.visual_token_num
    text_start = int(fastv_config["text_tokens_start_index"])
    text_len = int(fastv_config["text_tokens_length"])

    hidden_states_at_prune_layer = state["hidden_states_at_prune_layer"]  # [seq_len, hidden_dim]
    last_layer_attention = state["last_layer_attention"]  # [num_heads, seq_len, seq_len]
    self_attention_last_token = last_layer_attention.mean(dim=0)[-1]  # [seq_len]
    visual_self_attention = self_attention_last_token[image_start:image_start + image_len].clone()

    sink_selector = llama_model.sink_selector
    sink_local_ids = sink_selector.sink_tokens_ids.detach().to("cpu").clone()
    sink_scores = sink_selector.sink_tokens_scores.detach().to("cpu", torch.float32).clone()
    sink_abs_ids = sink_local_ids + image_start

    prune_layer = llama_model.layers[hook_layer_idx]
    prune_layer_input_layernorm_weight = prune_layer.input_layernorm.weight.detach().to("cpu", torch.float32).clone()
    prune_layer_input_layernorm_eps = float(prune_layer.input_layernorm.variance_epsilon)

    # --- the three text->visual cross-attention variants, all keyed on the question tokens
    # (after the visual block), computed with the real pipeline classes so they exactly match
    # what _receiver_cross_scores would produce. Only receiver_importance_source's variant runs
    # during generate, so we (re)compute all three here explicitly. ---
    prompt_length = hidden_states_at_prune_layer.shape[0]
    question_start = image_start + image_len
    question_length = prompt_length - question_start
    question_positions = torch.arange(question_start, prompt_length)

    cross_attn = llama_model.cross_attention_importants
    pre_visual_attn = llama_model.pre_visual_cross_attention_importants

    # text_to_visual (from the real post-softmax attention weights); also fills the existing fields
    cross_attn.compute_cross_attention(
        last_layer_attention.unsqueeze(0), question_positions,
        visual_token_start=image_start, visual_token_num=image_len, sink_local_ids=sink_local_ids,
    )
    visual_cross_text_to_visual = cross_attn.important_tokens_scores_raw.detach().to("cpu", torch.float32).clone()
    visual_cross_text_to_visual_rows = cross_attn.row_probabilities.detach().to("cpu", torch.float32).clone()
    visual_cross_text_to_visual_head_rows = _visual_restricted_per_head_rows(
        last_layer_attention.unsqueeze(0),
        text_query_positions=question_positions,
        visual_token_start=image_start,
        visual_token_num=image_len,
    ).detach().to("cpu", torch.float32).clone()
    visual_cross_attention_raw = visual_cross_text_to_visual.clone()
    visual_cross_attention_masked = cross_attn.important_tokens_scores.detach().to("cpu", torch.float32).clone()

    # text_to_visual_from_qk (from recomputed post-RoPE Q/K); numerically equals the weights path
    visual_cross_from_qk = None
    visual_cross_from_qk_rows = None
    visual_cross_from_qk_logits = None
    if "self_attn_input" in state:
        q_states, k_states = _recompute_prune_layer_qk(
            prune_layer.self_attn, state["self_attn_input"], state.get("self_attn_position_ids")
        )
        visual_cross_from_qk_logits = _visual_qk_logits(
            q_states,
            k_states,
            text_query_positions=question_positions,
            visual_token_start=image_start,
            visual_token_num=image_len,
        ).detach().to("cpu", torch.float32).clone()
        cross_attn.compute_cross_attention_from_qk(
            q_states, k_states, question_positions,
            visual_token_start=image_start, visual_token_num=image_len, sink_local_ids=sink_local_ids,
        )
        visual_cross_from_qk = cross_attn.important_tokens_scores_raw.detach().to("cpu", torch.float32).clone()
        visual_cross_from_qk_rows = cross_attn.row_probabilities.detach().to("cpu", torch.float32).clone()

    # pre_visual (V_self over the pre-decoder image features)
    visual_cross_pre_visual = None
    visual_cross_pre_visual_rows = None
    visual_cross_pre_visual_logits = None
    if "inputs_embeds" in state:
        visual_cross_pre_visual_logits = _pre_visual_logits(
            pre_visual_attn,
            hidden_states_at_prune_layer,
            state["inputs_embeds"],
            text_tokens_start_index=question_start,
            text_tokens_length=question_length,
            visual_token_start=image_start,
            visual_token_num=image_len,
        ).detach().to("cpu", torch.float32).clone()
        pre_visual_attn.compute_cross_attention(
            hidden_states_at_prune_layer.unsqueeze(0), state["inputs_embeds"].unsqueeze(0),
            text_tokens_start_index=question_start, text_tokens_length=question_length, sink_local_ids=sink_local_ids,
        )
        visual_cross_pre_visual = pre_visual_attn.important_tokens_scores_raw.detach().to("cpu", torch.float32).clone()
        visual_cross_pre_visual_rows = pre_visual_attn.row_probabilities.detach().to("cpu", torch.float32).clone()

    last_result = llama_model.sink_attention_redistributor.last_result
    if last_result is None:
        raise RuntimeError("sink_attention_redistributor.redistribute() was never called during this run.")
    # new faithful redistributor already returns the full [N] vector (sinks zeroed, budget added)
    visual_redistributed_attention = last_result.redistributed_scores.detach().to("cpu", torch.float32).clone()
    receiver_indices = last_result.receiver_indices.detach().to("cpu", torch.long).clone()
    receiver_weights = last_result.receiver_weights.detach().to("cpu", torch.float32).clone()
    sink_budget = float(last_result.sink_budget)
    receiver_selection_scores = (
        None
        if last_result.selection_scores is None
        else last_result.selection_scores.detach().to("cpu", torch.float32).clone()
    )
    receiver_weight_scores = (
        None
        if last_result.weight_scores is None
        else last_result.weight_scores.detach().to("cpu", torch.float32).clone()
    )

    keep_count = round(image_len * (1 - args.fastv_r))
    kept_visual_local_ids = visual_redistributed_attention.topk(keep_count).indices.sort().values.clone()
    keep_mask = torch.zeros(image_len, dtype=torch.bool)
    keep_mask[kept_visual_local_ids] = True
    pruned_visual_local_ids = torch.nonzero(~keep_mask, as_tuple=False).squeeze(1)

    decode_query_attentions = state.get("decode_query_attentions", [])
    if decode_query_attentions:
        generated_query_visual_attentions = torch.stack(
            [row[image_start:image_start + image_len].clone() for row in decode_query_attentions],
            dim=0,
        )
        generated_query_count = generated_query_visual_attentions.shape[0]
        generated_query_token_ids = generated_token_ids[:generated_query_count].clone()
        generated_query_token_strings = processor.tokenizer.convert_ids_to_tokens(generated_query_token_ids.tolist())
    else:
        generated_query_visual_attentions = torch.empty(0, image_len, dtype=torch.float32)
        generated_query_token_ids = torch.empty(0, dtype=torch.long)
        generated_query_token_strings = []

    snapshot = PrefillDebugSnapshot(
        model_id=args.model_id,
        question_id=question.get("question_id"),
        image_path=image_path,
        question=cur_prompt,
        prompt=prompt,
        generated_answer=generated_answer,
        fastv_k=args.fastv_k,
        fastv_r=args.fastv_r,
        image_token_start_index=image_start,
        image_token_length=image_len,
        text_tokens_start_index=text_start,
        text_tokens_length=text_len,
        prompt_length=hidden_states_at_prune_layer.shape[0],
        hidden_dim=hidden_states_at_prune_layer.shape[1],
        prompt_token_ids=prompt_token_ids,
        prompt_token_strings=prompt_token_strings,
        generated_token_ids=generated_token_ids,
        generated_token_strings=generated_token_strings,
        generated_query_token_ids=generated_query_token_ids,
        generated_query_token_strings=generated_query_token_strings,
        hidden_states_at_prune_layer=hidden_states_at_prune_layer,
        self_attention_last_token=self_attention_last_token,
        visual_hidden_states=hidden_states_at_prune_layer[image_start:image_start + image_len].clone(),
        visual_self_attention=visual_self_attention,
        visual_cross_attention_raw=visual_cross_attention_raw,
        visual_cross_attention_masked=visual_cross_attention_masked,
        visual_cross_pre_visual=visual_cross_pre_visual,
        visual_cross_text_to_visual=visual_cross_text_to_visual,
        visual_cross_from_qk=visual_cross_from_qk,
        visual_cross_pre_visual_rows=visual_cross_pre_visual_rows,
        visual_cross_text_to_visual_rows=visual_cross_text_to_visual_rows,
        visual_cross_from_qk_rows=visual_cross_from_qk_rows,
        visual_cross_pre_visual_logits=visual_cross_pre_visual_logits,
        visual_cross_text_to_visual_head_rows=visual_cross_text_to_visual_head_rows,
        visual_cross_from_qk_logits=visual_cross_from_qk_logits,
        visual_redistributed_attention=visual_redistributed_attention,
        generated_query_visual_attentions=generated_query_visual_attentions,
        prune_layer_input_layernorm_weight=prune_layer_input_layernorm_weight,
        prune_layer_input_layernorm_eps=prune_layer_input_layernorm_eps,
        sink_local_ids=sink_local_ids,
        sink_abs_ids=sink_abs_ids,
        sink_scores=sink_scores,
        kept_visual_local_ids=kept_visual_local_ids,
        pruned_visual_local_ids=pruned_visual_local_ids,
        receiver_indices=receiver_indices,
        receiver_weights=receiver_weights,
        sink_budget=sink_budget,
        receiver_selection_scores=receiver_selection_scores,
        receiver_weight_scores=receiver_weight_scores,
    )
    return snapshot


@torch.no_grad()
def capture(args: argparse.Namespace) -> PrefillDebugSnapshot:
    device = resolve_device(args)
    dtype = resolve_dtype(args.dtype, device)
    fastv_config = build_fastv_config(args)
    model, processor = load_model_and_processor(args, device, dtype, fastv_config)
    question = load_single_question(args.question_file, args.question_id, args.line_index)
    return capture_question(args, model, processor, question, device, dtype, fastv_config)


def _safe_name(value) -> str:
    """Turn a question id into a filesystem-safe file/folder name."""
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return text.strip("_") or "unknown"


def _archive_path(out_path: Path, question_id) -> Path:
    """Per-question-id snapshot path kept around across runs.

    e.g. snapshots/textvqa_single_case.pt -> snapshots/textvqa/<question_id>.pt
    """
    dataset = out_path.stem
    for suffix in ("_single_case", "_case"):
        if dataset.endswith(suffix):
            dataset = dataset[: -len(suffix)]
            break
    return out_path.parent / dataset / f"{_safe_name(question_id)}{out_path.suffix or '.pt'}"


def _refresh_latest_pointer(canonical: Path, target: Path) -> None:
    """Point the canonical --out path at `target` (symlink, or a copy as fallback) so the
    render script's default snapshot path keeps resolving to the most recent case."""
    if canonical.resolve() == target.resolve():
        return
    canonical.parent.mkdir(parents=True, exist_ok=True)
    if canonical.is_symlink() or canonical.exists():
        canonical.unlink()
    try:
        canonical.symlink_to(os.path.relpath(target, canonical.parent))
    except OSError:
        shutil.copyfile(target, canonical)


def main():
    args = parse_args()
    snapshot = capture(args)
    # Save the real snapshot under a per-question-id name so changing the case does not
    # overwrite earlier ones; keep --out as a "latest" pointer for the render default.
    archive_path = _archive_path(Path(args.out), snapshot.question_id)
    saved_path = save_snapshot(snapshot, archive_path)
    _refresh_latest_pointer(Path(args.out), saved_path)
    print(f"question_id={snapshot.question_id} answer={snapshot.generated_answer!r}")
    print(f"sink tokens: {snapshot.sink_local_ids.tolist()} (scores={[round(s, 3) for s in snapshot.sink_scores.tolist()]})")
    print(f"wrote snapshot to {saved_path}")
    print(f"latest pointer: {Path(args.out)} -> {saved_path}")


if __name__ == "__main__":
    main()
