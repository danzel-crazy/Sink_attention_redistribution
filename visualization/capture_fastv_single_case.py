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
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

from cross_attention_sink_redistribution.attention_redistribution import (
    REDISTRIBUTION_SOFTMAX_MODES,
    REDISTRIBUTION_STRATEGIES,
)
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
    parser.add_argument("--redistribution-strategy", type=str, default="topk_text_visual_tokens", choices=REDISTRIBUTION_STRATEGIES)
    parser.add_argument("--redistribution-softmax-mode", type=str, default="post_softmax_resoftmax", choices=REDISTRIBUTION_SOFTMAX_MODES)
    parser.add_argument("--receiver-token-count", type=int, default=32)
    parser.add_argument("--receiver-score-power", type=float, default=1.0)
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


@torch.no_grad()
def capture(args: argparse.Namespace) -> PrefillDebugSnapshot:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = torch.device("cpu")
    dtype = resolve_dtype(args.dtype, device)
    revision = args.revision if args.revision else None
    fastv_config = build_fastv_config(args)

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

    question = load_single_question(args.question_file, args.question_id, args.line_index)
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
    handle = llama_model.layers[hook_layer_idx].register_forward_hook(make_prune_layer_hook(state))
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

    cross_attn = llama_model.cross_attention_importants
    visual_cross_attention_raw = cross_attn.important_tokens_scores_raw.detach().to("cpu", torch.float32).clone()
    visual_cross_attention_masked = cross_attn.important_tokens_scores.detach().to("cpu", torch.float32).clone()

    last_result = llama_model.sink_attention_redistributor.last_result
    if last_result is None:
        raise RuntimeError("sink_attention_redistributor.redistribute() was never called during this run.")
    receiver_local_positions = torch.as_tensor(last_result.receiver_local_positions, dtype=torch.long)
    visual_redistributed_attention = visual_self_attention.clone()
    visual_redistributed_attention[sink_local_ids] = 0.0
    visual_redistributed_attention[receiver_local_positions] = (
        last_result.redistributed_scores.detach().to("cpu", torch.float32)
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
        visual_redistributed_attention=visual_redistributed_attention,
        generated_query_visual_attentions=generated_query_visual_attentions,
        sink_local_ids=sink_local_ids,
        sink_abs_ids=sink_abs_ids,
        sink_scores=sink_scores,
        kept_visual_local_ids=kept_visual_local_ids,
        pruned_visual_local_ids=pruned_visual_local_ids,
    )
    return snapshot


def main():
    args = parse_args()
    snapshot = capture(args)
    out_path = save_snapshot(snapshot, Path(args.out))
    print(f"question_id={snapshot.question_id} answer={snapshot.generated_answer!r}")
    print(f"sink tokens: {snapshot.sink_local_ids.tolist()} (scores={[round(s, 3) for s in snapshot.sink_scores.tolist()]})")
    print(f"wrote snapshot to {out_path}")


if __name__ == "__main__":
    main()
