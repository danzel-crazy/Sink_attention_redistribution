"""Run a RANGE of benchmark examples through the current PyramidDrop + sink-redistribution
pipeline and dump the FIRST pruning stage of each one to its own snapshot .pt file.

Only the first pruning is captured. PyramidDrop prunes once per entry in `layer_list`
(e.g. [2,10,20]), but at the first pruning the visual block is still the full, untouched
grid (`image_token_ratio_list` gets 1.0 inserted at the front by the eval loader), so a
visual token's local id *is* its patch id in the image grid. That means the snapshot fits
the existing `PrefillDebugSnapshot` schema unchanged and renders with the existing
`visualization.combined_fastv_collection` renderer. Later stages renumber the survivors
0..K-1 and would need a survivor->grid mapping; they are deliberately not recorded.

Nothing in PyramidDrop/ or cross_attention_sink_redistribution_llava/ is modified. The
data is collected with two read-only pass-through wrappers installed on the live model
instance for the duration of `generate()` and removed in a `finally` block:

  * `LlamaModel.pdrop_rank_drop` - gives the stage number (`cur_num`), the prune layer,
    and `features`, the hidden states entering the pruning step.
  * `sink_attention_redistributor.redistribute` - gives the self-attention vector going in
    (the "before"), the redistributed vector coming out (the "after"), the sink ids and the
    sink budget; the cross-attention importance and sink scores are read off the pipeline
    objects at the same moment.

Both wrappers forward their arguments unchanged and return the original object untouched,
so the model produces exactly the answer it would produce without capture attached.

Usage:
    python -m visualization.capture_pdrop_multiple_cases \\
        --model-path liuhaotian/llava-v1.5-7b \\
        --question-file /path/to/llava_pope_test.jsonl \\
        --image-folder /path/to/val2014 \\
        --layer_list '[2,10,20]' \\
        --image_token_ratio_list '[0.32,0.16,0.08]' \\
        --cases 1-10 \\
        --out visualization/snapshots/pdrop/pope_single_case.pt
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image

from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

from cross_attention_sink_redistribution_llava.attention_redistribution import (
    RECEIVER_WEIGHT_MODES,
    REDISTRIBUTION_SOFTMAX_MODES,
    REDISTRIBUTION_STRATEGY_CHOICES,
    sink_attention_redistributor,
)
from cross_attention_sink_redistribution_llava.cross_attention import cross_attention_importants
from cross_attention_sink_redistribution_llava.sink_tokens import sink_token_selector
from visualization.fastv_snapshot import PrefillDebugSnapshot

# The stage we capture. PyramidDrop numbers pruning stages from 0
# (`stage = layer_list.index(rank_layer)`), so stage 0 is the first pruning.
CAPTURE_STAGE = 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")

    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument(
        "--cases",
        type=str,
        default="1-10",
        help="1-based inclusive case range: 'START-END' (e.g. 1-10), a single case ('5'), or open-ended ('3-').",
    )

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=8)

    parser.add_argument("--out", type=str, default="visualization/snapshots/pdrop/pope_single_case.pt")

    # pdrop knobs - same names/defaults as PyramidDrop/llava/eval/model_vqa_loader_cross.py
    parser.add_argument("--layer_list", type=str, default="[2,10,20]")
    parser.add_argument("--image_token_ratio_list", type=str, default="[0.32,0.16,0.08]")

    # sink_token_selector (cross_attention_sink_redistribution_llava/sink_tokens.py)
    parser.add_argument("--sink_dims", type=int, nargs="+", default=[2533])
    parser.add_argument("--sink_score_min", type=float, default=None)
    parser.add_argument("--sink_score_max", type=float, default=None)
    parser.add_argument("--sink_score_quantile", type=float, default=0.99)

    # cross_attention_importants (cross_attention_sink_redistribution_llava/cross_attention.py)
    parser.add_argument("--disable_sink_masked", action="store_false", dest="enable_sink_masked")

    # sink_attention_redistributor (cross_attention_sink_redistribution_llava/attention_redistribution.py)
    parser.add_argument("--redistribution_ratio", type=float, default=1.0)
    parser.add_argument("--redistribution_strategy", type=str, default="topk_text_visual_tokens", choices=REDISTRIBUTION_STRATEGY_CHOICES)
    # accepted but ignored - see the back-compat note in attention_redistribution.py
    parser.add_argument("--redistribution_softmax_mode", type=str, default="post_softmax", choices=REDISTRIBUTION_SOFTMAX_MODES)
    parser.add_argument("--receiver_token_count", type=int, default=0)
    parser.add_argument("--receiver_score_power", type=float, default=1.0)
    parser.add_argument("--receiver_weight_mode", type=str, default="cross", choices=RECEIVER_WEIGHT_MODES)
    parser.set_defaults(enable_sink_masked=True)
    return parser


# --- snapshot path helpers -------------------------------------------------------------
# Deliberately duplicated from visualization/capture_fastv_single_case.py rather than
# imported: that module pulls in FastV's patched transformers fork at import time, which
# must not be loaded into a process running PyramidDrop's own llava/transformers stack.


def _safe_name(value) -> str:
    """Turn a question id into a filesystem-safe file/folder name."""
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return text.strip("_") or "unknown"


def _archive_path(out_path: Path, question_id) -> Path:
    """Per-question-id snapshot path kept around across runs.

    e.g. snapshots/pdrop/pope_single_case.pt -> snapshots/pdrop/pope/<question_id>.pt
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


def save_pdrop_snapshot(snapshot: PrefillDebugSnapshot, extras: Dict[str, Any], path: Path) -> Path:
    """Save a PrefillDebugSnapshot plus PDrop-only bookkeeping keys.

    The extras (prune_layer, image_token_ratio_list, keep_length) are not fields of
    PrefillDebugSnapshot; `fastv_snapshot.load_snapshot` filters unknown keys, so they ride
    along harmlessly and are available to anything that reads the raw payload.
    """
    payload = asdict(snapshot)
    payload.update(extras)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


# --- read-only capture wrappers --------------------------------------------------------


class FirstPruneRecorder:
    """Records the first pruning stage of one prefill, via pass-through wrappers.

    Install with `install(model)`, always uninstall with `remove()` in a finally block.
    `reset()` before each question so "the first redistribute call" is unambiguous.
    """

    def __init__(self):
        self.reset()
        self._model = None
        self._original_rank_drop = None
        self._original_redistribute = None

    def reset(self) -> None:
        self.active_stage: Optional[int] = None
        self.record: Optional[Dict[str, Any]] = None

    # -- wrappers --

    def _wrap_rank_drop(self, original):
        def wrapper(*args, **kwargs):
            # `pdrop_rank_drop` is always called with keywords from LlamaModel.forward.
            cur_num = kwargs.get("cur_num", args[0] if args else None)
            self.active_stage = cur_num
            if cur_num == CAPTURE_STAGE and self.record is None:
                features = kwargs.get("features")
                self.record = {
                    "prune_layer": kwargs.get("rank_layer"),
                    # features is [B, seq, hidden]; batch is 1 for this capture path
                    "hidden_states_at_prune_layer": features[0].detach().to("cpu", torch.float32).clone(),
                }
            return original(*args, **kwargs)

        return wrapper

    def _wrap_redistribute(self, redistributor, cross_attn, sink_selector, original):
        def wrapper(image_attention, important_scores, sink_local_ids, sink_budget=None, *args, **kwargs):
            result = original(image_attention, important_scores, sink_local_ids, sink_budget, *args, **kwargs)
            if self.active_stage == CAPTURE_STAGE and self.record is not None and "baseline" not in self.record:
                self.record.update(
                    baseline=image_attention.detach().to("cpu", torch.float32).clone(),
                    redistributed=result.detach().to("cpu", torch.float32).clone(),
                    cross_raw=_to_cpu(cross_attn.important_tokens_scores_raw),
                    cross_masked=_to_cpu(cross_attn.important_tokens_scores),
                    sink_local_ids=sink_local_ids.detach().to("cpu", torch.long).clone(),
                    sink_scores=_to_cpu(sink_selector.sink_tokens_scores),
                    sink_budget=float(sink_budget) if sink_budget is not None else None,
                    last_result=redistributor.last_result,
                )
            return result

        return wrapper

    # -- lifecycle --

    def install(self, model) -> None:
        llama_model = model.model
        self._model = llama_model
        self._original_rank_drop = llama_model.pdrop_rank_drop
        self._original_redistribute = llama_model.sink_attention_redistributor.redistribute
        llama_model.pdrop_rank_drop = self._wrap_rank_drop(self._original_rank_drop)
        llama_model.sink_attention_redistributor.redistribute = self._wrap_redistribute(
            llama_model.sink_attention_redistributor,
            llama_model.cross_attention_importants,
            llama_model.sink_selector,
            self._original_redistribute,
        )

    def remove(self) -> None:
        if self._model is None:
            return
        # Delete the instance attributes so the class-level method is visible again,
        # leaving the model byte-for-byte as it was before install().
        self._model.__dict__.pop("pdrop_rank_drop", None)
        self._model.sink_attention_redistributor.__dict__.pop("redistribute", None)
        self._model = None
        self._original_rank_drop = None
        self._original_redistribute = None


def _to_cpu(value) -> Optional[torch.Tensor]:
    if value is None or (isinstance(value, list) and len(value) == 0):
        return None
    return torch.as_tensor(value).detach().to("cpu", torch.float32).clone()


# --- capture ---------------------------------------------------------------------------


def parse_case_range(cases: str, total: int) -> Tuple[int, int]:
    """Parse a 1-based inclusive '--cases' spec into 0-based [start, end) indices.

    Accepts 'A-B', a single 'A', or open-ended 'A-'. Clamps to [1, total]."""
    text = cases.strip()
    if "-" in text:
        start_str, end_str = text.split("-", 1)
        start = int(start_str) if start_str.strip() else 1
        end = int(end_str) if end_str.strip() else total
    else:
        start = end = int(text)
    if start < 1 or end < start:
        raise ValueError(f"Invalid --cases {cases!r} (parsed start={start}, end={end}).")
    if start > total:
        raise ValueError(f"--cases start {start} is past the last case ({total}).")
    end = min(end, total)
    return start - 1, end  # 0-based half-open


def load_questions(question_file: str) -> List[Dict[str, Any]]:
    with open(os.path.expanduser(question_file), "r") as file_handle:
        return [json.loads(line) for line in file_handle]


def build_prompt(args: argparse.Namespace, model, question_text: str) -> str:
    """Same prompt construction as PyramidDrop/llava/eval/model_vqa_loader_cross.py."""
    if model.config.mm_use_im_start_end:
        question_text = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + question_text
    else:
        question_text = DEFAULT_IMAGE_TOKEN + "\n" + question_text
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], question_text)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def _token_strings(tokenizer, token_ids: torch.Tensor) -> List[str]:
    """convert_ids_to_tokens, but tolerant of llava's IMAGE_TOKEN_INDEX (-200) placeholder."""
    strings = []
    for token_id in token_ids.tolist():
        if token_id == IMAGE_TOKEN_INDEX:
            strings.append(DEFAULT_IMAGE_TOKEN)
        else:
            strings.append(tokenizer.convert_ids_to_tokens(token_id))
    return strings


@torch.inference_mode()
def capture_question(
    args: argparse.Namespace,
    model,
    tokenizer,
    image_processor,
    question: Dict[str, Any],
    recorder: FirstPruneRecorder,
    ratio_list: List[float],
    layer_list: List[int],
) -> Tuple[PrefillDebugSnapshot, Dict[str, Any]]:
    """Run one question through the already-loaded model and build its snapshot."""
    cur_prompt = question["text"]
    image_path = os.path.join(args.image_folder, question["image"])
    prompt = build_prompt(args, model, cur_prompt)

    image = Image.open(image_path).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)[0]
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
    input_ids = input_ids.to(device="cuda", non_blocking=True)

    recorder.reset()
    output_ids = model.generate(
        input_ids,
        images=image_tensor.unsqueeze(0).to(dtype=torch.float16, device="cuda", non_blocking=True),
        image_sizes=[image.size],
        do_sample=args.temperature > 0,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        use_cache=True,
    )

    record = recorder.record
    if record is None:
        raise RuntimeError(
            "The pdrop_rank_drop wrapper never fired for stage 0 - is --layer_list set and "
            "is the model really LlavaLlamaForCausalLM_PDrop?"
        )
    if "baseline" not in record:
        raise RuntimeError(
            "sink_attention_redistributor.redistribute() was never called during the first "
            "pruning stage - is the cross-attention pipeline actually wired up?"
        )

    generated_answer = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    generated_token_ids = output_ids[0].detach().to("cpu").clone()
    prompt_token_ids = input_ids[0].detach().to("cpu").clone()

    hidden_states_at_prune_layer = record["hidden_states_at_prune_layer"]  # [seq_len, hidden_dim]
    prompt_length, hidden_dim = hidden_states_at_prune_layer.shape

    image_start = int(model.model.image_token_posi[0])
    # stage 0 sees the full visual block: ratio_list[0] is the 1.0 the eval loader inserts
    image_len = int(record["baseline"].numel())
    keep_length = int(image_len * ratio_list[1])

    baseline = record["baseline"]
    redistributed = record["redistributed"]

    # PyramidDrop only ever materializes the last-instruction-token attention row over the
    # visual span, so the full-sequence row FastV stores is not available here. Keep the
    # field shaped correctly with the visual span filled; nothing in the renderer reads it.
    self_attention_last_token = torch.zeros(prompt_length, dtype=torch.float32)
    self_attention_last_token[image_start:image_start + image_len] = baseline

    sink_local_ids = record["sink_local_ids"]
    sink_scores = record["sink_scores"]
    if sink_scores is None:
        sink_scores = torch.empty(0, dtype=torch.float32)

    last_result = record["last_result"]
    receiver_indices = last_result.receiver_indices.detach().to("cpu", torch.long).clone()
    receiver_weights = last_result.receiver_weights.detach().to("cpu", torch.float32).clone()
    receiver_selection_scores = _to_cpu(last_result.selection_scores)
    receiver_weight_scores = _to_cpu(last_result.weight_scores)

    # replay of the real keep rule (modeling_llama_pdrop.py: pdrop_rank_drop)
    kept_visual_local_ids = redistributed.topk(keep_length).indices.sort().values.clone()
    keep_mask = torch.zeros(image_len, dtype=torch.bool)
    keep_mask[kept_visual_local_ids] = True
    pruned_visual_local_ids = torch.nonzero(~keep_mask, as_tuple=False).squeeze(1)

    snapshot = PrefillDebugSnapshot(
        model_id=args.model_path,
        question_id=question.get("question_id"),
        image_path=image_path,
        question=cur_prompt,
        prompt=prompt,
        generated_answer=generated_answer,
        # PyramidDrop equivalents of the FastV knobs, so the shared renderer needs no changes:
        # the prune layer and the fraction of visual tokens dropped at this stage.
        fastv_k=int(layer_list[0]),
        fastv_r=1.0 - float(ratio_list[1]),
        image_token_start_index=image_start,
        image_token_length=image_len,
        text_tokens_start_index=image_start + image_len,
        text_tokens_length=prompt_length - (image_start + image_len),
        prompt_length=prompt_length,
        hidden_dim=hidden_dim,
        prompt_token_ids=prompt_token_ids,
        prompt_token_strings=_token_strings(tokenizer, prompt_token_ids),
        generated_token_ids=generated_token_ids,
        generated_token_strings=_token_strings(tokenizer, generated_token_ids),
        hidden_states_at_prune_layer=hidden_states_at_prune_layer,
        self_attention_last_token=self_attention_last_token,
        visual_hidden_states=hidden_states_at_prune_layer[image_start:image_start + image_len].clone(),
        visual_self_attention=baseline,
        visual_cross_attention_raw=record["cross_raw"],
        visual_cross_attention_masked=record["cross_masked"],
        visual_redistributed_attention=redistributed,
        sink_local_ids=sink_local_ids,
        sink_abs_ids=sink_local_ids + image_start,
        sink_scores=sink_scores,
        kept_visual_local_ids=kept_visual_local_ids,
        pruned_visual_local_ids=pruned_visual_local_ids,
        receiver_indices=receiver_indices,
        receiver_weights=receiver_weights,
        sink_budget=record["sink_budget"],
        receiver_selection_scores=receiver_selection_scores,
        receiver_weight_scores=receiver_weight_scores,
    )
    extras = {
        "prune_layer": record["prune_layer"],
        "image_token_ratio_list": list(ratio_list),
        "layer_list": list(layer_list),
        "keep_length": keep_length,
        "capture_stage": CAPTURE_STAGE,
    }
    return snapshot, extras


def main():
    args = _build_parser().parse_args()

    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, args.model_base, model_name, True  # pdrop_infer
    )

    model_class_name = type(model).__name__
    if model_class_name != "LlavaLlamaForCausalLM_PDrop":
        raise RuntimeError(f"Expected LlavaLlamaForCausalLM_PDrop, got {model_class_name}.")

    layer_list = eval(args.layer_list)
    ratio_list = eval(args.image_token_ratio_list)
    ratio_list.insert(0, 1.0)  # same convention as model_vqa_loader_cross.py
    model.model.layer_list = layer_list
    model.model.image_token_ratio_list = ratio_list

    # rebuild the add-on pipeline from the real CLI args, exactly as the eval loader does
    model.model.sink_selector = sink_token_selector(args)
    model.model.cross_attention_importants = cross_attention_importants(args, sink_selector=model.model.sink_selector)
    model.model.sink_attention_redistributor = sink_attention_redistributor(args)

    questions = load_questions(args.question_file)
    start_idx, end_idx = parse_case_range(args.cases, len(questions))

    out_path = Path(args.out)
    saved_paths: List[Path] = []
    recorder = FirstPruneRecorder()
    recorder.install(model)
    try:
        for zero_based in range(start_idx, end_idx):
            question = questions[zero_based]
            snapshot, extras = capture_question(
                args, model, tokenizer, image_processor, question, recorder, ratio_list, layer_list
            )
            archive_path = _archive_path(out_path, snapshot.question_id)
            saved_path = save_pdrop_snapshot(snapshot, extras, archive_path)
            saved_paths.append(saved_path)
            print(
                f"[case {zero_based + 1}/{end_idx}] question_id={snapshot.question_id} "
                f"answer={snapshot.generated_answer!r} -> {saved_path}"
            )
    finally:
        recorder.remove()

    if saved_paths:
        _refresh_latest_pointer(out_path, saved_paths[-1])
        print(f"captured {len(saved_paths)} case(s); latest pointer: {out_path} -> {saved_paths[-1]}")
    else:
        print("No cases captured (empty range).")


if __name__ == "__main__":
    main()
