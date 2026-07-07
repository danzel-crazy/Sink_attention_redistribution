import argparse
import json
import math
import os
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    StoppingCriteria,
    StoppingCriteriaList,
    TextStreamer,
)

from cross_attention_sink_redistribution.attention_redistribution import (
    REDISTRIBUTION_SOFTMAX_MODES,
    REDISTRIBUTION_STRATEGIES,
)

ANSWER_SUFFIX = "\nAnswer the question using a single word or phrase."
PROMPT_TEMPLATE = "USER: <image>\n{question}\nASSISTANT:"


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks."""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    return split_list(lst, n)[k]


def strip_answer_suffix(text):
    return text.replace(ANSWER_SUFFIX, "")


def build_prompt(question):
    return PROMPT_TEMPLATE.format(question=question)


def resolve_image_path(image_folder, image_path):
    if os.path.isabs(image_path):
        return image_path
    return os.path.join(image_folder, image_path)


def extract_answer(decoded, stop_texts=None):
    if "ASSISTANT:" in decoded:
        decoded = decoded.split("ASSISTANT:")[-1]
    if stop_texts:
        for text in stop_texts:
            if text in decoded:
                decoded = decoded.split(text)[0]
    return decoded.strip()


def resolve_dtype(dtype_name, device):
    if dtype_name == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


class StopOnTokenSequences(StoppingCriteria):
    def __init__(self, sequences):
        self.sequences = [torch.tensor(seq, dtype=torch.long) for seq in sequences if seq]
        self._device = None
        self._sequences_on_device = None

    def _ensure_device(self, device):
        if self._device != device:
            self._sequences_on_device = [seq.to(device) for seq in self.sequences]
            self._device = device

    def __call__(self, input_ids, scores, **kwargs):
        if not self.sequences:
            return False
        self._ensure_device(input_ids.device)
        for seq in self._sequences_on_device:
            seq_len = seq.numel()
            if input_ids.shape[1] < seq_len:
                continue
            if (input_ids[:, -seq_len:] == seq).all(dim=1).any():
                return True
        return False


def build_stop_texts(args):
    stop_texts = list(args.stop_strings)
    if args.stop_on_user:
        stop_texts.extend(["\nUSER:", "USER:"])
    return stop_texts


def build_stopping_criteria(tokenizer, stop_texts):
    sequences = []
    for text in stop_texts:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if token_ids:
            sequences.append(tuple(token_ids))
    if not sequences:
        return None
    unique_sequences = list(dict.fromkeys(sequences))
    return StoppingCriteriaList([StopOnTokenSequences(unique_sequences)])


def build_fastv_config(args):
    if not args.use_fastv:
        return None
    return {
        "use_fastv": True,
        "fastv_k": args.fastv_k,
        "fastv_r": args.fastv_r,
        "image_token_start_index": args.image_token_start_index,
        "image_token_length": args.visual_token_num,
        # sink_token_selector (cross_attention_sink_redistribution/sink_tokens.py)
        "sink_dims": args.sink_dims,
        "sink_score_min": args.sink_score_min,
        "sink_score_max": args.sink_score_max,
        "sink_score_quantile": args.sink_score_quantile,
        # cross_attention_importants (cross_attention_sink_redistribution/cross_attention.py)
        "enable_sink_masked": args.enable_sink_masked,
        "text_tokens_start_index": 0,
        "text_tokens_length": (
            args.text_tokens_length if args.text_tokens_length is not None else args.image_token_start_index
        ),
        "visual_tokens_length": args.visual_token_num,
        # sink_attention_redistributor (cross_attention_sink_redistribution/attention_redistribution.py)
        "redistribution_ratio": args.redistribution_ratio,
        "redistribution_strategy": args.redistribution_strategy,
        "redistribution_softmax_mode": args.redistribution_softmax_mode,
        "receiver_token_count": args.receiver_token_count,
        "receiver_score_power": args.receiver_score_power,
    }


def prepare_inputs(processor, prompt, image, device, dtype):
    inputs = processor(prompt, image, return_tensors="pt")
    for key, value in inputs.items():
        if torch.is_floating_point(value):
            inputs[key] = value.to(device=device, dtype=dtype)
        else:
            inputs[key] = value.to(device=device)
    return inputs


def eval_model(args):
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
        fastv_config=fastv_config, # comment this line to use vanilla decoding
    ).to(device)
    model.visual_token_num = args.visual_token_num
    model.enable_cdpruner = False
    model.visualize_cdpruner = False
    model.enable_fastV = True
    model.fastv_info = None
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model_id, revision=revision)
    tokenizer = processor.tokenizer
    stop_texts = build_stop_texts(args)
    stopping_criteria = build_stopping_criteria(tokenizer, stop_texts)

    with open(os.path.expanduser(args.question_file), "r") as file_handle:
        questions = [json.loads(q) for q in file_handle]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    answers_dir = os.path.dirname(answers_file)
    if answers_dir:
        os.makedirs(answers_dir, exist_ok=True)

    model_id = args.model_id
    with open(answers_file, "w") as ans_file:
        for line in tqdm(questions):
            idx = line["question_id"]
            cur_prompt = line["text"]
            image_path = resolve_image_path(args.image_folder, line["image"])
            question = strip_answer_suffix(cur_prompt)

            prompt = build_prompt(cur_prompt)
            image = Image.open(image_path).convert("RGB")
            inputs = prepare_inputs(processor, prompt, image, device, dtype)

            streamer = TextStreamer(processor)

            gen_kwargs = {
                "min_new_tokens": args.min_new_tokens,
                "max_new_tokens": args.max_new_tokens,
                "do_sample": args.temperature > 0,
                "num_beams": args.num_beams,
                "use_cache": True,
                "streamer": streamer,
                "return_dict_in_generate": True,
                "output_attentions": True,
            }
            if args.temperature > 0:
                gen_kwargs["temperature"] = args.temperature
                if args.top_p is not None:
                    gen_kwargs["top_p"] = args.top_p
            if stopping_criteria is not None:
                gen_kwargs["stopping_criteria"] = stopping_criteria

            output = model.generate(**inputs, **gen_kwargs)
            decoded = processor.batch_decode(
                output.sequences,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            answer = extract_answer(decoded, stop_texts=stop_texts)

            # ans_id = shortuuid.uuid()
            ans_file.write(
                json.dumps(
                    {
                        "question_id": idx,
                        "prompt": cur_prompt,
                        "text": answer,
                        "model_id": model_id,
                        "metadata": {},
                    }
                )
                + "\n"
            )
            ans_file.flush()
            # break
        ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--revision", type=str, default="a272c74")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--min_new_tokens", type=int, default=0)
    parser.add_argument("--visual_token_num", type=int, default=128)
    parser.add_argument("--use_fastv", action="store_true", dest="use_fastv")
    parser.add_argument("--fastv_k", type=int, default=5)
    parser.add_argument("--fastv_r", type=float, default=0.75)
    parser.add_argument("--image_token_start_index", type=int, default=5)
    parser.add_argument("--use_cache", action="store_true", dest="use_cache")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--stop_on_user", action="store_true", dest="stop_on_user")
    parser.add_argument("--stop_strings", action="append", default=[])
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["auto", "float16", "bfloat16", "float32"],
    )

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
        help="Defaults to --image_token_start_index (the system-prompt prefix length) if unset.",
    )

    # sink_attention_redistributor (cross_attention_sink_redistribution/attention_redistribution.py)
    parser.add_argument("--redistribution-ratio", type=float, default=1.0)
    parser.add_argument(
        "--redistribution-strategy",
        type=str,
        default="topk_text_visual_tokens",
        choices=REDISTRIBUTION_STRATEGIES,
    )
    parser.add_argument(
        "--redistribution-softmax-mode",
        type=str,
        default="post_softmax",
        choices=REDISTRIBUTION_SOFTMAX_MODES,
    )
    parser.add_argument("--receiver-token-count", type=int, default=0)
    parser.add_argument("--receiver-score-power", type=float, default=1.0)
    parser.set_defaults(use_fastv=True, use_cache=True, stop_on_user=True, enable_sink_masked=True)
    args = parser.parse_args()

    eval_model(args)
