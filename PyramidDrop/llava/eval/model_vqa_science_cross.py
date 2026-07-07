import argparse
import json
import math
import os

import shortuuid
import torch
from PIL import Image
from tqdm import tqdm

from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

from cross_attention_sink_redistribution_llava.attention_redistribution import (
    REDISTRIBUTION_SOFTMAX_MODES,
    REDISTRIBUTION_STRATEGIES,
    sink_attention_redistributor,
)
from cross_attention_sink_redistribution_llava.cross_attention import cross_attention_importants
from cross_attention_sink_redistribution_llava.sink_tokens import sink_token_selector


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks."""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    pdrop_infer = args.pdrop_infer or args.layer_list is not None
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path,
        args.model_base,
        model_name,
        pdrop_infer,
    )

    model_class_name = type(model).__name__
    if model_class_name == "LlavaLlamaForCausalLM_PDrop":
        model.model.layer_list = eval(args.layer_list)
        model.model.image_token_ratio_list = eval(args.image_token_ratio_list)
        model.model.image_token_ratio_list.insert(0, 1.0)
        model.model.sink_selector = sink_token_selector(args)
        model.model.cross_attention_importants = cross_attention_importants(
            args, sink_selector=model.model.sink_selector
        )
        model.model.sink_attention_redistributor = sink_attention_redistributor(args)

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    with open(answers_file, "w") as ans_file:
        for line in tqdm(questions):
            idx = line["id"]
            question = line["conversations"][0]
            qs = question["value"].replace("<image>", "").strip()
            cur_prompt = qs

            if "image" in line:
                image_file = line["image"]
                image = Image.open(os.path.join(args.image_folder, image_file))
                image_tensor = process_images([image], image_processor, model.config)[0]
                images = image_tensor.unsqueeze(0).half().cuda()
                image_sizes = [image.size]
                if getattr(model.config, "mm_use_im_start_end", False):
                    qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
                else:
                    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
                cur_prompt = "<image>\n" + cur_prompt
            else:
                images = None
                image_sizes = None

            if args.single_pred_prompt:
                suffix = "Answer with the option's letter from the given choices directly."
                qs = qs + "\n" + suffix
                cur_prompt = cur_prompt + "\n" + suffix

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).cuda()

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images,
                    image_sizes=image_sizes,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    max_new_tokens=1024,
                    use_cache=True,
                )

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            ans_id = shortuuid.uuid()
            ans_file.write(
                json.dumps(
                    {
                        "question_id": idx,
                        "prompt": cur_prompt,
                        "text": outputs,
                        "answer_id": ans_id,
                        "model_id": model_name,
                        "metadata": {},
                    }
                )
                + "\n"
            )
            ans_file.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v0")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--layer_list", type=str, default=None)
    parser.add_argument("--image_token_ratio_list", type=str, default=None)
    parser.add_argument("--pdrop_infer", action="store_true")

    parser.add_argument("--sink_dims", type=int, nargs="+", default=[2533])
    parser.add_argument("--sink_score_min", type=float, default=None)
    parser.add_argument("--sink_score_max", type=float, default=None)
    parser.add_argument("--sink_score_quantile", type=float, default=0.99)

    parser.add_argument("--disable_sink_masked", action="store_false", dest="enable_sink_masked")

    parser.add_argument("--redistribution_ratio", type=float, default=1.0)
    parser.add_argument(
        "--redistribution_strategy",
        type=str,
        default="topk_text_visual_tokens",
        choices=REDISTRIBUTION_STRATEGIES,
    )
    parser.add_argument(
        "--redistribution_softmax_mode",
        type=str,
        default="post_softmax",
        choices=REDISTRIBUTION_SOFTMAX_MODES,
    )
    parser.add_argument("--receiver_token_count", type=int, default=0)
    parser.add_argument("--receiver_score_power", type=float, default=1.0)
    parser.set_defaults(enable_sink_masked=True)

    args = parser.parse_args()
    eval_model(args)
