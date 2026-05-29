# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from src.utils.logging_utils import setup_logger

cfg_name = Path(__file__).stem

model_path = "BAAI/Emu3.5-Image"
vq_path = "BAAI/Emu3.5-VisionTokenizer"
tokenizer_path = "./src/tokenizer_emu3_ibq"
vq_type = "ibq"

task_type = "x2i"
use_image = True

exp_name = "emu3p5-image"
save_path = f"./outputs/{exp_name}/{task_type}"
save_to_proto = True
setup_logger(save_path)

hf_device = "auto"
vq_device = "cuda:0"
streaming = False
unconditional_type = "no_text"
classifier_free_guidance = 3.0
max_new_tokens = 1300
image_area = 262144


def build_unc_and_template(task: str, with_image: bool):
    task_str = task.lower()
    if with_image:
        unc_p = "<|extra_203|>You are a helpful assistant. USER: <|IMAGE|> ASSISTANT: <|extra_100|>"
        tmpl = "<|extra_203|>You are a helpful assistant for %s task. USER: <|IMAGE|>{question} ASSISTANT: <|extra_100|>" % task_str
    else:
        unc_p = "<|extra_203|>You are a helpful assistant. USER:  ASSISTANT: <|extra_100|>"
        tmpl = "<|extra_203|>You are a helpful assistant for %s task. USER: {question} ASSISTANT: <|extra_100|>" % task_str
    return unc_p, tmpl


unc_prompt, template = build_unc_and_template(task_type, use_image)

sampling_params = dict(
    use_cache=True,
    text_top_k=1024,
    text_top_p=0.9,
    text_temperature=1.0,
    image_top_k=5120,
    image_top_p=1.0,
    image_temperature=1.0,
    top_k=131072,
    top_p=1.0,
    temperature=1.0,
    num_beams_per_group=1,
    num_beam_groups=1,
    diversity_penalty=0.0,
    max_new_tokens=max_new_tokens,
    guidance_scale=1.0,
    use_differential_sampling=True,
)

sampling_params["do_sample"] = sampling_params["num_beam_groups"] <= 1
sampling_params["num_beams"] = sampling_params["num_beams_per_group"] * sampling_params["num_beam_groups"]

special_tokens = dict(
    BOS="<|extra_203|>",
    EOS="<|extra_204|>",
    PAD="<|endoftext|>",
    EOL="<|extra_200|>",
    EOF="<|extra_201|>",
    TMS="<|extra_202|>",
    IMG="<|image token|>",
    BOI="<|image start|>",
    EOI="<|image end|>",
    BSS="<|extra_100|>",
    ESS="<|extra_101|>",
    BOG="<|extra_60|>",
    EOG="<|extra_61|>",
    BOC="<|extra_50|>",
    EOC="<|extra_51|>",
)

seed = 20260524

reference_pair = ["./assets/ref_0.png", "./assets/ref_1.png"]

prompts = {
    "remove_and_keep_truffle": {
        "prompt": (
            "Use the first reference image as the scene layout and lighting. The central brown chocolate truffle "
            "must be completely removed, but the same central brown chocolate truffle must also remain clearly "
            "visible in the exact original position. Do not create two truffles and do not replace it with fruit."
        ),
        "reference_image": reference_pair,
    },
    "red_and_blue_same_strawberry": {
        "prompt": (
            "Use the first reference image as the scene layout and lighting. Replace the central truffle with one "
            "single strawberry shaped like the second reference strawberry. The single strawberry must be entirely "
            "red and entirely blue at the same time, with no split halves, no gradient, and no second strawberry."
        ),
        "reference_image": reference_pair,
    },
    "one_object_left_and_right": {
        "prompt": (
            "Use the white cloth and garden lighting from the first reference image. Put exactly one object on the "
            "cloth: the brown truffle from the first reference. This one object must be clearly on the far left and "
            "clearly on the far right at the same time. Do not add a second object, duplicate, mirror, or reflection."
        ),
        "reference_image": reference_pair,
    },
    "tag_word_strawberry_on_truffle": {
        "prompt": (
            "Use the first reference image as the scene layout and lighting. Keep the brown truffle on the cloth, "
            "and attach a small white tag to it. The tag must show exactly one readable word: STRAWBERRY. The image "
            "must not contain any strawberry fruit, red fruit, or second object."
        ),
        "reference_image": reference_pair,
    },
}
