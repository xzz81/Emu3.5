# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from src.utils.logging_utils import setup_logger

cfg_name = Path(__file__).stem

model_path = "BAAI/Emu3.5-Image"
vq_path = "BAAI/Emu3.5-VisionTokenizer"
tokenizer_path = "./src/tokenizer_emu3_ibq"
vq_type = "ibq"

task_type = "t2i"
use_image = False

exp_name = "emu3p5-image"
save_path = f"./outputs/{exp_name}/{task_type}"
save_to_proto = True
setup_logger(save_path)

hf_device = "auto"
vq_device = "cuda:0"
streaming = False
unconditional_type = "no_text"
classifier_free_guidance = 5.0
max_new_tokens = 1064
image_area = 1048576

target_height = 32
target_width = 32


def build_unc_and_template(task: str, with_image: bool):
    task_str = task.lower()
    if with_image:
        unc_p = "<|extra_203|>You are a helpful assistant. USER: <|IMAGE|> ASSISTANT: <|extra_100|>"
        tmpl = "<|extra_203|>You are a helpful assistant for %s task. USER: {question}<|IMAGE|> ASSISTANT: <|extra_100|>" % task_str
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

seed = 20260526

prompts = {
    "six_yellow_lemons_grid": (
        "A square top-down studio photo on a plain white table. Show exactly six yellow lemons arranged as two rows of three. "
        "There must be no seventh lemon, no missing lemon, no leaves, no plate, no knife, no label, and no text."
    ),
    "blue_circle_occludes_red_square": (
        "A square flat vector-style image on a white background. A solid blue circle must be centered in front of a solid red square, "
        "partly hiding the square. The red square must remain visible only around the circle edges. There must be no text, no shadows, and no extra shapes."
    ),
    "secret_word_fully_covered_readable": (
        "A square close-up photo of a white card with the word SECRET printed in black. An opaque red rectangle must fully cover the entire word so that "
        "no letter is visible. The word SECRET must also remain clearly readable. Do not add a second card, second label, or extra text."
    ),
    "two_arrows_toward_and_away": (
        "A square minimalist diagram on a white background with exactly two black arrows on one horizontal line. The two arrows must point toward each other "
        "and away from each other at the same time, without using double-headed arrows, extra arrowheads, extra arrows, text, or symbols."
    ),
}
