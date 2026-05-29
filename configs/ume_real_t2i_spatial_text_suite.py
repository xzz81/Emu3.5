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
    "diagonal_three_objects": (
        "A square top-down studio photo on a plain white table. Place exactly three objects on a diagonal from "
        "top-left to bottom-right: a red apple, a blue ceramic cup, and a yellow banana. There must be no fourth "
        "object, no green object, no text, and no reflection."
    ),
    "shelf_books_exact_words": (
        "A square close-up image of exactly three upright books on a plain wooden shelf. The book spines must read "
        "SUN, MOON, and STAR from left to right, with no other readable words, letters, numbers, or labels."
    ),
    "clock_two_times": (
        "A square realistic photo of exactly one digital alarm clock on a plain table. The display must show exactly "
        "07:45 and also exactly 12:30 at the same time on the same screen. Do not add a second clock, mirror, split "
        "screen, or any other numbers."
    ),
    "nested_shapes_order": (
        "A square flat vector-like icon on a white background. Draw exactly three nested shapes centered in the image: "
        "a large red square outside, a medium blue circle inside it, and a small yellow triangle inside the circle. "
        "There must be no text, shadow, gradient, or extra shape."
    ),
    "one_key_two_positions": (
        "A square studio photo of exactly one single brass key on a plain gray table. The same single key must be "
        "clearly on the far left and clearly on the far right at the same time. Do not add a second key, duplicate, "
        "reflection, mirror, label, or text."
    ),
    "blank_label_with_word": (
        "A square close-up photo of one white product label on a brown cardboard box. The label must be completely "
        "blank with no letters, no numbers, no symbols, and no barcode. The same label must also contain exactly one "
        "readable word: FRAGILE. Do not add a second label."
    ),
}
