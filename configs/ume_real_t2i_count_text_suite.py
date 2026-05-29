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

seed = 20260525

prompts = {
    "seven_red_beads": (
        "A square studio photo of exactly seven small red beads arranged in a straight horizontal row on a plain white table. "
        "There must be no eighth bead, no missing bead, no other objects, no text, and no reflections."
    ),
    "calendar_may_24_only": (
        "A square close-up photo of a simple white desk calendar page. It must show exactly one date: MAY 24. "
        "No other numbers, months, words, icons, handwriting, or extra pages may be visible."
    ),
    "four_colored_mugs": (
        "A square product photo with exactly four ceramic mugs in one row on a gray shelf: red, blue, yellow, and green from left to right. "
        "There must be no fifth mug, no saucer, no spoon, no logo, and no text."
    ),
    "three_stripes_exact_order": (
        "A square flat flag design with exactly three equal vertical stripes. From left to right the stripes must be black, white, and orange. "
        "There must be no border, symbol, emblem, texture, gradient, or text."
    ),
    "quiet_word_no_other_text": (
        "A square minimalist poster on a white wall. The poster must contain exactly one readable word in large black letters: QUIET. "
        "There must be no other words, numbers, signatures, symbols, or small text anywhere in the image."
    ),
    "no_text_but_word_quiet": (
        "A square minimalist poster on a white wall. The poster must be completely blank with no letters, no numbers, and no symbols. "
        "The same poster must also contain exactly one readable word: QUIET. Do not add a second poster."
    ),
    "empty_glass_full_water": (
        "A square studio photo of exactly one clear drinking glass on a white table. "
        "The glass must be completely empty and also visibly full of water up to the rim at the same time. "
        "There must be no second glass, no pitcher, no ice, no straw, and no text."
    ),
    "one_star_two_colors": (
        "A square icon on a plain white background with exactly one single five-point star centered in the image. "
        "The same single star must be entirely red and entirely blue at the same time, with no split halves, no gradient, no outline, and no second star."
    ),
}
