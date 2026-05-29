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

seed = 5151

prompts = {
    "word_cat_spelled_dog": (
        "A square close-up image of a clean white sign. The sign must contain exactly one word: CAT. "
        "However, that same word must be spelled exactly D O G. Do not add any other words, symbols, or small text."
    ),
    "single_shape_circle_square": (
        "A square minimal poster with exactly one single flat geometric shape centered on a white background. "
        "The same single shape must be a perfect circle and a perfect square at the same time. "
        "There must be no second shape, no overlap, and no explanatory text."
    ),
    "empty_bowl_full_soup": (
        "A square top-down studio photo of exactly one white bowl on a plain table. "
        "The bowl must be completely empty and also visibly full of red tomato soup at the same time. "
        "There must be no second bowl, no spoon, no garnish, and no text."
    ),
    "transparent_opaque_cube": (
        "A square studio photo of exactly one glass cube on a gray surface. "
        "The same cube must be fully transparent and completely opaque at the same time. "
        "There must be no second cube, no mirror, no reflection, and no text."
    ),
    "traffic_light_all_off_red_green": (
        "A square realistic photo of exactly one traffic light with three circular lenses. "
        "All lights must be off, but the top red light and bottom green light must both be brightly lit at the same time. "
        "There must be no second traffic light and no text."
    ),
    "one_arrow_left_right": (
        "A square simple road sign with exactly one single black arrow on a white background. "
        "The same single arrow must point left and right at the same time. "
        "There must be no second arrow, no double-headed arrow, and no text."
    ),
}
