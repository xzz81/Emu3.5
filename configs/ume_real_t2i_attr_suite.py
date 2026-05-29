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

seed = 7777

prompts = {
    "grid_color_count": (
        "A clean square educational poster showing exactly four flat colored squares arranged in a 2 by 2 grid. "
        "Top-left is red, top-right is blue, bottom-left is green, and bottom-right is yellow. "
        "Use a plain white background and crisp black outlines."
    ),
    "left_right_negation": (
        "A square studio photograph of a matte blue cube on the left and a glossy red sphere on the right. "
        "There are no green objects anywhere in the image. Use soft shadows on a neutral gray floor."
    ),
    "text_open_24h": (
        "A square close-up product-style image of a small neon shop sign on a dark wall. "
        "The sign must contain the readable text OPEN 24H in bright letters, with no other words."
    ),
    "material_glass_teapot": (
        "A square still life of a transparent glass teapot filled with amber tea, next to a polished silver spoon. "
        "Place both objects on a dark wooden table with warm window light."
    ),
    "stacked_boxes_order": (
        "A square simple illustration of three stacked boxes. "
        "The largest red box is on the bottom, a medium yellow box is in the middle, and the smallest blue box is on top. "
        "Keep the background plain and make the stack vertical."
    ),
    "conflicting_door_state": (
        "A square realistic hallway scene focused on a single door that should look both open and closed at the same time. "
        "Make the contradictory door state visually specific while keeping the rest of the hallway simple."
    ),
}
