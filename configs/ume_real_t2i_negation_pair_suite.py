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

seed = 4242

prompts = {
    "blank_stop_sign_shape": (
        "A square realistic street photo of a red octagonal traffic sign on a metal pole. "
        "The sign face must be completely blank: no letters, no symbols, no border text, no word STOP, and no other text anywhere."
    ),
    "stop_sign_positive_control": (
        "A square realistic street photo of a red octagonal traffic sign on a metal pole. "
        "The sign must clearly contain only the readable white word STOP, with no other words or symbols."
    ),
    "cafe_not_coffee": (
        "A square close-up image of a small storefront sign. "
        "The sign must read exactly CAFE in large letters. It must not say COFFEE, CAFÉ, CAFE SHOP, OPEN, or any other word."
    ),
    "empty_plate_no_utensils": (
        "A square top-down studio photo of one empty white ceramic plate centered on a plain gray table. "
        "There must be no food, no fork, no spoon, no knife, no napkin, no cup, and no text."
    ),
    "exactly_three_no_green": (
        "A square minimal studio image with exactly three objects: a red apple on the left, a blue cup in the center, "
        "and a yellow banana on the right. There must be no green object, no leaf, no stem, no fourth object, and no text."
    ),
    "one_person_two_shadows": (
        "A square realistic photo of exactly one standing person under one lamp, but the person casts two clear shadows on the wall. "
        "There must be no second person, no mannequin, no mirror, and no reflection."
    ),
}
