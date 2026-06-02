# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from src.utils.logging_utils import setup_logger


cfg_name = Path(__file__).stem

model_path = "/workspace/home/AAAI 2027/models/BAAI/Emu3.5"
vq_path = "/workspace/home/AAAI 2027/models/BAAI/Emu3.5-VisionTokenizer"
tokenizer_path = "./src/tokenizer_emu3_ibq"
vq_type = "ibq"

task_type = "x2i"
use_image = True

exp_name = "emu3p5-main"
save_path = f"./outputs/{exp_name}/{task_type}"
save_to_proto = True
setup_logger(save_path)

hf_device = "auto"
vq_device = "cuda:0"
streaming = False
unconditional_type = "no_text"
classifier_free_guidance = 2.0

image_area = 1048576
target_height = 32
target_width = 32
max_new_tokens = 1400
stop_after_completed_images = 1
stop_after_eoi_extra_tokens = 0
trace_topk_visual = 64


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
    text_top_k=512,
    text_top_p=0.85,
    text_temperature=0.8,
    image_top_k=10240,
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

seed = 24019

REFERENCE_ROOT = Path("assets/ume_reference_layout_seed19")

PROMPT_TEMPLATE = (
    "Use the reference image as a precise layout guide. Generate exactly one square top-down flat lay image "
    "and then stop. Recreate the plain neutral gray tabletop and exactly one small matte {object_phrase} at "
    "the same {position_phrase} corner position as in the reference image. Keep the object clearly away from "
    "the center of the image. Do not add grid lines, boxes, arrows, labels, readable text, borders, corner "
    "markers, extra objects, or decorations."
)

CASES = [
    ("ul_red_button", "red button", "upper-left"),
    ("ur_red_button", "red button", "upper-right"),
    ("ll_purple_cube", "purple cube", "lower-left"),
    ("lr_purple_cube", "purple cube", "lower-right"),
    ("ul_yellow_token", "yellow disk", "upper-left"),
    ("ur_yellow_token", "yellow disk", "upper-right"),
    ("ll_blue_cube", "blue cube", "lower-left"),
    ("lr_blue_cube", "blue cube", "lower-right"),
]

prompts = {
    f"x2i_ref_layout_{case_key}": {
        "prompt": PROMPT_TEMPLATE.format(object_phrase=object_phrase, position_phrase=position_phrase),
        "reference_image": str(REFERENCE_ROOT / f"{case_key}.png"),
    }
    for case_key, object_phrase, position_phrase in CASES
}
