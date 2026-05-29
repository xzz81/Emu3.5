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

seed = 20260525

reference_image = "./assets/ref_img.png"

prompts = {
    "astronaut_top_red_border": {
        "prompt": (
            "Use the reference image layout and keep exactly one astronaut in the same pose. Add one thin bright red "
            "horizontal stripe that touches the very top image border from left to right. Do not add any bottom stripe, "
            "text, corner dots, or second astronaut. Keep the astronaut body and chest panel otherwise unchanged."
        ),
        "reference_image": reference_image,
    },
    "astronaut_four_blue_corner_dots": {
        "prompt": (
            "Use the reference image layout and keep exactly one astronaut in the same pose. Add exactly four small blue "
            "dots, one dot at each image corner. Do not add dots anywhere else, text, border lines, or a second astronaut."
        ),
        "reference_image": reference_image,
    },
    "astronaut_black_outer_frame": {
        "prompt": (
            "Use the reference image layout and keep exactly one astronaut in the same pose. Add a thin solid black frame "
            "touching all four outer image borders. The frame must be only at the outer edge. Do not add inner boxes, "
            "text, dots, or a second astronaut."
        ),
        "reference_image": reference_image,
    },
    "astronaut_three_background_bands": {
        "prompt": (
            "Use the reference image layout and keep exactly one astronaut in the same pose. Replace only the background "
            "behind the astronaut with exactly three horizontal color bands: green top band, white middle band, and purple "
            "bottom band. Keep the astronaut visible in front. Do not add text, border lines, dots, or a second astronaut."
        ),
        "reference_image": reference_image,
    },
}
