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

reference_pair = ["./assets/ref_0.png", "./assets/ref_1.png"]

prompts = {
    "two_blue_strawberries": {
        "prompt": (
            "Use the first reference image as the scene layout and lighting. Replace the central brown truffle "
            "with exactly two blue strawberries side by side on the white cloth. The strawberries should have the "
            "shape and leaves of the strawberry in the second reference image. There must be no brown truffle, no "
            "red strawberry, no third object, and no text."
        ),
        "reference_image": reference_pair,
    },
    "striped_truffle_three_bands": {
        "prompt": (
            "Use the first reference image as the scene layout and lighting. Keep exactly one central brown truffle "
            "on the white cloth, but paint its visible surface with exactly three clean horizontal color bands: red "
            "on top, white in the middle, and blue on the bottom. Do not add a tag, fruit, text, or a second object."
        ),
        "reference_image": reference_pair,
    },
    "three_objects_order": {
        "prompt": (
            "Use the white cloth and garden lighting from the first reference image. Place exactly three objects in "
            "one row on the cloth from left to right: the brown truffle, the red strawberry from the second reference, "
            "and the brown truffle again. Do not add any fourth object, label, or text."
        ),
        "reference_image": reference_pair,
    },
    "empty_cloth_preserve_shadow": {
        "prompt": (
            "Use the first reference image as the scene layout and lighting. Remove the central brown truffle and all "
            "visible crumbs completely, leaving only the white cloth, table, garden background, and natural shadows. "
            "Do not add fruit, candy, balls, labels, or text."
        ),
        "reference_image": reference_pair,
    },
    "strawberry_and_no_strawberry": {
        "prompt": (
            "Use the first reference image as the scene layout and lighting. Replace the central truffle with one ripe "
            "red strawberry from the second reference image, but the final image must contain no strawberry, no red fruit, "
            "and no strawberry-shaped object anywhere. Do not leave the truffle."
        ),
        "reference_image": reference_pair,
    },
    "single_object_truffle_strawberry": {
        "prompt": (
            "Use the white cloth and garden lighting from the first reference image. Put exactly one single object on "
            "the cloth. This same single object must be clearly the brown truffle from the first reference and clearly "
            "the red strawberry from the second reference at the same time. Do not use split halves, blending, duplicates, "
            "reflections, labels, or text."
        ),
        "reference_image": reference_pair,
    },
}
