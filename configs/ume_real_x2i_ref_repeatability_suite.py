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

blank_panel_prompt = (
    "Use the reference image layout and keep exactly one astronaut. Make the rectangular chest panel completely "
    "blank white with no buttons, dials, icons, letters, numbers, marks, or symbols. Keep the orange torso, "
    "helmet, gloves, and pose unchanged. Do not add text anywhere."
)

visor_conflict_prompt = (
    "Use the reference image layout and keep exactly one astronaut. The same helmet visor must be completely "
    "transparent with the face clearly visible and completely opaque black with no face visible at the same time. "
    "Do not split the visor, add a second visor, add a second astronaut, or add text."
)

prompts = {
    "blank_chest_panel_r1": {"prompt": blank_panel_prompt, "reference_image": reference_image},
    "blank_chest_panel_r2": {"prompt": blank_panel_prompt, "reference_image": reference_image},
    "blank_chest_panel_r3": {"prompt": blank_panel_prompt, "reference_image": reference_image},
    "blank_chest_panel_r4": {"prompt": blank_panel_prompt, "reference_image": reference_image},
    "visor_transparent_and_opaque_r1": {"prompt": visor_conflict_prompt, "reference_image": reference_image},
    "visor_transparent_and_opaque_r2": {"prompt": visor_conflict_prompt, "reference_image": reference_image},
    "visor_transparent_and_opaque_r3": {"prompt": visor_conflict_prompt, "reference_image": reference_image},
    "visor_transparent_and_opaque_r4": {"prompt": visor_conflict_prompt, "reference_image": reference_image},
}
