# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from src.utils.logging_utils import setup_logger

cfg_name = Path(__file__).stem

model_path = "model/Emu3.5"
vq_path = "model/Emu3.5-VisionTokenizer"
tokenizer_path = "./src/tokenizer_emu3_ibq"
vq_type = "ibq"

task_type = "howto"
use_image = False

exp_name = "emu3p5-main"
save_path = f"./outputs/{exp_name}/{task_type}"
save_to_proto = True
setup_logger(save_path)

hf_device = "auto"
vq_device = "cuda:0"
streaming = False
unconditional_type = "no_text"
classifier_free_guidance = 3.0
max_new_tokens = 1200
image_area = 65536
target_height = 16
target_width = 16


def build_unc_and_template(task: str, with_image: bool):
    task_str = task.lower()
    extra_system_prompt = " Please generate a response with one interleaved image and concise text."
    if with_image:
        unc_p = "<|extra_203|>You are a helpful assistant. USER: <|IMAGE|> ASSISTANT: <|extra_100|>"
        tmpl = "<|extra_203|>You are a helpful assistant for %s task.%s USER: {question}<|IMAGE|> ASSISTANT: <|extra_100|>" % (task_str, extra_system_prompt)
    else:
        unc_p = "<|extra_203|>You are a helpful assistant. USER:  ASSISTANT: <|extra_100|>"
        tmpl = "<|extra_203|>You are a helpful assistant for %s task.%s USER: {question} ASSISTANT: <|extra_100|>" % (task_str, extra_system_prompt)
    return unc_p, tmpl


unc_prompt, template = build_unc_and_template(task_type, use_image)

sampling_params = dict(
    use_cache=True,
    text_top_k=200,
    text_top_p=0.8,
    text_temperature=0.7,
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

seed = 9092

prompts = {
    "howto_one_step_teapot": (
        "Show exactly one illustrated step: place a small transparent glass teapot and two clear cups on a "
        "wooden tray. After the image, write one short instruction sentence, then stop."
    ),
    "howto_one_step_seedling": (
        "Show exactly one illustrated step: put a tiny green seedling into a red clay pot on a white table. "
        "After the image, write one short instruction sentence, then stop."
    ),
}
