# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from src.utils.logging_utils import setup_logger


cfg_name = Path(__file__).stem

model_path = "model/Emu3.5"
vq_path = "model/Emu3.5-VisionTokenizer"
tokenizer_path = "./src/tokenizer_emu3_ibq"
vq_type = "ibq"

task_type = "image_read_entropy"
use_image = True

exp_name = "emu3p5-main"
save_path = f"./outputs/{exp_name}/{task_type}"
save_to_proto = True
setup_logger(save_path)

hf_device = "auto"
vq_device = "cuda:0"
streaming = False
unconditional_type = "no_text"
classifier_free_guidance = 1.0

image_area = 262144
max_new_tokens = 80


def build_unc_and_template(task: str, with_image: bool):
    if with_image:
        unc_p = "<|extra_203|>You are a helpful assistant. USER: <|IMAGE|> ASSISTANT: <|extra_100|>"
        tmpl = "<|extra_203|>You are a helpful assistant. USER: <|IMAGE|>{question} ASSISTANT: <|extra_100|>"
    else:
        unc_p = "<|extra_203|>You are a helpful assistant. USER:  ASSISTANT: <|extra_100|>"
        tmpl = "<|extra_203|>You are a helpful assistant. USER: {question} ASSISTANT: <|extra_100|>"
    return unc_p, tmpl


unc_prompt, template = build_unc_and_template(task_type, use_image)

sampling_params = dict(
    use_cache=True,
    text_top_k=1024,
    text_top_p=0.9,
    text_temperature=1.0,
    max_new_tokens=max_new_tokens,
    do_sample=True,
    num_beams=1,
    pad_token_id=None,
    eos_token_id=None,
)

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

seed = 24021

IMAGE_ROOT = Path("outputs/emu3p5-main/x2i/ume_trace_runs/main_x2i_ref_copy_seed20_20260531/decoded")

CASES = [
    "x2i_ref_copy_ll_blue_cube",
    "x2i_ref_copy_ll_purple_cube",
    "x2i_ref_copy_lr_blue_cube",
    "x2i_ref_copy_lr_purple_cube",
    "x2i_ref_copy_ul_red_button",
    "x2i_ref_copy_ul_yellow_token",
    "x2i_ref_copy_ur_red_button",
    "x2i_ref_copy_ur_yellow_token",
]

prompts = {}
for case_key in CASES:
    image_path = str(IMAGE_ROOT / f"{case_key}_image_00.png")
    prompts[f"{case_key}__describe"] = {
        "prompt": "Describe this image carefully.",
        "reference_image": image_path,
    }
    prompts[f"{case_key}__imageonly"] = {
        "prompt": "",
        "reference_image": image_path,
    }
