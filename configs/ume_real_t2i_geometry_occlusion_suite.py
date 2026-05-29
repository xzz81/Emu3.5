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
    "five_coins_pentagon": (
        "A square top-down studio photo on a plain white table. Place exactly five identical gold coins at the five vertices "
        "of a regular pentagon, with an empty center. There must be no sixth coin, no missing coin, no text, and no other object."
    ),
    "red_cube_behind_glass": (
        "A square realistic studio photo of a transparent drinking glass in the center and a small red cube clearly behind "
        "the glass, visible through the glass. There must be no second cube, no text, no reflection, and no other object."
    ),
    "mirror_without_reflection": (
        "A square studio photo of exactly one black vase standing directly in front of a wall mirror. The mirror must show "
        "no reflection of the vase at all. Do not add a curtain, blur, second vase, text, or broken mirror."
    ),
    "barcode_no_bars": (
        "A square close-up photo of one white product box with a single label area. The label area must be completely blank "
        "with no letters, numbers, symbols, or barcode bars. The same label area must also contain exactly one black barcode "
        "with vertical bars. Do not add a second label."
    ),
    "rope_over_under_same_ring": (
        "A square close-up photo of one tan rope crossing one silver ring exactly once. At the single crossing point, the same "
        "rope must pass over the ring and under the ring at the same time. There must be no second rope, no knot, no text, and no extra ring."
    ),
    "hidden_word_fully_covered": (
        "A square close-up photo of a white card with the word HIDDEN printed in black. An opaque yellow sticky note must fully "
        "cover the entire word so that no letter is visible. The word HIDDEN must also remain clearly readable. Do not add a second card or note."
    ),
}
