# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from configs.ume_main_image_read_entropy_seed21 import *  # noqa: F401,F403


seed = 24023
max_new_tokens = 64
sampling_params["max_new_tokens"] = max_new_tokens

IMAGE_ROOT = Path("assets/ume_image_read_simple_seed23")

CASES = [
    ("ul_red_circle", "red circle", "upper-left"),
    ("ur_red_circle", "red circle", "upper-right"),
    ("ll_red_circle", "red circle", "lower-left"),
    ("lr_red_circle", "red circle", "lower-right"),
    ("ul_blue_square", "blue square", "upper-left"),
    ("ur_blue_square", "blue square", "upper-right"),
    ("ll_blue_square", "blue square", "lower-left"),
    ("lr_blue_square", "blue square", "lower-right"),
    ("ul_yellow_triangle", "yellow triangle", "upper-left"),
    ("ur_yellow_triangle", "yellow triangle", "upper-right"),
    ("ll_yellow_triangle", "yellow triangle", "lower-left"),
    ("lr_yellow_triangle", "yellow triangle", "lower-right"),
]

prompts = {}
for case_key, object_phrase, position_phrase in CASES:
    image_path = str(IMAGE_ROOT / f"{case_key}.png")
    prompts[f"simple_{case_key}__describe"] = {
        "prompt": "Describe this image carefully.",
        "reference_image": image_path,
        "expected_object": object_phrase,
        "expected_position": position_phrase,
    }
    prompts[f"simple_{case_key}__imageonly"] = {
        "prompt": "",
        "reference_image": image_path,
        "expected_object": object_phrase,
        "expected_position": position_phrase,
    }
