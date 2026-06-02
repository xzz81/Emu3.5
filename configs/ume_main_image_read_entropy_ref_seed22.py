# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from configs.ume_main_image_read_entropy_seed21 import *  # noqa: F401,F403


seed = 24022

IMAGE_ROOT = Path("assets/ume_reference_layout_seed19")

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

prompts = {}
for case_key, object_phrase, position_phrase in CASES:
    image_path = str(IMAGE_ROOT / f"{case_key}.png")
    prompts[f"ref_{case_key}__describe"] = {
        "prompt": "Describe this image carefully.",
        "reference_image": image_path,
        "expected_object": object_phrase,
        "expected_position": position_phrase,
    }
    prompts[f"ref_{case_key}__imageonly"] = {
        "prompt": "",
        "reference_image": image_path,
        "expected_object": object_phrase,
        "expected_position": position_phrase,
    }
