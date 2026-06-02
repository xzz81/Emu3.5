# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from configs.ume_main_image_read_entropy_seed21 import *  # noqa: F401,F403


seed = 24030
max_new_tokens = 24
sampling_params["max_new_tokens"] = max_new_tokens
sampling_params["do_sample"] = True
sampling_params["text_temperature"] = 1.0

IMAGE_ROOT = Path("assets/ume_image_read_same_shape_distractor_seed30")

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

repeats = 5

prompts = {}
for rep in range(repeats):
    for case_key, object_phrase, position_phrase in CASES:
        image_path = str(IMAGE_ROOT / f"{case_key}.png")
        prompt = (
            f"Where is the {object_phrase} in the image? "
            "Answer with exactly one of these labels: upper-left, upper-right, lower-left, lower-right."
        )
        prompts[f"sameshape_rep{rep:02d}_{case_key}__forcedposition"] = {
            "prompt": prompt,
            "reference_image": image_path,
            "expected_object": object_phrase,
            "expected_position": position_phrase,
            "repeat": rep,
        }
