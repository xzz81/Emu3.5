# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from configs.ume_main_t2i_first_image_stop_probe import *  # noqa: F401,F403

seed = 19031
trace_topk_visual = 64

image_area = 1048576
target_height = 32
target_width = 32
max_new_tokens = 1400
sampling_params["max_new_tokens"] = max_new_tokens

NATURAL_CORNER_PROMPT = (
    "Generate exactly one square top-down flat lay image and then stop. Use a plain matte light-gray tabletop "
    "with no grid lines, no corner markers, no arrows, and no text. Place one small {object_phrase} entirely "
    "inside the {corner} corner area of the image, far from the center. Keep the rest of the image empty."
)

CASES = [
    ("ul_red_apple", "red apple", "upper-left"),
    ("ul_red_button", "red button", "upper-left"),
    ("ul_black_sunglasses", "pair of black sunglasses", "upper-left"),
    ("ul_purple_cube", "purple cube", "upper-left"),
    ("ur_red_apple", "red apple", "upper-right"),
    ("ur_red_button", "red button", "upper-right"),
    ("ur_black_sunglasses", "pair of black sunglasses", "upper-right"),
    ("ur_purple_cube", "purple cube", "upper-right"),
    ("ll_red_apple", "red apple", "lower-left"),
    ("ll_red_button", "red button", "lower-left"),
    ("ll_black_sunglasses", "pair of black sunglasses", "lower-left"),
    ("ll_purple_cube", "purple cube", "lower-left"),
    ("lr_red_apple", "red apple", "lower-right"),
    ("lr_red_button", "red button", "lower-right"),
    ("lr_black_sunglasses", "pair of black sunglasses", "lower-right"),
    ("lr_purple_cube", "purple cube", "lower-right"),
]

prompts = {
    f"t2i_object32o_naturalcorner_{case_key}": NATURAL_CORNER_PROMPT.format(
        object_phrase=object_phrase,
        corner=corner,
    )
    for case_key, object_phrase, corner in CASES
}
