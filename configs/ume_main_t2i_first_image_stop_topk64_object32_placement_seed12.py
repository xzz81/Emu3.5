# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from configs.ume_main_t2i_first_image_stop_probe import *  # noqa: F401,F403

seed = 18077
trace_topk_visual = 64

image_area = 1048576
target_height = 32
target_width = 32
max_new_tokens = 1400
sampling_params["max_new_tokens"] = max_new_tokens

GRID_PROMPT = (
    "Generate exactly one square top-down flat lay image and then stop. Show a pure white background with "
    "one thin light gray vertical line and one thin light gray horizontal line crossing at the image center, "
    "dividing the image into four equal quadrants. Place one {object_phrase} entirely inside the {quadrant} "
    "quadrant. Keep the other three quadrants empty. Use no text."
)

CASES = [
    ("ul_a_red_apple", "red apple", "upper-left"),
    ("ul_a_red_button", "red button", "upper-left"),
    ("ul_a_black_sunglasses", "pair of black sunglasses", "upper-left"),
    ("ul_a_purple_cube", "purple cube", "upper-left"),
    ("ul_b_red_apple", "red apple", "upper-left"),
    ("ul_b_red_button", "red button", "upper-left"),
    ("ul_b_black_sunglasses", "pair of black sunglasses", "upper-left"),
    ("ul_b_purple_cube", "purple cube", "upper-left"),
    ("ur_red_apple", "red apple", "upper-right"),
    ("ur_red_button", "red button", "upper-right"),
    ("ur_black_sunglasses", "pair of black sunglasses", "upper-right"),
    ("ur_purple_cube", "purple cube", "upper-right"),
    ("lr_red_apple", "red apple", "lower-right"),
    ("lr_red_button", "red button", "lower-right"),
    ("lr_black_sunglasses", "pair of black sunglasses", "lower-right"),
    ("lr_purple_cube", "purple cube", "lower-right"),
]

prompts = {
    f"t2i_object32n_gridquota_{case_key}": GRID_PROMPT.format(
        object_phrase=object_phrase,
        quadrant=quadrant,
    )
    for case_key, object_phrase, quadrant in CASES
}
