# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from configs.ume_main_t2i_first_image_stop_probe import *  # noqa: F401,F403

seed = 20037
trace_topk_visual = 64

image_area = 1048576
target_height = 32
target_width = 32
max_new_tokens = 1400
sampling_params["max_new_tokens"] = max_new_tokens

LOW_CONTRAST_CORNER_PROMPT = (
    "Generate exactly one square top-down flat lay image and then stop. Use a uniform matte medium-gray "
    "background with no grid lines, no corner markers, no arrows, no text, no reflections, and almost no shadow. "
    "Place one small flat matte {object_phrase} entirely inside the {corner} corner area, far from the center. "
    "Make the object clearly recognizable by shape and color, but keep the lighting soft and low-contrast. "
    "Keep the rest of the image empty."
)

CASES = [
    ("ll_a_red_button", "red button", "lower-left"),
    ("ll_a_purple_cube", "purple cube", "lower-left"),
    ("ll_a_red_apple", "red apple", "lower-left"),
    ("ll_a_black_sunglasses", "pair of black sunglasses", "lower-left"),
    ("ll_b_red_button", "red button", "lower-left"),
    ("ll_b_purple_cube", "purple cube", "lower-left"),
    ("lr_a_red_button", "red button", "lower-right"),
    ("lr_a_purple_cube", "purple cube", "lower-right"),
    ("lr_a_red_apple", "red apple", "lower-right"),
    ("lr_a_black_sunglasses", "pair of black sunglasses", "lower-right"),
    ("lr_b_red_button", "red button", "lower-right"),
    ("lr_b_purple_cube", "purple cube", "lower-right"),
    ("ul_red_button", "red button", "upper-left"),
    ("ul_purple_cube", "purple cube", "upper-left"),
    ("ur_red_button", "red button", "upper-right"),
    ("ur_purple_cube", "purple cube", "upper-right"),
]

prompts = {
    f"t2i_object32p_lowcontrastcorner_{case_key}": LOW_CONTRAST_CORNER_PROMPT.format(
        object_phrase=object_phrase,
        corner=corner,
    )
    for case_key, object_phrase, corner in CASES
}
