# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from configs.ume_main_t2i_first_image_stop_probe import *  # noqa: F401,F403

seed = 21015
trace_topk_visual = 64

image_area = 1048576
target_height = 32
target_width = 32
max_new_tokens = 1400
sampling_params["max_new_tokens"] = max_new_tokens

CONTROLLED_CORNER_PROMPT = (
    "Generate exactly one square top-down flat lay image and then stop. Use a perfectly uniform matte neutral-gray "
    "background with no grid lines, no corner markers, no arrows, no text, no borders, no reflections, and only a "
    "very soft contact shadow. Place exactly one small flat matte {object_phrase}. The object should have a similar "
    "apparent size in every image, about one fifth of the image width. Put the center of the object in the {corner} "
    "corner safety zone, roughly {coordinate_hint}, entirely inside the image and far from the center. Leave the rest "
    "of the image empty."
)

CASES = [
    ("ul_red_button", "red button", "upper-left", "one quarter from the left edge and one quarter from the top edge"),
    ("ur_red_button", "red button", "upper-right", "one quarter from the right edge and one quarter from the top edge"),
    ("ll_red_button", "red button", "lower-left", "one quarter from the left edge and one quarter from the bottom edge"),
    ("lr_red_button", "red button", "lower-right", "one quarter from the right edge and one quarter from the bottom edge"),
    ("ul_purple_cube", "purple cube", "upper-left", "one quarter from the left edge and one quarter from the top edge"),
    ("ur_purple_cube", "purple cube", "upper-right", "one quarter from the right edge and one quarter from the top edge"),
    ("ll_purple_cube", "purple cube", "lower-left", "one quarter from the left edge and one quarter from the bottom edge"),
    ("lr_purple_cube", "purple cube", "lower-right", "one quarter from the right edge and one quarter from the bottom edge"),
    ("ul_black_sunglasses", "pair of black sunglasses", "upper-left", "one quarter from the left edge and one quarter from the top edge"),
    ("ur_black_sunglasses", "pair of black sunglasses", "upper-right", "one quarter from the right edge and one quarter from the top edge"),
    ("ll_black_sunglasses", "pair of black sunglasses", "lower-left", "one quarter from the left edge and one quarter from the bottom edge"),
    ("lr_black_sunglasses", "pair of black sunglasses", "lower-right", "one quarter from the right edge and one quarter from the bottom edge"),
    ("ul_yellow_banana", "yellow banana", "upper-left", "one quarter from the left edge and one quarter from the top edge"),
    ("ur_yellow_banana", "yellow banana", "upper-right", "one quarter from the right edge and one quarter from the top edge"),
    ("ll_yellow_banana", "yellow banana", "lower-left", "one quarter from the left edge and one quarter from the bottom edge"),
    ("lr_yellow_banana", "yellow banana", "lower-right", "one quarter from the right edge and one quarter from the bottom edge"),
]

prompts = {
    f"t2i_object32q_controlcorner_{case_key}": CONTROLLED_CORNER_PROMPT.format(
        object_phrase=object_phrase,
        corner=corner,
        coordinate_hint=coordinate_hint,
    )
    for case_key, object_phrase, corner, coordinate_hint in CASES
}
