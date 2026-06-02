# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from configs.ume_main_t2i_first_image_stop_probe import *  # noqa: F401,F403

seed = 22016
trace_topk_visual = 64

image_area = 1048576
target_height = 32
target_width = 32
max_new_tokens = 1400
sampling_params["max_new_tokens"] = max_new_tokens

SIMPLE_EDGE_PROMPT = (
    "Generate exactly one square top-down flat lay image and then stop. Use a plain matte neutral-gray tabletop "
    "with no lines, no arrows, no text, no labels, and no decorations. Place exactly one small matte {object_phrase} "
    "{place_phrase}. Keep it clearly away from the center of the image. Make the object about one fifth of the image "
    "width, fully visible, and leave the rest of the tabletop empty."
)

CASES = [
    ("ul_red_button", "red button", "near the top-left edge"),
    ("ur_red_button", "red button", "near the top-right edge"),
    ("ll_red_button", "red button", "near the bottom-left edge"),
    ("lr_red_button", "red button", "near the bottom-right edge"),
    ("ul_purple_cube", "purple cube", "near the top-left edge"),
    ("ur_purple_cube", "purple cube", "near the top-right edge"),
    ("ll_purple_cube", "purple cube", "near the bottom-left edge"),
    ("lr_purple_cube", "purple cube", "near the bottom-right edge"),
    ("ul_black_sunglasses", "pair of black sunglasses", "near the top-left edge"),
    ("ur_black_sunglasses", "pair of black sunglasses", "near the top-right edge"),
    ("ll_black_sunglasses", "pair of black sunglasses", "near the bottom-left edge"),
    ("lr_black_sunglasses", "pair of black sunglasses", "near the bottom-right edge"),
    ("ul_yellow_banana", "yellow banana", "near the top-left edge"),
    ("ur_yellow_banana", "yellow banana", "near the top-right edge"),
    ("ll_yellow_banana", "yellow banana", "near the bottom-left edge"),
    ("lr_yellow_banana", "yellow banana", "near the bottom-right edge"),
]

prompts = {
    f"t2i_object32r_simpleedge_{case_key}": SIMPLE_EDGE_PROMPT.format(
        object_phrase=object_phrase,
        place_phrase=place_phrase,
    )
    for case_key, object_phrase, place_phrase in CASES
}
