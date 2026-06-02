# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from configs.ume_main_t2i_first_image_stop_probe import *  # noqa: F401,F403

seed = 14057
trace_topk_visual = 64

image_area = 1048576
target_height = 32
target_width = 32
max_new_tokens = 1400
sampling_params["max_new_tokens"] = max_new_tokens

prompts = {
    "t2i_object32j_marker_ul_red_apple": (
        "Generate exactly one square top-down flat lay image and then stop. Show a pure white background with a "
        "small pale blue square marker in the upper-left corner only. Place one red apple entirely on that "
        "upper-left marker. Leave the rest of the image empty. Use no text."
    ),
    "t2i_object32j_marker_ul_blue_pen": (
        "Generate exactly one square top-down flat lay image and then stop. Show a pure white background with a "
        "small pale blue square marker in the upper-left corner only. Place one blue pen entirely on that "
        "upper-left marker. Leave the rest of the image empty. Use no text."
    ),
    "t2i_object32j_marker_ur_green_leaf": (
        "Generate exactly one square top-down flat lay image and then stop. Show a pure white background with a "
        "small pale blue square marker in the upper-right corner only. Place one green leaf entirely on that "
        "upper-right marker. Leave the rest of the image empty. Use no text."
    ),
    "t2i_object32j_marker_ur_red_button": (
        "Generate exactly one square top-down flat lay image and then stop. Show a pure white background with a "
        "small pale blue square marker in the upper-right corner only. Place one red button entirely on that "
        "upper-right marker. Leave the rest of the image empty. Use no text."
    ),
    "t2i_object32j_marker_ll_black_sunglasses": (
        "Generate exactly one square top-down flat lay image and then stop. Show a pure white background with a "
        "small pale blue square marker in the lower-left corner only. Place one pair of black sunglasses entirely "
        "on that lower-left marker. Leave the rest of the image empty. Use no text."
    ),
    "t2i_object32j_marker_ll_yellow_cookie": (
        "Generate exactly one square top-down flat lay image and then stop. Show a pure white background with a "
        "small pale blue square marker in the lower-left corner only. Place one round yellow cookie entirely on "
        "that lower-left marker. Leave the rest of the image empty. Use no text."
    ),
    "t2i_object32j_marker_lr_orange_carrot": (
        "Generate exactly one square top-down flat lay image and then stop. Show a pure white background with a "
        "small pale blue square marker in the lower-right corner only. Place one orange carrot entirely on that "
        "lower-right marker. Leave the rest of the image empty. Use no text."
    ),
    "t2i_object32j_marker_lr_purple_cube": (
        "Generate exactly one square top-down flat lay image and then stop. Show a pure white background with a "
        "small pale blue square marker in the lower-right corner only. Place one purple cube entirely on that "
        "lower-right marker. Leave the rest of the image empty. Use no text."
    ),
}
