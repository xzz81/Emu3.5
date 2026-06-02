# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from configs.ume_main_t2i_first_image_stop_probe import *  # noqa: F401,F403

seed = 11027
trace_topk_visual = 64

image_area = 1048576
target_height = 32
target_width = 32
max_new_tokens = 1400
sampling_params["max_new_tokens"] = max_new_tokens

prompts = {
    "t2i_object32g_left_red_apple": (
        "Generate exactly one square image and then stop. Show one large red apple entirely inside the leftmost "
        "third of a clean white tabletop. Keep the center and right two-thirds mostly empty, with a simple light "
        "background and no text."
    ),
    "t2i_object32g_left_blue_cup": (
        "Generate exactly one square image and then stop. Show one large blue cup entirely inside the leftmost "
        "third of a clean white tabletop. Keep the center and right two-thirds mostly empty, with a simple light "
        "background and no text."
    ),
    "t2i_object32g_right_yellow_banana": (
        "Generate exactly one square image and then stop. Show one large yellow banana entirely inside the "
        "rightmost third of a clean white tabletop. Keep the center and left two-thirds mostly empty, with a "
        "simple light background and no text."
    ),
    "t2i_object32g_right_green_bottle": (
        "Generate exactly one square image and then stop. Show one large green bottle entirely inside the "
        "rightmost third of a clean white tabletop. Keep the center and left two-thirds mostly empty, with a "
        "simple light background and no text."
    ),
    "t2i_object32g_upper_left_orange": (
        "Generate exactly one square image and then stop. Show one large orange fruit placed in the upper-left "
        "corner of a clean white tabletop, away from the center. Keep the center mostly empty, with a simple "
        "light background and no text."
    ),
    "t2i_object32g_upper_right_black_hat": (
        "Generate exactly one square image and then stop. Show one large black hat placed in the upper-right "
        "corner of a clean white tabletop, away from the center. Keep the center mostly empty, with a simple "
        "light background and no text."
    ),
    "t2i_object32g_lower_left_purple_box": (
        "Generate exactly one square image and then stop. Show one large purple box placed in the lower-left "
        "corner of a clean white tabletop, away from the center. Keep the center mostly empty, with a simple "
        "light background and no text."
    ),
    "t2i_object32g_lower_right_red_tomato": (
        "Generate exactly one square image and then stop. Show one large red tomato placed in the lower-right "
        "corner of a clean white tabletop, away from the center. Keep the center mostly empty, with a simple "
        "light background and no text."
    ),
}
