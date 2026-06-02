# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from configs.ume_main_t2i_first_image_stop_probe import *  # noqa: F401,F403

seed = 12031
trace_topk_visual = 64

image_area = 1048576
target_height = 32
target_width = 32
max_new_tokens = 1400
sampling_params["max_new_tokens"] = max_new_tokens

prompts = {
    "t2i_object32h_upper_left_gold_key": (
        "Generate exactly one square top-down flat lay image and then stop. On a pure white tabletop, place one "
        "large gold key entirely inside the upper-left quadrant of the image. Leave the center, right side, and "
        "bottom half empty. Use no text."
    ),
    "t2i_object32h_upper_left_blue_pen": (
        "Generate exactly one square top-down flat lay image and then stop. On a pure white tabletop, place one "
        "large blue pen entirely inside the upper-left quadrant of the image. Leave the center, right side, and "
        "bottom half empty. Use no text."
    ),
    "t2i_object32h_upper_right_red_button": (
        "Generate exactly one square top-down flat lay image and then stop. On a pure white tabletop, place one "
        "large red button entirely inside the upper-right quadrant of the image. Leave the center, left side, and "
        "bottom half empty. Use no text."
    ),
    "t2i_object32h_upper_right_green_leaf": (
        "Generate exactly one square top-down flat lay image and then stop. On a pure white tabletop, place one "
        "large green leaf entirely inside the upper-right quadrant of the image. Leave the center, left side, and "
        "bottom half empty. Use no text."
    ),
    "t2i_object32h_lower_left_yellow_cookie": (
        "Generate exactly one square top-down flat lay image and then stop. On a pure white tabletop, place one "
        "large round yellow cookie entirely inside the lower-left quadrant of the image. Leave the center, right "
        "side, and top half empty. Use no text."
    ),
    "t2i_object32h_lower_left_black_sunglasses": (
        "Generate exactly one square top-down flat lay image and then stop. On a pure white tabletop, place one "
        "large pair of black sunglasses entirely inside the lower-left quadrant of the image. Leave the center, "
        "right side, and top half empty. Use no text."
    ),
    "t2i_object32h_lower_right_white_soap": (
        "Generate exactly one square top-down flat lay image and then stop. On a pure white tabletop, place one "
        "large white bar of soap entirely inside the lower-right quadrant of the image. Leave the center, left "
        "side, and top half empty. Use no text."
    ),
    "t2i_object32h_lower_right_orange_carrot": (
        "Generate exactly one square top-down flat lay image and then stop. On a pure white tabletop, place one "
        "large orange carrot entirely inside the lower-right quadrant of the image. Leave the center, left side, "
        "and top half empty. Use no text."
    ),
}
