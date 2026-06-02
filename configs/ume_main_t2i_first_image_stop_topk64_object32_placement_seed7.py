# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from configs.ume_main_t2i_first_image_stop_probe import *  # noqa: F401,F403

seed = 13043
trace_topk_visual = 64

image_area = 1048576
target_height = 32
target_width = 32
max_new_tokens = 1400
sampling_params["max_new_tokens"] = max_new_tokens

prompts = {
    "t2i_object32i_grid_ul_red_apple": (
        "Generate exactly one square top-down flat lay image and then stop. Show a pure white background with "
        "one thin light gray vertical line and one thin light gray horizontal line crossing at the image center, "
        "dividing the image into four equal quadrants. Place one red apple entirely inside the upper-left "
        "quadrant. Keep the other three quadrants empty. Use no text."
    ),
    "t2i_object32i_grid_ul_blue_pen": (
        "Generate exactly one square top-down flat lay image and then stop. Show a pure white background with "
        "one thin light gray vertical line and one thin light gray horizontal line crossing at the image center, "
        "dividing the image into four equal quadrants. Place one blue pen entirely inside the upper-left "
        "quadrant. Keep the other three quadrants empty. Use no text."
    ),
    "t2i_object32i_grid_ur_green_leaf": (
        "Generate exactly one square top-down flat lay image and then stop. Show a pure white background with "
        "one thin light gray vertical line and one thin light gray horizontal line crossing at the image center, "
        "dividing the image into four equal quadrants. Place one green leaf entirely inside the upper-right "
        "quadrant. Keep the other three quadrants empty. Use no text."
    ),
    "t2i_object32i_grid_ur_red_button": (
        "Generate exactly one square top-down flat lay image and then stop. Show a pure white background with "
        "one thin light gray vertical line and one thin light gray horizontal line crossing at the image center, "
        "dividing the image into four equal quadrants. Place one red button entirely inside the upper-right "
        "quadrant. Keep the other three quadrants empty. Use no text."
    ),
    "t2i_object32i_grid_ll_black_sunglasses": (
        "Generate exactly one square top-down flat lay image and then stop. Show a pure white background with "
        "one thin light gray vertical line and one thin light gray horizontal line crossing at the image center, "
        "dividing the image into four equal quadrants. Place one pair of black sunglasses entirely inside the "
        "lower-left quadrant. Keep the other three quadrants empty. Use no text."
    ),
    "t2i_object32i_grid_ll_yellow_cookie": (
        "Generate exactly one square top-down flat lay image and then stop. Show a pure white background with "
        "one thin light gray vertical line and one thin light gray horizontal line crossing at the image center, "
        "dividing the image into four equal quadrants. Place one round yellow cookie entirely inside the "
        "lower-left quadrant. Keep the other three quadrants empty. Use no text."
    ),
    "t2i_object32i_grid_lr_orange_carrot": (
        "Generate exactly one square top-down flat lay image and then stop. Show a pure white background with "
        "one thin light gray vertical line and one thin light gray horizontal line crossing at the image center, "
        "dividing the image into four equal quadrants. Place one orange carrot entirely inside the lower-right "
        "quadrant. Keep the other three quadrants empty. Use no text."
    ),
    "t2i_object32i_grid_lr_purple_cube": (
        "Generate exactly one square top-down flat lay image and then stop. Show a pure white background with "
        "one thin light gray vertical line and one thin light gray horizontal line crossing at the image center, "
        "dividing the image into four equal quadrants. Place one purple cube entirely inside the lower-right "
        "quadrant. Keep the other three quadrants empty. Use no text."
    ),
}
