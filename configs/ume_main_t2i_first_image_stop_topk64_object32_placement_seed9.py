# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from configs.ume_main_t2i_first_image_stop_probe import *  # noqa: F401,F403

seed = 15061
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
    "quadrant, centered within that quadrant and far from the central grid lines. Keep the other three "
    "quadrants empty. Use no text."
)

prompts = {
    "t2i_object32k_grid_ul_red_tomato": GRID_PROMPT.format(
        object_phrase="red tomato",
        quadrant="upper-left",
    ),
    "t2i_object32k_grid_ul_blue_mug": GRID_PROMPT.format(
        object_phrase="blue mug",
        quadrant="upper-left",
    ),
    "t2i_object32k_grid_ur_green_pear": GRID_PROMPT.format(
        object_phrase="green pear",
        quadrant="upper-right",
    ),
    "t2i_object32k_grid_ur_red_button": GRID_PROMPT.format(
        object_phrase="red button",
        quadrant="upper-right",
    ),
    "t2i_object32k_grid_ll_black_sunglasses": GRID_PROMPT.format(
        object_phrase="pair of black sunglasses",
        quadrant="lower-left",
    ),
    "t2i_object32k_grid_ll_yellow_star": GRID_PROMPT.format(
        object_phrase="yellow star-shaped sticker",
        quadrant="lower-left",
    ),
    "t2i_object32k_grid_lr_orange_ball": GRID_PROMPT.format(
        object_phrase="orange ball",
        quadrant="lower-right",
    ),
    "t2i_object32k_grid_lr_purple_cube": GRID_PROMPT.format(
        object_phrase="purple cube",
        quadrant="lower-right",
    ),
}
