# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from configs.ume_main_t2i_first_image_stop_probe import *  # noqa: F401,F403

seed = 16067
trace_topk_visual = 64

image_area = 1048576
target_height = 32
target_width = 32
max_new_tokens = 1400
sampling_params["max_new_tokens"] = max_new_tokens

GRID_PROMPT = (
    "Generate exactly one square top-down flat lay image and then stop. Show a pure white background with "
    "one thin light gray vertical line and one thin light gray horizontal line crossing at the image center, "
    "dividing the image into four equal quadrants. Place one small {object_phrase} in the center of the "
    "{quadrant} quadrant. The object should take up about one eighth of the image width. Leave a wide empty "
    "white margin between the object and the central grid lines; the object must not touch or cross any grid "
    "line. Keep the other three quadrants empty. Use no text."
)

prompts = {
    "t2i_object32l_gridsmall_ul_red_apple": GRID_PROMPT.format(
        object_phrase="red apple",
        quadrant="upper-left",
    ),
    "t2i_object32l_gridsmall_ul_blue_cup": GRID_PROMPT.format(
        object_phrase="blue cup",
        quadrant="upper-left",
    ),
    "t2i_object32l_gridsmall_ur_green_pear": GRID_PROMPT.format(
        object_phrase="green pear",
        quadrant="upper-right",
    ),
    "t2i_object32l_gridsmall_ur_red_button": GRID_PROMPT.format(
        object_phrase="red button",
        quadrant="upper-right",
    ),
    "t2i_object32l_gridsmall_ll_black_sunglasses": GRID_PROMPT.format(
        object_phrase="pair of black sunglasses",
        quadrant="lower-left",
    ),
    "t2i_object32l_gridsmall_ll_yellow_star": GRID_PROMPT.format(
        object_phrase="yellow star sticker",
        quadrant="lower-left",
    ),
    "t2i_object32l_gridsmall_lr_orange_disk": GRID_PROMPT.format(
        object_phrase="orange disk",
        quadrant="lower-right",
    ),
    "t2i_object32l_gridsmall_lr_purple_cube": GRID_PROMPT.format(
        object_phrase="purple cube",
        quadrant="lower-right",
    ),
}
