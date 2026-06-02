# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from configs.ume_main_t2i_first_image_stop_probe import *  # noqa: F401,F403

seed = 10039
trace_topk_visual = 64

image_area = 1048576
target_height = 32
target_width = 32
max_new_tokens = 1400
sampling_params["max_new_tokens"] = max_new_tokens

prompts = {
    "t2i_object32f_left_blue_mug": (
        "Generate exactly one square image and then stop. Show one large plain blue mug placed clearly on the "
        "left side of a clean white table, not centered. The right side of the table should be mostly empty. "
        "Use a simple light background and no text."
    ),
    "t2i_object32f_right_tennis_ball": (
        "Generate exactly one square image and then stop. Show one large yellow tennis ball placed clearly on "
        "the right side of a clean white table, not centered. The left side of the table should be mostly empty. "
        "Use a simple light background and no text."
    ),
    "t2i_object32f_upper_left_watch": (
        "Generate exactly one square image and then stop. Show one large wristwatch placed near the upper-left "
        "area of a clean white table, not centered. The rest of the table should be mostly empty. Use a simple "
        "light background and no text."
    ),
    "t2i_object32f_lower_right_book": (
        "Generate exactly one square image and then stop. Show one large closed red book placed near the "
        "lower-right area of a clean white table, not centered. The rest of the table should be mostly empty. "
        "Use a simple light background and no text."
    ),
}
