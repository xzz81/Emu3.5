# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from configs.ume_main_t2i_first_image_stop_probe import *  # noqa: F401,F403

seed = 9897
trace_topk_visual = 64

image_area = 1048576
target_height = 32
target_width = 32
max_new_tokens = 1400
sampling_params["max_new_tokens"] = max_new_tokens

prompts = {
    "t2i_object32e_white_teapot": (
        "Generate exactly one square image and then stop. Show one large white teapot centered on a clean "
        "white table. The teapot should be the only main object, with a simple light background and no text."
    ),
    "t2i_object32e_red_alarm_clock": (
        "Generate exactly one square image and then stop. Show one large red alarm clock centered on a clean "
        "white table. The alarm clock should be the only main object, with a simple light background and no text."
    ),
    "t2i_object32e_green_watermelon": (
        "Generate exactly one square image and then stop. Show one large green watermelon centered on a clean "
        "white table. The watermelon should be the only main object, with a simple light background and no text."
    ),
    "t2i_object32e_yellow_lemon": (
        "Generate exactly one square image and then stop. Show one large yellow lemon centered on a clean "
        "white table. The lemon should be the only main object, with a simple light background and no text."
    ),
}
