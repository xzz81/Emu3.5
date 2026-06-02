# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from configs.ume_main_t2i_first_image_stop_probe import *  # noqa: F401,F403

seed = 9896
trace_topk_visual = 64

image_area = 1048576
target_height = 32
target_width = 32
max_new_tokens = 1400
sampling_params["max_new_tokens"] = max_new_tokens

prompts = {
    "t2i_object32d_soccer_ball": (
        "Generate exactly one square image and then stop. Show one large soccer ball centered on a clean "
        "white table. The soccer ball should be the only main object, with a simple light background and no text."
    ),
    "t2i_object32d_orange_pumpkin": (
        "Generate exactly one square image and then stop. Show one large orange pumpkin centered on a clean "
        "white table. The pumpkin should be the only main object, with a simple light background and no text."
    ),
    "t2i_object32d_blue_backpack": (
        "Generate exactly one square image and then stop. Show one large blue backpack centered on a clean "
        "white table. The backpack should be the only main object, with a simple light background and no text."
    ),
    "t2i_object32d_black_umbrella": (
        "Generate exactly one square image and then stop. Show one large closed black umbrella centered on a clean "
        "white table. The umbrella should be the only main object, with a simple light background and no text."
    ),
}
