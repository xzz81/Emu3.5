# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from configs.ume_main_t2i_first_image_stop_probe import *  # noqa: F401,F403

seed = 9893
trace_topk_visual = 64

image_area = 1048576
target_height = 32
target_width = 32
max_new_tokens = 1400
sampling_params["max_new_tokens"] = max_new_tokens

prompts = {
    "t2i_object32_red_apple": (
        "Generate exactly one square image and then stop. Show one large red apple centered on a clean white "
        "table. The apple should be the only main object, with a simple light background and no text."
    ),
    "t2i_object32_blue_cup": (
        "Generate exactly one square image and then stop. Show one large blue ceramic cup centered on a clean "
        "white table. The cup should be the only main object, with a simple light background and no text."
    ),
    "t2i_object32_yellow_banana": (
        "Generate exactly one square image and then stop. Show one large yellow banana centered on a clean "
        "white table. The banana should be the only main object, with a simple light background and no text."
    ),
    "t2i_object32_green_cactus": (
        "Generate exactly one square image and then stop. Show one large green cactus centered in a small brown "
        "pot on a clean white table. Use a simple light background and no text."
    ),
}
