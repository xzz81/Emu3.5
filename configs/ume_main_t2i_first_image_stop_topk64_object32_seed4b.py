# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from configs.ume_main_t2i_first_image_stop_probe import *  # noqa: F401,F403

seed = 9894
trace_topk_visual = 64

image_area = 1048576
target_height = 32
target_width = 32
max_new_tokens = 1400
sampling_params["max_new_tokens"] = max_new_tokens

prompts = {
    "t2i_object32b_orange_carrot": (
        "Generate exactly one square image and then stop. Show one large orange carrot centered on a clean "
        "white table. The carrot should be the only main object, with a simple light background and no text."
    ),
    "t2i_object32b_purple_flower": (
        "Generate exactly one square image and then stop. Show one large purple flower centered in a small "
        "plain vase on a clean white table. Use a simple light background and no text."
    ),
    "t2i_object32b_black_camera": (
        "Generate exactly one square image and then stop. Show one large black camera centered on a clean "
        "white table. The camera should be the only main object, with a simple light background and no text."
    ),
    "t2i_object32b_silver_spoon": (
        "Generate exactly one square image and then stop. Show one large silver spoon centered on a clean "
        "white table. The spoon should be the only main object, with a simple light background and no text."
    ),
}
