# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from configs.ume_main_t2i_first_image_stop_probe import *  # noqa: F401,F403

seed = 9895
trace_topk_visual = 64

image_area = 1048576
target_height = 32
target_width = 32
max_new_tokens = 1400
sampling_params["max_new_tokens"] = max_new_tokens

prompts = {
    "t2i_object32c_yellow_duck": (
        "Generate exactly one square image and then stop. Show one large yellow rubber duck centered on a clean "
        "white table. The duck should be the only main object, with a simple light background and no text."
    ),
    "t2i_object32c_red_toy_car": (
        "Generate exactly one square image and then stop. Show one large red toy car centered on a clean "
        "white table. The toy car should be the only main object, with a simple light background and no text."
    ),
    "t2i_object32c_green_pear": (
        "Generate exactly one square image and then stop. Show one large green pear centered on a clean "
        "white table. The pear should be the only main object, with a simple light background and no text."
    ),
    "t2i_object32c_brown_teddy_bear": (
        "Generate exactly one square image and then stop. Show one large brown teddy bear centered on a clean "
        "white table. The teddy bear should be the only main object, with a simple light background and no text."
    ),
}
