# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from configs.ume_main_t2i_first_image_stop_probe import *  # noqa: F401,F403

seed = 9393
trace_topk_visual = 64

prompts = {
    "t2i_topk64_mugs": (
        "Generate exactly one square image and then stop. The image shows exactly three ceramic mugs on a table: "
        "red on the left, blue in the center, yellow on the right, with one small green apple in front of the blue mug."
    ),
}
