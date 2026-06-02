# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from configs.ume_main_t2i_first_image_stop_probe import *  # noqa: F401,F403

seed = 9293

prompts = {
    "t2i_one_image_lantern_books": (
        "Generate exactly one square image and then stop. The image shows one warm yellow lantern on top of "
        "two stacked purple books, with a small silver key lying to the right of the books."
    ),
    "t2i_one_image_bicycle_flowers": (
        "Generate exactly one square image and then stop. The image shows one blue bicycle leaning against a "
        "white fence, with three red flowers growing beside the front wheel."
    ),
    "t2i_one_image_clock_cactus": (
        "Generate exactly one square image and then stop. The image shows one round black wall clock above a "
        "green cactus in a terracotta pot on a narrow shelf."
    ),
    "t2i_one_image_cupcake_plate": (
        "Generate exactly one square image and then stop. The image shows exactly four cupcakes on a white "
        "plate: two pink cupcakes in the back row and two chocolate cupcakes in the front row."
    ),
}
