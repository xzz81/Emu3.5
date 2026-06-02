# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from configs.ume_main_howto_first_image_stop_probe import *  # noqa: F401,F403

seed = 9492
trace_topk_visual = 64

prompts = {
    "howto_topk64_teapot": (
        "Show exactly one illustrated step: place a small transparent glass teapot and two clear cups on a "
        "wooden tray. After the image, write one short instruction sentence, then stop."
    ),
    "howto_topk64_seedling": (
        "Show exactly one illustrated step: put a tiny green seedling into a red clay pot on a white table. "
        "After the image, write one short instruction sentence, then stop."
    ),
}
