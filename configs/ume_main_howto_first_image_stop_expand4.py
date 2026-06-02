# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from configs.ume_main_howto_first_image_stop_probe import *  # noqa: F401,F403

seed = 9292

prompts = {
    "howto_one_step_thread_needle": (
        "Show exactly one illustrated step: pass a red thread through the eye of one silver needle on a blue "
        "cloth. After the image, write one short instruction sentence, then stop."
    ),
    "howto_one_step_slice_lemon": (
        "Show exactly one illustrated step: slice one yellow lemon into three round pieces on a wooden cutting "
        "board with a small knife. After the image, write one short instruction sentence, then stop."
    ),
    "howto_one_step_wrap_gift": (
        "Show exactly one illustrated step: fold green wrapping paper around one small square gift box and place "
        "a red ribbon beside it. After the image, write one short instruction sentence, then stop."
    ),
    "howto_one_step_mix_paint": (
        "Show exactly one illustrated step: mix blue paint and white paint on a flat palette with one wooden "
        "brush. After the image, write one short instruction sentence, then stop."
    ),
}
