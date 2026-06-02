# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from configs.ume_main_t2i_first_image_stop_probe import *  # noqa: F401,F403

seed = 9793
trace_topk_visual = 64

prompts = {
    "t2i_highcontrast_red_circle": (
        "Generate exactly one square image and then stop. Create a bold high-contrast flat icon: one large "
        "solid red circle centered on a pure black background. No text, no shadows, no other objects."
    ),
    "t2i_highcontrast_yellow_star": (
        "Generate exactly one square image and then stop. Create a bold high-contrast flat icon: one large "
        "solid yellow five-point star centered on a pure black background. No text and no other objects."
    ),
    "t2i_highcontrast_white_triangle": (
        "Generate exactly one square image and then stop. Create a bold high-contrast flat icon: one large "
        "solid white triangle centered on a pure black background. No text and no other objects."
    ),
    "t2i_highcontrast_blue_square": (
        "Generate exactly one square image and then stop. Create a bold high-contrast flat icon: one large "
        "solid blue square centered on a pure white background. No text and no other objects."
    ),
    "t2i_highcontrast_green_cross": (
        "Generate exactly one square image and then stop. Create a bold high-contrast flat icon: one large "
        "solid green cross centered on a pure white background. No text and no other objects."
    ),
    "t2i_highcontrast_black_diamond": (
        "Generate exactly one square image and then stop. Create a bold high-contrast flat icon: one large "
        "solid black diamond centered on a pure white background. No text and no other objects."
    ),
    "t2i_highcontrast_orange_heart": (
        "Generate exactly one square image and then stop. Create a bold high-contrast flat icon: one large "
        "solid orange heart centered on a pure black background. No text and no other objects."
    ),
    "t2i_highcontrast_purple_moon": (
        "Generate exactly one square image and then stop. Create a bold high-contrast flat icon: one large "
        "solid purple crescent moon centered on a pure white background. No text and no other objects."
    ),
}
