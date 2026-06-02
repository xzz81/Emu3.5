# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from configs.ume_main_t2i_first_image_stop_probe import *  # noqa: F401,F403

seed = 17071
trace_topk_visual = 64

image_area = 1048576
target_height = 32
target_width = 32
max_new_tokens = 1400
sampling_params["max_new_tokens"] = max_new_tokens

GRID_PROMPT = (
    "Generate exactly one square top-down flat lay image and then stop. Show a pure white background with "
    "one thin light gray vertical line and one thin light gray horizontal line crossing at the image center, "
    "dividing the image into four equal quadrants. Place one {object_phrase} entirely inside the {quadrant} "
    "quadrant. Keep the other three quadrants empty. Use no text."
)

OBJECTS = {
    "red_apple": ("red apple", "apple"),
    "red_button": ("red button", "button"),
    "black_sunglasses": ("pair of black sunglasses", "sunglasses"),
    "purple_cube": ("purple cube", "cube"),
}

QUADRANTS = {
    "ul": ("upper-left", "upper-left"),
    "ur": ("upper-right", "upper-right"),
    "ll": ("lower-left", "lower-left"),
    "lr": ("lower-right", "lower-right"),
}

prompts = {}
for quadrant_key, (quadrant_phrase, _) in QUADRANTS.items():
    for object_key, (object_phrase, _) in OBJECTS.items():
        prompts[f"t2i_object32m_gridpool_{quadrant_key}_{object_key}"] = GRID_PROMPT.format(
            object_phrase=object_phrase,
            quadrant=quadrant_phrase,
        )
