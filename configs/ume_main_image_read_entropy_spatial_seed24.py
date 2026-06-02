# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from configs.ume_main_image_read_entropy_seed21 import *  # noqa: F401,F403
from configs.ume_main_image_read_entropy_simple_seed23 import CASES, IMAGE_ROOT  # noqa: F401


seed = 24024
max_new_tokens = 64
sampling_params["max_new_tokens"] = max_new_tokens

prompts = {}
for case_key, object_phrase, position_phrase in CASES:
    image_path = str(IMAGE_ROOT / f"{case_key}.png")
    prompts[f"simple_{case_key}__describe"] = {
        "prompt": "Describe this image carefully.",
        "reference_image": image_path,
        "expected_object": object_phrase,
        "expected_position": position_phrase,
    }
    prompts[f"simple_{case_key}__spatialdescribe"] = {
        "prompt": "Describe this image carefully. Include the object's color, shape, and exact position in the image.",
        "reference_image": image_path,
        "expected_object": object_phrase,
        "expected_position": position_phrase,
    }
    prompts[f"simple_{case_key}__imageonly"] = {
        "prompt": "",
        "reference_image": image_path,
        "expected_object": object_phrase,
        "expected_position": position_phrase,
    }
