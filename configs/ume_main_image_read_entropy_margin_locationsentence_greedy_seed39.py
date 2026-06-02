# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from configs.ume_main_image_read_entropy_margin_locationsentence_repeat_seed38 import *  # noqa: F401,F403


seed = 24039
max_new_tokens = 32
sampling_params["max_new_tokens"] = max_new_tokens
sampling_params["do_sample"] = False
sampling_params.pop("text_temperature", None)

repeats = 1

prompts = {}
for margin in MARGINS:
    for case_key, object_phrase, position_phrase in BASE_CASES:
        margin_case_key = f"m{margin:03d}_{case_key}"
        image_path = str(IMAGE_ROOT / f"{margin_case_key}.png")
        prompt = (
            "Answer in one sentence. Use exactly one location phrase from this list: "
            "upper-left, upper-right, lower-left, lower-right. "
            "Format your answer exactly like: It is in the <location> part of the image."
        )
        prompts[f"margingreedylocsent_{margin_case_key}__locationsentence"] = {
            "prompt": prompt,
            "reference_image": image_path,
            "expected_object": object_phrase,
            "expected_position": position_phrase,
            "repeat": 0,
            "margin_px": margin,
        }
