# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from configs.ume_main_image_read_entropy_seed21 import *  # noqa: F401,F403
from configs.ume_main_image_read_entropy_simple_seed23 import CASES, IMAGE_ROOT  # noqa: F401


seed = 24060
task_type = "g2u_spatial_prompt_strategies"
max_new_tokens = 48
sampling_params["max_new_tokens"] = max_new_tokens
sampling_params["do_sample"] = False


POSITION_OPTIONS = "upper-left, upper-right, lower-left, or lower-right"

STRATEGIES = {
    "answer_only": (
        "What is the exact position of the {object_phrase} in this image? "
        "Answer with only one phrase: " + POSITION_OPTIONS + "."
    ),
    "describe_then_answer": (
        "Briefly describe this image, then on the final line write "
        "`Position: <upper-left|upper-right|lower-left|lower-right>` for the {object_phrase}."
    ),
    "grid_then_answer": (
        "Mentally overlay a 2x2 grid on the image. First identify whether the {object_phrase} "
        "is in the top or bottom half and left or right half. Then on the final line write "
        "`Position: <upper-left|upper-right|lower-left|lower-right>`."
    ),
}


prompts = {}
for case_key, object_phrase, position_phrase in CASES:
    image_path = str(IMAGE_ROOT / f"{case_key}.png")
    for strategy, prompt_template in STRATEGIES.items():
        prompts[f"g2u_{case_key}__{strategy}"] = {
            "prompt": prompt_template.format(object_phrase=object_phrase),
            "reference_image": image_path,
            "expected_object": object_phrase,
            "expected_position": position_phrase,
            "strategy": strategy,
            "case_key": case_key,
        }
