# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from configs.ume_main_image_read_entropy_seed21 import *  # noqa: F401,F403
from configs.ume_main_image_read_entropy_simple_seed23 import CASES, IMAGE_ROOT  # noqa: F401


seed = 24062
task_type = "g2u_spatial_transform"
max_new_tokens = 64
sampling_params["max_new_tokens"] = max_new_tokens
sampling_params["do_sample"] = False


POSITION_OPTIONS = "upper-left, upper-right, lower-left, or lower-right"


def mirror_horizontal(position: str) -> str:
    return {
        "upper-left": "upper-right",
        "upper-right": "upper-left",
        "lower-left": "lower-right",
        "lower-right": "lower-left",
    }[position]


def mirror_vertical(position: str) -> str:
    return {
        "upper-left": "lower-left",
        "upper-right": "lower-right",
        "lower-left": "upper-left",
        "lower-right": "upper-right",
    }[position]


def rotate_180(position: str) -> str:
    return {
        "upper-left": "lower-right",
        "upper-right": "lower-left",
        "lower-left": "upper-right",
        "lower-right": "upper-left",
    }[position]


TRANSFORMS = {
    "mirror_horizontal": {
        "instruction": "mirrored horizontally, swapping left and right",
        "fn": mirror_horizontal,
    },
    "mirror_vertical": {
        "instruction": "mirrored vertically, swapping top and bottom",
        "fn": mirror_vertical,
    },
    "rotate_180": {
        "instruction": "rotated 180 degrees",
        "fn": rotate_180,
    },
}

STRATEGIES = {
    "answer_only": (
        "If this image were {transform_instruction}, what would be the exact position of the {object_phrase}? "
        "Answer with only one phrase: " + POSITION_OPTIONS + "."
    ),
    "state_then_answer": (
        "First identify the current position of the {object_phrase}. Then apply this transformation: "
        "{transform_instruction}. On the final line write `Position: <upper-left|upper-right|lower-left|lower-right>`."
    ),
    "grid_transform_answer": (
        "Mentally overlay a 2x2 grid on the image. Locate the {object_phrase} in the current grid, then imagine the image "
        "{transform_instruction}. On the final line write `Position: <upper-left|upper-right|lower-left|lower-right>`."
    ),
}


prompts = {}
for case_key, object_phrase, position_phrase in CASES:
    image_path = str(IMAGE_ROOT / f"{case_key}.png")
    for transform_name, transform in TRANSFORMS.items():
        expected_after_transform = transform["fn"](position_phrase)
        for strategy, prompt_template in STRATEGIES.items():
            sample_id = f"g2u_transform_{case_key}__{transform_name}__{strategy}"
            prompts[sample_id] = {
                "prompt": prompt_template.format(
                    object_phrase=object_phrase,
                    transform_instruction=transform["instruction"],
                ),
                "reference_image": image_path,
                "expected_object": object_phrase,
                "source_position": position_phrase,
                "expected_position": expected_after_transform,
                "strategy": strategy,
                "case_key": f"{case_key}__{transform_name}",
                "base_case_key": case_key,
                "transform": transform_name,
            }
