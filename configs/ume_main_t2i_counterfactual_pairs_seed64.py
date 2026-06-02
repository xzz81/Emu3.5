# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Counterfactual T2I prompt pairs for input-output UME coupling."""

from __future__ import annotations

from configs.ume_main_t2i_first_image_stop_probe import *  # noqa: F401,F403


task_type = "t2i_counterfactual_pairs"
save_path = f"./outputs/{exp_name}/{task_type}"

seed = 24064
trace_topk_visual = 64

image_area = 1048576
target_height = 32
target_width = 32
max_new_tokens = 1400
sampling_params["max_new_tokens"] = max_new_tokens
sampling_params["do_sample"] = True
sampling_params["image_temperature"] = 1.0
sampling_params["image_top_p"] = 1.0


PAIR_SPECS = [
    (
        "color_apple_red_vs_blue",
        "color",
        "red",
        "blue",
        "Generate exactly one square image and then stop. Show one large {value} apple centered on a clean white table. "
        "The apple should be the only main object, with a simple light background and no text.",
    ),
    (
        "color_cube_green_vs_yellow",
        "color",
        "green",
        "yellow",
        "Generate exactly one square image and then stop. Show one large {value} cube centered on a clean white table. "
        "The cube should be the only main object, with a simple light background and no text.",
    ),
    (
        "color_cup_blue_vs_red",
        "color",
        "blue",
        "red",
        "Generate exactly one square image and then stop. Show one large {value} ceramic cup centered on a clean white table. "
        "The cup should be the only main object, with a simple light background and no text.",
    ),
    (
        "color_flower_purple_vs_orange",
        "color",
        "purple",
        "orange",
        "Generate exactly one square image and then stop. Show one large {value} flower centered in a plain white vase on a clean table. "
        "Use a simple light background and no text.",
    ),
    (
        "position_ball_ul_vs_lr",
        "position",
        "upper-left",
        "lower-right",
        "Generate exactly one square image and then stop. Show one small red ball in the {value} area of a clean white background. "
        "The ball should be the only object and there should be no text.",
    ),
    (
        "position_cube_ur_vs_ll",
        "position",
        "upper-right",
        "lower-left",
        "Generate exactly one square image and then stop. Show one small blue cube in the {value} area of a clean white background. "
        "The cube should be the only object and there should be no text.",
    ),
    (
        "count_apples_one_vs_two",
        "count",
        "one red apple",
        "two red apples",
        "Generate exactly one square image and then stop. Show {value} on a clean white table with a simple light background and no text.",
    ),
    (
        "count_mugs_one_vs_two",
        "count",
        "one blue ceramic mug",
        "two blue ceramic mugs",
        "Generate exactly one square image and then stop. Show {value} on a clean white table with a simple light background and no text.",
    ),
]


prompts = {}
pair_metadata = {}
for pair_id, factor, value_a, value_b, template_text in PAIR_SPECS:
    for variant, value in (("a", value_a), ("b", value_b)):
        sample_id = f"cf_{pair_id}__{variant}"
        prompts[sample_id] = template_text.format(value=value)
        pair_metadata[sample_id] = {
            "pair_id": pair_id,
            "factor": factor,
            "variant": variant,
            "counterfactual_value": value,
        }
