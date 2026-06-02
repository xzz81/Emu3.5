# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Hue-only counterfactual T2I controls with matched outline geometry."""

from configs.ume_main_t2i_first_image_stop_probe import *  # noqa: F401,F403


task_type = "t2i_counterfactual_pairs_hueonly"
save_path = f"./outputs/{exp_name}/{task_type}"

seed = 24069
trace_topk_visual = 0

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
        "hueonly_circle_red_vs_blue",
        "circle",
        "red",
        "blue",
    ),
    (
        "hueonly_square_green_vs_purple",
        "square",
        "green",
        "purple",
    ),
    (
        "hueonly_triangle_yellow_vs_cyan",
        "triangle",
        "yellow",
        "cyan",
    ),
    (
        "hueonly_diamond_orange_vs_magenta",
        "diamond",
        "orange",
        "magenta",
    ),
]


prompts = {}
pair_metadata = {}
for pair_id, shape, color_a, color_b in PAIR_SPECS:
    for variant, color in (("a", color_a), ("b", color_b)):
        sample_id = f"cf_{pair_id}__{variant}"
        prompts[sample_id] = (
            "Generate exactly one square image and then stop. Make a simple flat icon on a plain white background: "
            f"one centered geometric {shape}, fixed size, thick black outline, and a solid {color} interior. "
            "Keep the same outline geometry and position; only the interior hue should differ. "
            "No text, no shadows, no gradients, no extra objects."
        )
        pair_metadata[sample_id] = {
            "pair_id": pair_id,
            "factor": "hue_only_edge_matched",
            "variant": variant,
            "counterfactual_value": f"{color} {shape}",
        }
