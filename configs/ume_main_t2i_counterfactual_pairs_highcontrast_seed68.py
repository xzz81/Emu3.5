# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""High-contrast 32x32 color-only counterfactual T2I expansion."""

from configs.ume_main_t2i_first_image_stop_probe import *  # noqa: F401,F403


task_type = "t2i_counterfactual_pairs_highcontrast_color_expand"
save_path = f"./outputs/{exp_name}/{task_type}"

seed = 24068
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
        "hc_color_diamond_cyan_vs_magenta",
        "cyan diamond",
        "magenta diamond",
    ),
    (
        "hc_color_star_black_vs_green",
        "black star",
        "green star",
    ),
    (
        "hc_color_heart_red_vs_purple",
        "red heart",
        "purple heart",
    ),
    (
        "hc_color_ring_orange_vs_blue",
        "orange ring",
        "blue ring",
    ),
]


prompts = {}
pair_metadata = {}
for pair_id, value_a, value_b in PAIR_SPECS:
    for variant, value in (("a", value_a), ("b", value_b)):
        sample_id = f"cf_{pair_id}__{variant}"
        prompts[sample_id] = (
            "Generate exactly one square image and then stop. Make a simple flat icon: "
            f"one large solid {value} centered on a plain white background. "
            "No text, no shadows, no extra objects."
        )
        pair_metadata[sample_id] = {
            "pair_id": pair_id,
            "factor": "color",
            "variant": variant,
            "counterfactual_value": value,
        }
