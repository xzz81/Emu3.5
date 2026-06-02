# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""High-contrast 32x32 color/object counterfactual T2I expansion."""

from configs.ume_main_t2i_first_image_stop_probe import *  # noqa: F401,F403


task_type = "t2i_counterfactual_pairs_highcontrast_expand"
save_path = f"./outputs/{exp_name}/{task_type}"

seed = 24067
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
        "hc_color_circle_red_vs_blue",
        "color",
        "red circle",
        "blue circle",
        "one large solid {value} centered on a plain white background",
    ),
    (
        "hc_color_square_green_vs_yellow",
        "color",
        "green square",
        "yellow square",
        "one large solid {value} centered on a plain white background",
    ),
    (
        "hc_color_triangle_purple_vs_orange",
        "color",
        "purple triangle",
        "orange triangle",
        "one large solid {value} centered on a plain white background",
    ),
    (
        "hc_shape_red_circle_vs_square",
        "shape",
        "red circle",
        "red square",
        "one large solid {value} centered on a plain white background",
    ),
]


prompts = {}
pair_metadata = {}
for pair_id, factor, value_a, value_b, object_template in PAIR_SPECS:
    for variant, value in (("a", value_a), ("b", value_b)):
        sample_id = f"cf_{pair_id}__{variant}"
        prompts[sample_id] = (
            "Generate exactly one square image and then stop. Make a simple flat icon: "
            f"{object_template.format(value=value)}. No text, no shadows, no extra objects."
        )
        pair_metadata[sample_id] = {
            "pair_id": pair_id,
            "factor": factor,
            "variant": variant,
            "counterfactual_value": value,
        }
