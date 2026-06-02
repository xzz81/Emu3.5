# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Small high-contrast counterfactual T2I prompt-pair probe."""

from configs.ume_main_t2i_first_image_stop_probe import *  # noqa: F401,F403


task_type = "t2i_counterfactual_pairs_highcontrast"
save_path = f"./outputs/{exp_name}/{task_type}"

seed = 24066
trace_topk_visual = 0

image_area = 1048576
target_height = 32
target_width = 32
max_new_tokens = 1400
sampling_params["max_new_tokens"] = max_new_tokens
sampling_params["do_sample"] = True
sampling_params["image_temperature"] = 1.0
sampling_params["image_top_p"] = 1.0

prompts = {
    "cf_hc_color_circle_red_vs_blue__a": (
        "Generate exactly one square image and then stop. Make a simple flat icon: one large solid red circle centered "
        "on a plain white background. No text, no shadows, no extra objects."
    ),
    "cf_hc_color_circle_red_vs_blue__b": (
        "Generate exactly one square image and then stop. Make a simple flat icon: one large solid blue circle centered "
        "on a plain white background. No text, no shadows, no extra objects."
    ),
    "cf_hc_position_dot_ul_vs_lr__a": (
        "Generate exactly one square image and then stop. Make a simple flat icon: one small solid black dot in the "
        "upper-left corner on a plain white background. No text, no shadows, no extra objects."
    ),
    "cf_hc_position_dot_ul_vs_lr__b": (
        "Generate exactly one square image and then stop. Make a simple flat icon: one small solid black dot in the "
        "lower-right corner on a plain white background. No text, no shadows, no extra objects."
    ),
}

pair_metadata = {
    "cf_hc_color_circle_red_vs_blue__a": {
        "pair_id": "hc_color_circle_red_vs_blue",
        "factor": "color",
        "variant": "a",
        "counterfactual_value": "red circle",
    },
    "cf_hc_color_circle_red_vs_blue__b": {
        "pair_id": "hc_color_circle_red_vs_blue",
        "factor": "color",
        "variant": "b",
        "counterfactual_value": "blue circle",
    },
    "cf_hc_position_dot_ul_vs_lr__a": {
        "pair_id": "hc_position_dot_ul_vs_lr",
        "factor": "position",
        "variant": "a",
        "counterfactual_value": "upper-left dot",
    },
    "cf_hc_position_dot_ul_vs_lr__b": {
        "pair_id": "hc_position_dot_ul_vs_lr",
        "factor": "position",
        "variant": "b",
        "counterfactual_value": "lower-right dot",
    },
}
