# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Low-resolution counterfactual T2I prompt pairs for fast complete analysis."""

from configs.ume_main_t2i_counterfactual_pairs_seed64 import *  # noqa: F401,F403


task_type = "t2i_counterfactual_pairs_lowres"
save_path = f"./outputs/{exp_name}/{task_type}"

seed = 24065
trace_topk_visual = 0

image_area = 65536
target_height = 16
target_width = 16
max_new_tokens = 700
sampling_params["max_new_tokens"] = max_new_tokens
