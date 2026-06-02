# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Greedy readback of exactly matched GT hue-control icon pairs."""

from configs.ume_main_image_read_hue_control_gt_seed70 import *  # noqa: F401,F403


task_type = "image_read_hue_control_gt_greedy"
save_path = f"./outputs/{exp_name}/{task_type}"

seed = 24071
sampling_params["do_sample"] = False
sampling_params["text_temperature"] = 0.8
