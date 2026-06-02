# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from configs.ume_main_story_one_scene_probe import *  # noqa: F401,F403

stop_after_completed_images = 1
stop_after_eoi_extra_tokens = 0
max_new_tokens = 1200
sampling_params["max_new_tokens"] = max_new_tokens
seed = 9191
