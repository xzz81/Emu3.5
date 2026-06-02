# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from configs.ume_main_image_read_entropy_object32_generated_prompt_anchor_seed47 import *  # noqa: F401,F403


seed = 24048
max_new_tokens = 80
sampling_params["max_new_tokens"] = max_new_tokens
sampling_params["do_sample"] = False
sampling_params.pop("text_temperature", None)

