# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from configs.ume_main_image_read_entropy_object32_generated_describe_imageonly_seed41 import *  # noqa: F401,F403


seed = 24046
max_new_tokens = 80
sampling_params["max_new_tokens"] = max_new_tokens
sampling_params["do_sample"] = True
sampling_params["text_temperature"] = 1.0

_base_prompts = prompts
prompts = {}
for name, question in _base_prompts.items():
    base, mode = name.rsplit("__", 1)
    for rep in range(3):
        prompts[f"{base}_rep{rep:02d}__{mode}"] = dict(question, repeat=rep)

