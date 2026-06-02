# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Repeated text-from-memory poster prompts for Setting F stability analysis."""

from configs.ume_main_modal_aphasia_posters_text_seed40 import *  # noqa: F401,F403


seed = 40150
max_new_tokens = 260
sampling_params["max_new_tokens"] = max_new_tokens
sampling_params["do_sample"] = True
sampling_params["text_temperature"] = 0.8

_base_prompts = prompts
prompts = {}
for name, prompt in _base_prompts.items():
    base = name.replace("__text", "")
    for rep in range(3):
        prompts[f"{base}_rep{rep:02d}__text"] = prompt

