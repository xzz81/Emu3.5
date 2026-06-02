# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from configs.ume_main_g2u_spatial_prompt_strategies_seed60 import (  # noqa: F401,F403
    CASES,
    IMAGE_ROOT,
    STRATEGIES,
)
from configs.ume_main_g2u_spatial_prompt_strategies_seed60 import *  # noqa: F401,F403


seed = 24061
sampling_params["do_sample"] = True
sampling_params["text_temperature"] = 1.0
sampling_params["text_top_p"] = 0.9
sampling_params["text_top_k"] = 1024

REPEATS = 3

prompts = {}
for case_key, object_phrase, position_phrase in CASES:
    image_path = str(IMAGE_ROOT / f"{case_key}.png")
    for strategy, prompt_template in STRATEGIES.items():
        for rep in range(REPEATS):
            prompts[f"g2u_{case_key}_rep{rep:02d}__{strategy}"] = {
                "prompt": prompt_template.format(object_phrase=object_phrase),
                "reference_image": image_path,
                "expected_object": object_phrase,
                "expected_position": position_phrase,
                "strategy": strategy,
                "case_key": case_key,
                "repeat_id": rep,
            }
