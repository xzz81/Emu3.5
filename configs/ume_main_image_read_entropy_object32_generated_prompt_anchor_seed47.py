# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from configs.ume_main_image_read_entropy_object32_generated_describe_imageonly_seed41 import *  # noqa: F401,F403


seed = 24047
max_new_tokens = 80
sampling_params["max_new_tokens"] = max_new_tokens
sampling_params["do_sample"] = True
sampling_params["text_temperature"] = 1.0

PROMPT_VARIANTS = {
    "imageonly": "",
    "describe_short": "Describe.",
    "caption": "Caption this image.",
    "whatin": "What is in this image?",
    "describe": "Describe this image carefully.",
}

prompts = {}
for sample_id, item in qa_spec.items():
    if sample_id == "__replace_defaults__":
        continue
    run_id = generation_run_for(sample_id)
    image_path = T2I_RUN_ROOT / run_id / "decoded" / f"{sample_id}_image_00.png"
    expected_object = item["answer"]
    for variant_key, prompt_text in PROMPT_VARIANTS.items():
        prompts[f"{sample_id}__{variant_key}"] = {
            "prompt": prompt_text,
            "reference_image": str(image_path),
            "expected_object": expected_object,
            "generation_sample_id": sample_id,
            "generation_run_id": run_id,
            "prompt_variant": variant_key,
        }
