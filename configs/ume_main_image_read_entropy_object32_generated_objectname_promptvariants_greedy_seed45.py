# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from configs.ume_main_image_read_entropy_object32_generated_objectname_greedy_seed43 import *  # noqa: F401,F403


seed = 24045
max_new_tokens = 16
sampling_params["max_new_tokens"] = max_new_tokens
sampling_params["do_sample"] = False
sampling_params.pop("text_temperature", None)

PROMPT_VARIANTS = {
    "shown": "What object is shown in the image? Answer with only the object name.",
    "centered_table": "What object is centered on the table? Answer with only the object name.",
    "main": "What is the main object in the image? Answer with only the object name.",
    "salient": "What is the most visually salient object in the image? Answer with only the object name.",
}

prompts = {}
for sample_id, item in qa_spec.items():
    if sample_id == "__replace_defaults__":
        continue
    run_id = generation_run_for(sample_id)
    image_path = T2I_RUN_ROOT / run_id / "decoded" / f"{sample_id}_image_00.png"
    expected_object = item["answer"]
    for variant_key, prompt in PROMPT_VARIANTS.items():
        prompts[f"{sample_id}__objectname__{variant_key}"] = {
            "prompt": prompt,
            "reference_image": str(image_path),
            "expected_object": expected_object,
            "generation_sample_id": sample_id,
            "generation_run_id": run_id,
            "prompt_variant": variant_key,
        }
