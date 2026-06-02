# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from configs.ume_main_image_read_entropy_object32_generated_describe_imageonly_seed41 import *  # noqa: F401,F403


seed = 24043
max_new_tokens = 16
sampling_params["max_new_tokens"] = max_new_tokens
sampling_params["do_sample"] = False
sampling_params.pop("text_temperature", None)

prompts = {}
for sample_id, item in qa_spec.items():
    if sample_id == "__replace_defaults__":
        continue
    run_id = generation_run_for(sample_id)
    image_path = T2I_RUN_ROOT / run_id / "decoded" / f"{sample_id}_image_00.png"
    expected_object = item["answer"]
    prompts[f"{sample_id}__objectname"] = {
        "prompt": "What object is shown in the image? Answer with only the object name.",
        "reference_image": str(image_path),
        "expected_object": expected_object,
        "generation_sample_id": sample_id,
        "generation_run_id": run_id,
    }
