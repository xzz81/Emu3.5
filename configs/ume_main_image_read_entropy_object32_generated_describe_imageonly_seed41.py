# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import json

from configs.ume_main_image_read_entropy_seed21 import *  # noqa: F401,F403


seed = 24041
max_new_tokens = 80
sampling_params["max_new_tokens"] = max_new_tokens
sampling_params["do_sample"] = True
sampling_params["text_temperature"] = 1.0

QA_SPEC = Path("../research_logs/main_model_20260530/object32_manual_gold_20sample_analysis/manual_object_gold_20sample_qa_spec.json")

GEN_RUN_BY_PREFIX = {
    "t2i_object32_": "main_t2i_first_image_stop_topk64_object32_seed4_20260530",
    "t2i_object32b_": "main_t2i_first_image_stop_topk64_object32_seed4b_20260530",
    "t2i_object32c_": "main_t2i_first_image_stop_topk64_object32_seed4c_20260530",
    "t2i_object32d_": "main_t2i_first_image_stop_topk64_object32_seed4d_20260530",
    "t2i_object32e_": "main_t2i_first_image_stop_topk64_object32_seed4e_20260530",
}

T2I_RUN_ROOT = Path("outputs/emu3p5-main/t2i/ume_trace_runs")


def generation_run_for(sample_id: str) -> str:
    for prefix, run_id in sorted(GEN_RUN_BY_PREFIX.items(), key=lambda item: len(item[0]), reverse=True):
        if sample_id.startswith(prefix):
            return run_id
    raise KeyError(f"no generation run mapped for {sample_id}")


with QA_SPEC.open(encoding="utf-8") as f:
    qa_spec = json.load(f)

prompts = {}
for sample_id, item in qa_spec.items():
    if sample_id == "__replace_defaults__":
        continue
    run_id = generation_run_for(sample_id)
    image_path = T2I_RUN_ROOT / run_id / "decoded" / f"{sample_id}_image_00.png"
    expected_object = item["answer"]
    prompts[f"{sample_id}__describe"] = {
        "prompt": "Describe this image carefully.",
        "reference_image": str(image_path),
        "expected_object": expected_object,
        "generation_sample_id": sample_id,
        "generation_run_id": run_id,
    }
    prompts[f"{sample_id}__imageonly"] = {
        "prompt": "",
        "reference_image": str(image_path),
        "expected_object": expected_object,
        "generation_sample_id": sample_id,
        "generation_run_id": run_id,
    }
