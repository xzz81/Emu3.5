# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Read exactly matched GT hue-control icon pairs with two input forms."""

from __future__ import annotations

import json
from pathlib import Path

from configs.ume_main_image_read_entropy_seed21 import *  # noqa: F401,F403


task_type = "image_read_hue_control_gt"
save_path = f"./outputs/{exp_name}/{task_type}"

seed = 24070
image_area = 262144
max_new_tokens = 120
sampling_params["max_new_tokens"] = max_new_tokens
sampling_params["do_sample"] = True
sampling_params["text_temperature"] = 0.8
sampling_params["text_top_p"] = 0.9

GT_DIR = Path("/workspace/home/AAAI 2027/research_logs/hue_control_gt_images_seed70")
MANIFEST = GT_DIR / "manifest.json"


prompts = {}
for row in json.loads(MANIFEST.read_text(encoding="utf-8")):
    image_id = row["image_id"]
    base = {
        "reference_image": row["local_path"],
        "pair_id": row["pair_id"],
        "variant": row["variant"],
        "shape": row["shape"],
        "fill_color": row["fill_color"],
        "allowed_colors": row["allowed_colors"],
        "gt_description": row["gt_description"],
    }
    prompts[f"{image_id}__describe"] = {
        **base,
        "prompt": "Describe this image carefully.",
        "input_form": "image_plus_text",
    }
    prompts[f"{image_id}__imageonly"] = {
        **base,
        "prompt": "",
        "input_form": "pure_image",
    }
