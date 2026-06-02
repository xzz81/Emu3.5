# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Greedy real-poster GT readback for describe and visible-only prompts."""

from __future__ import annotations

import json
import re
from pathlib import Path

from configs.ume_main_image_read_entropy_seed21 import *  # noqa: F401,F403


task_type = "modal_aphasia_poster_gt_describe_visible_greedy"
save_path = f"./outputs/{exp_name}/{task_type}"

seed = 40157
max_new_tokens = 180
sampling_params["max_new_tokens"] = max_new_tokens
sampling_params["do_sample"] = False
sampling_params["text_temperature"] = 0.8

PROMPTS = {
    "describe": "Describe this image carefully.",
    "visible_only": (
        "Describe only the visible content of this poster. Focus on visible people, objects, "
        "text, colors, and composition. Do not use outside knowledge."
    ),
}
GT_DIR = Path("/workspace/home/AAAI 2027/research_logs/modal_aphasia_posters/gt_posters_wikipedia_20260531")
MANIFEST = GT_DIR / "manifest.json"


def _slug(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")[:72]


prompts = {}
for row in json.loads(MANIFEST.read_text(encoding="utf-8")):
    poster_id = int(row["poster_id"])
    name = row["poster_name"]
    slug = _slug(name)
    image_path = row["local_path"]
    base = {
        "reference_image": image_path,
        "poster_name": name,
        "poster_id": poster_id,
        "source_page_url": row["source_page_url"],
        "file_page_url": row["file_page_url"],
        "image_url": row["image_url"],
    }
    for variant, prompt_text in PROMPTS.items():
        prompts[f"poster_{poster_id:02d}_{slug}__greedy_{variant}"] = {
            **base,
            "prompt": prompt_text,
            "prompt_variant": variant,
            "decoding": "greedy",
        }
