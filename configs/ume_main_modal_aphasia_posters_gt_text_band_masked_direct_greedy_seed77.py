# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Greedy direct readback of text-band masked real-poster GT images."""

from __future__ import annotations

import json
import re
from pathlib import Path

from configs.ume_main_image_read_entropy_seed21 import *  # noqa: F401,F403


task_type = "modal_aphasia_poster_gt_text_band_masked_direct_greedy"
save_path = f"./outputs/{exp_name}/{task_type}"

seed = 40177
max_new_tokens = 140
sampling_params["max_new_tokens"] = max_new_tokens
sampling_params["do_sample"] = False
sampling_params["text_temperature"] = 0.8

PROMPT = (
    "Describe the visible poster image in 4 to 6 concise sentences. "
    "Do not write steps, plans, reasoning, headings, or meta commentary. "
    "Neutral gray rectangles cover likely text bands; do not infer, copy, reconstruct, "
    "or guess any hidden text. Mention the gray rectangles only as masked text bands. "
    "Describe only visible people, objects, colors, lighting, and composition."
)

GT_DIR = Path("/workspace/home/AAAI 2027/research_logs/modal_aphasia_posters/gt_posters_text_band_masked_seed75")
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
    prompts[f"poster_{poster_id:02d}_{slug}__text_band_masked_direct_greedy"] = {
        "reference_image": image_path,
        "poster_name": name,
        "poster_id": poster_id,
        "original_reference_image": row.get("original_local_path", ""),
        "mask_rects_fraction": row.get("mask_rects_fraction", []),
        "mask_rects_pixels": row.get("mask_rects_pixels", []),
        "source_page_url": row["source_page_url"],
        "file_page_url": row["file_page_url"],
        "image_url": row["image_url"],
        "prompt": PROMPT,
        "prompt_variant": "text_band_masked_direct_greedy",
        "decoding": "greedy",
    }
