# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Repeated readback of text-band masked real-poster GT images."""

from __future__ import annotations

import json
import re
from pathlib import Path

from configs.ume_main_image_read_entropy_seed21 import *  # noqa: F401,F403


task_type = "modal_aphasia_poster_gt_text_band_masked_repeat"
save_path = f"./outputs/{exp_name}/{task_type}"

seed = 40175
max_new_tokens = 180
sampling_params["max_new_tokens"] = max_new_tokens
sampling_params["do_sample"] = True
sampling_params["text_temperature"] = 0.8

REPEATS = 3
PROMPT = (
    "This poster image has neutral gray rectangles covering likely text bands. "
    "Do not infer, copy, or guess any text hidden by the gray masks. Describe only the unmasked "
    "visible visual content: people, objects, colors, lighting, and composition. If you mention "
    "the gray rectangles, call them masked text bands, not readable text."
)

GT_DIR = Path("data/modal_aphasia_posters/gt_posters_text_band_masked_seed75")
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
        "original_reference_image": row.get("original_local_path", ""),
        "mask_rects_fraction": row.get("mask_rects_fraction", []),
        "mask_rects_pixels": row.get("mask_rects_pixels", []),
        "source_page_url": row["source_page_url"],
        "file_page_url": row["file_page_url"],
        "image_url": row["image_url"],
    }
    for rep in range(REPEATS):
        prompts[f"poster_{poster_id:02d}_{slug}_rep{rep:02d}__text_band_masked"] = {
            **base,
            "prompt": PROMPT,
            "prompt_variant": "text_band_masked",
            "repeat": rep,
        }
