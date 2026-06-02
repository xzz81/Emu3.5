# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Repeated strict visible/no-guess prompt for real-poster GT readback."""

from __future__ import annotations

import json
import re
from pathlib import Path

from configs.ume_main_image_read_entropy_seed21 import *  # noqa: F401,F403


task_type = "modal_aphasia_poster_gt_strict_visible_repeat"
save_path = f"./outputs/{exp_name}/{task_type}"

seed = 40173
max_new_tokens = 180
sampling_params["max_new_tokens"] = max_new_tokens
sampling_params["do_sample"] = True
sampling_params["text_temperature"] = 0.8

REPEATS = 3
PROMPT = (
    "Describe only what is directly visible in the poster image. Do not identify the movie, "
    "actors, director, franchise, release year, or story from memory. Do not guess unreadable "
    "text or names. If text is unclear, say that it is unreadable. Mention only visible shapes, "
    "people, objects, colors, layout, and clearly readable text fragments."
)

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
    for rep in range(REPEATS):
        prompts[f"poster_{poster_id:02d}_{slug}_rep{rep:02d}__strict_visible_no_guess"] = {
            **base,
            "prompt": PROMPT,
            "prompt_variant": "strict_visible_no_guess",
            "repeat": rep,
        }
