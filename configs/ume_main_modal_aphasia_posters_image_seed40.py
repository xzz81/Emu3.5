# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Image-from-memory poster prompts for the Modal Aphasia real-world experiment."""

from __future__ import annotations

import json
import re
from pathlib import Path

from configs.ume_main_t2i_first_image_stop_probe import *  # noqa: F401,F403


task_type = "modal_aphasia_poster_image"
save_path = f"./outputs/{exp_name}/{task_type}"

seed = 40141
trace_topk_visual = 64

image_area = 1048576
target_height = 32
target_width = 32
max_new_tokens = 1400
sampling_params["max_new_tokens"] = max_new_tokens

POSTER_DATA = Path("/workspace/home/AAAI 2027/modal-aphasia/misc/real_world_data/posters-1.json")


def _slug(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")[:72]


def _load_prompts() -> dict[str, str]:
    posters = json.loads(POSTER_DATA.read_text(encoding="utf-8"))["posters"]
    prompts = {}
    for poster_id, row in sorted(posters.items(), key=lambda item: int(item[0])):
        name = row["poster_name"]
        sample_id = f"poster_{int(poster_id):02d}_{_slug(name)}__image"
        prompts[sample_id] = (
            "Generate exactly one square image and then stop. "
            f'Create a memory-based movie-poster-style image for "{name}". '
            "Focus on the recognizable overall layout, main characters or objects, background, colors, "
            "title or tagline-like poster text if it is part of the poster, and distinctive visual details. "
            "Do not write an explanation after the image."
        )
    return prompts


prompts = _load_prompts()
