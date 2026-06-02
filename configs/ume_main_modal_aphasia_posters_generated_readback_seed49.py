# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Read back Emu3.5-generated Modal Aphasia poster images."""

from __future__ import annotations

import json
import re
from pathlib import Path

from configs.ume_main_image_read_entropy_seed21 import *  # noqa: F401,F403


task_type = "modal_aphasia_poster_readback"
save_path = f"./outputs/{exp_name}/{task_type}"

seed = 40149
max_new_tokens = 180
sampling_params["max_new_tokens"] = max_new_tokens
sampling_params["do_sample"] = True
sampling_params["text_temperature"] = 0.8

POSTER_DATA = Path("/workspace/home/AAAI 2027/modal-aphasia/misc/real_world_data/posters-1.json")
GENERATED_IMAGE_DIR = Path(
    "outputs/emu3p5-main/modal_aphasia_poster_image/ume_trace_runs/"
    "main_modal_aphasia_posters_image_seed40_20260531/decoded"
)


def _slug(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")[:72]


prompts = {}
posters = json.loads(POSTER_DATA.read_text(encoding="utf-8"))["posters"]
for poster_id, row in sorted(posters.items(), key=lambda item: int(item[0])):
    name = row["poster_name"]
    slug = _slug(name)
    image_path = GENERATED_IMAGE_DIR / f"poster_{int(poster_id):02d}_{slug}__image_image_00.png"
    prompts[f"poster_{int(poster_id):02d}_{slug}__readback"] = {
        "prompt": (
            "Describe this generated movie poster carefully. Focus only on visible poster content: "
            "main characters or objects, background, colors, composition, title-like text, and distinctive details. "
            "Do not infer beyond the image."
        ),
        "reference_image": str(image_path),
        "poster_name": name,
        "poster_id": int(poster_id),
        "source_generated_image": str(image_path),
    }

