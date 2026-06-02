# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Text-from-memory poster prompts for the Modal Aphasia real-world experiment."""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.utils.logging_utils import setup_logger


cfg_name = Path(__file__).stem

model_path = "/workspace/home/AAAI 2027/models/BAAI/Emu3.5"
vq_path = "/workspace/home/AAAI 2027/models/BAAI/Emu3.5-VisionTokenizer"
tokenizer_path = "./src/tokenizer_emu3_ibq"
vq_type = "ibq"

task_type = "modal_aphasia_poster_text"

exp_name = "emu3p5-main"
save_path = f"./outputs/{exp_name}/{task_type}"
save_to_proto = True
setup_logger(save_path)

hf_device = "auto"
vq_device = "cuda:0"
streaming = False

max_new_tokens = 260
sampling_params = dict(
    use_cache=True,
    text_top_k=1024,
    text_top_p=0.9,
    text_temperature=0.8,
    max_new_tokens=max_new_tokens,
    do_sample=True,
    num_beams=1,
    pad_token_id=None,
    eos_token_id=None,
)

special_tokens = dict(
    BOS="<|extra_203|>",
    EOS="<|extra_204|>",
    PAD="<|endoftext|>",
    EOL="<|extra_200|>",
    EOF="<|extra_201|>",
    TMS="<|extra_202|>",
    IMG="<|image token|>",
    BOI="<|image start|>",
    EOI="<|image end|>",
    BSS="<|extra_100|>",
    ESS="<|extra_101|>",
    BOG="<|extra_60|>",
    EOG="<|extra_61|>",
    BOC="<|extra_50|>",
    EOC="<|extra_51|>",
)

seed = 40140

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
        sample_id = f"poster_{int(poster_id):02d}_{_slug(name)}__text"
        prompts[sample_id] = (
            f'Describe from memory the original theatrical US movie poster for "{name}". '
            "Focus only on visible poster content: main characters or objects, background, colors, "
            "composition, title, tagline or readable text, and distinctive visual details. "
            "Write a concrete visual description; do not discuss uncertainty."
        )
    return prompts


template = "<|extra_203|>You are a helpful assistant for visual memory description. USER: {question} ASSISTANT: <|extra_100|>"
prompts = _load_prompts()
