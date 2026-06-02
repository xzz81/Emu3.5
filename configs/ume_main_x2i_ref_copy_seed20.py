# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from configs.ume_main_x2i_ref_layout_seed19 import *  # noqa: F401,F403


seed = 24020

PROMPT_TEMPLATE = (
    "Copy the reference image as exactly as possible. Generate exactly one square image and then stop. "
    "Keep the same plain neutral gray tabletop, the same single {object_phrase}, the same object size, "
    "the same color, and the same {position_phrase} corner location as the reference. Do not reinterpret "
    "the scene, crop the object, move it toward the center, add camera tools, add texture patches, add "
    "grid lines, boxes, arrows, labels, text, borders, corner markers, extra objects, or decorations."
)

prompts = {
    f"x2i_ref_copy_{case_key}": {
        "prompt": PROMPT_TEMPLATE.format(object_phrase=object_phrase, position_phrase=position_phrase),
        "reference_image": str(REFERENCE_ROOT / f"{case_key}.png"),
    }
    for case_key, object_phrase, position_phrase in CASES
}
