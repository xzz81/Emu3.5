# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from configs.ume_main_story_first_image_stop_probe import *  # noqa: F401,F403

seed = 9491
trace_topk_visual = 64

prompts = {
    "story_topk64_car_tree": (
        "Create exactly one illustrated scene for a very short story. The scene must show one small red car "
        "beside one tall green tree under a cloudy gray sky. After the image, write only one concise sentence "
        "describing the scene, then stop."
    ),
    "story_topk64_key_backpack": (
        "Create exactly one illustrated scene for a very short story. The scene must show one brass key on a "
        "kitchen table next to one blue backpack. After the image, write only one concise sentence describing "
        "the scene, then stop."
    ),
}
