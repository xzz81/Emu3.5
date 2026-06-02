# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from configs.ume_main_story_first_image_stop_probe import *  # noqa: F401,F403

seed = 9291

prompts = {
    "story_one_scene_girl_kite": (
        "Create exactly one illustrated scene for a very short story. The scene must show one child in a yellow "
        "raincoat holding a red kite beside a puddle. After the image, write only one concise sentence "
        "describing the scene, then stop."
    ),
    "story_one_scene_robot_cat": (
        "Create exactly one illustrated scene for a very short story. The scene must show one small silver robot "
        "offering a blue ball to one orange cat on a sofa. After the image, write only one concise sentence "
        "describing the scene, then stop."
    ),
    "story_one_scene_boat_lighthouse": (
        "Create exactly one illustrated scene for a very short story. The scene must show one tiny green boat "
        "near a tall striped lighthouse at sunset. After the image, write only one concise sentence describing "
        "the scene, then stop."
    ),
    "story_one_scene_baker_window": (
        "Create exactly one illustrated scene for a very short story. The scene must show one baker holding a "
        "tray of round bread beside a bright bakery window. After the image, write only one concise sentence "
        "describing the scene, then stop."
    ),
}
