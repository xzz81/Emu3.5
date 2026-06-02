# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from configs.ume_main_t2i_first_image_stop_probe import *  # noqa: F401,F403

seed = 9693
trace_topk_visual = 64

prompts = {
    "t2i_qavalid_red_apple_plate": (
        "Generate exactly one square image and then stop. Show one large red apple centered on a plain white "
        "plate. Use a simple light gray background with no text and no other fruit."
    ),
    "t2i_qavalid_blue_cup_mat": (
        "Generate exactly one square image and then stop. Show one large blue cup centered on a yellow table "
        "mat. Use a simple background with no text and no other cups."
    ),
    "t2i_qavalid_green_plant_orange_pot": (
        "Generate exactly one square image and then stop. Show one small green plant growing from one orange "
        "flower pot in the center. Use a plain background with no text."
    ),
    "t2i_qavalid_black_key_pink_notebook": (
        "Generate exactly one square image and then stop. Show one black key lying on top of one pink notebook "
        "in the center. Use a simple background with no text."
    ),
    "t2i_qavalid_purple_umbrella_boot": (
        "Generate exactly one square image and then stop. Show one purple umbrella standing next to one brown "
        "boot. Use a plain background with no text and no extra objects."
    ),
    "t2i_qavalid_yellow_star_blue_box": (
        "Generate exactly one square image and then stop. Show one yellow star painted on the front of one blue "
        "box. Use a plain background with no other symbols and no text."
    ),
    "t2i_qavalid_white_candle_black_holder": (
        "Generate exactly one square image and then stop. Show one white candle inside one black candle holder "
        "in the center. Use a simple background with no text."
    ),
    "t2i_qavalid_brown_basket_oranges": (
        "Generate exactly one square image and then stop. Show one brown basket holding exactly three orange "
        "oranges. Use a plain table and no text."
    ),
    "t2i_qavalid_silver_spoon_red_bowl": (
        "Generate exactly one square image and then stop. Show one silver spoon resting inside one red bowl. "
        "Use a plain background with no text."
    ),
    "t2i_qavalid_green_book_pencil": (
        "Generate exactly one square image and then stop. Show one green book with one yellow pencil placed "
        "across it. Use a simple background with no text."
    ),
    "t2i_qavalid_blue_ball_chair": (
        "Generate exactly one square image and then stop. Show one blue ball under one wooden chair. Use a "
        "plain room background with no text."
    ),
    "t2i_qavalid_red_flag_sandcastle": (
        "Generate exactly one square image and then stop. Show one red flag on top of one small sandcastle. "
        "Use a simple beach-colored background with no text."
    ),
}
