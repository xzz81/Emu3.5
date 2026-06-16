# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from src.utils.logging_utils import setup_logger

cfg_name = Path(__file__).stem

model_path = "model/Emu3.5-Image"
vq_path = "model/Emu3.5-VisionTokenizer"
tokenizer_path = "./src/tokenizer_emu3_ibq"
vq_type = "ibq"

task_type = "t2i"
use_image = False

exp_name = "emu3p5-image"
save_path = f"./outputs/{exp_name}/{task_type}"
save_to_proto = True
setup_logger(save_path)

hf_device = "auto"
vq_device = "cuda:0"
streaming = False
unconditional_type = "no_text"
classifier_free_guidance = 5.0
max_new_tokens = 1200
image_area = 1048576

target_height = 32
target_width = 32


def build_unc_and_template(task: str, with_image: bool):
    task_str = task.lower()
    if with_image:
        unc_p = "<|extra_203|>You are a helpful assistant. USER: <|IMAGE|> ASSISTANT: <|extra_100|>"
        tmpl = "<|extra_203|>You are a helpful assistant for %s task. USER: {question}<|IMAGE|> ASSISTANT: <|extra_100|>" % task_str
    else:
        unc_p = "<|extra_203|>You are a helpful assistant. USER:  ASSISTANT: <|extra_100|>"
        tmpl = "<|extra_203|>You are a helpful assistant for %s task. USER: {question} ASSISTANT: <|extra_100|>" % task_str
    return unc_p, tmpl


unc_prompt, template = build_unc_and_template(task_type, use_image)

sampling_params = dict(
    use_cache=True,
    text_top_k=1024,
    text_top_p=0.9,
    text_temperature=1.0,
    image_top_k=5120,
    image_top_p=1.0,
    image_temperature=1.0,
    top_k=131072,
    top_p=1.0,
    temperature=1.0,
    num_beams_per_group=1,
    num_beam_groups=1,
    diversity_penalty=0.0,
    max_new_tokens=max_new_tokens,
    guidance_scale=1.0,
    use_differential_sampling=True,
)

sampling_params["do_sample"] = sampling_params["num_beam_groups"] <= 1
sampling_params["num_beams"] = sampling_params["num_beams_per_group"] * sampling_params["num_beam_groups"]

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

seed = 6666

_cases = [
    ("product_watch", "A square premium product photo of a brushed steel wristwatch on dark velvet, crisp reflections, shallow depth of field, studio lighting."),
    ("product_sneaker", "A square commercial product photo of a white running sneaker splashing through clear water, sharp droplets, clean blue background."),
    ("product_perfume", "A square luxury perfume bottle on mirrored glass with rose petals and warm backlight, elegant advertising style."),
    ("product_headphones", "A square product render of matte black wireless headphones on a concrete pedestal, soft shadows, minimal modern composition."),
    ("product_camera", "A square product photo of a vintage film camera beside a leather strap and lens cap on a wooden desk, natural window light."),
    ("portrait_elder", "A square realistic portrait of an elderly ceramic artist in a sunlit workshop, clay on hands, expressive face, documentary style."),
    ("portrait_child", "A square candid portrait of a child wearing a yellow raincoat under a transparent umbrella, rainy city street bokeh."),
    ("portrait_astronaut", "A square cinematic portrait of an astronaut with reflective visor standing in orange desert dust, dramatic rim light."),
    ("portrait_musician", "A square black-and-white portrait of a jazz saxophonist on a small stage, smoky atmosphere, high contrast lighting."),
    ("portrait_robot", "A square portrait of a friendly humanoid robot gardener holding a small seedling, greenhouse background, soft daylight."),
    ("animal_fox", "A square wildlife photo of a red fox walking through fresh snow at sunrise, visible breath, detailed fur."),
    ("animal_octopus", "A square underwater photo of a bright orange octopus curling around coral, sun rays filtering through water."),
    ("animal_hummingbird", "A square macro photo of a hummingbird hovering near purple flowers, wings motion-blurred, crisp beak and eye."),
    ("animal_cat_library", "A square cozy illustration of a gray cat sleeping on stacked books inside an old library, warm lamp light."),
    ("animal_elephant", "A square realistic scene of a baby elephant playing near a shallow river, green forest background, soft afternoon light."),
    ("landscape_mountain", "A square alpine landscape at dawn with jagged snow peaks reflected in a still turquoise lake, clear air, panoramic depth."),
    ("landscape_desert", "A square desert landscape with wind-carved dunes under a starry sky, a small lantern glowing near footprints."),
    ("landscape_rainforest", "A square rainforest waterfall surrounded by mossy rocks and mist, lush green foliage, long exposure water."),
    ("landscape_coast", "A square rocky coastline during golden hour, waves crashing against cliffs, seabirds in the distance."),
    ("landscape_city", "A square futuristic city skyline at dusk with elevated trains, glowing windows, wet streets reflecting neon."),
    ("food_ramen", "A square close-up food photo of a steaming bowl of ramen with soft-boiled egg, scallions, and chili oil, dark ceramic bowl."),
    ("food_pastry", "A square bakery display of flaky croissants and fruit tarts behind glass, warm morning light, realistic crumbs."),
    ("food_salad", "A square overhead photo of a colorful Mediterranean salad with olives, tomatoes, cucumbers, feta, and herbs."),
    ("food_sushi", "A square elegant sushi platter on black slate with soy sauce, wasabi, ginger, precise arrangement."),
    ("food_pancakes", "A square breakfast photo of stacked pancakes with blueberries and maple syrup pouring down the side."),
    ("architecture_museum", "A square architectural photo of a modern art museum atrium with curved white walls, skylight, polished floor."),
    ("architecture_cabin", "A square cozy wooden cabin in a snowy forest, warm windows glowing, smoke rising from chimney."),
    ("architecture_bridge", "A square dramatic photo of a suspension bridge disappearing into fog, wet steel cables, early morning."),
    ("architecture_courtyard", "A square Mediterranean courtyard with blue tiled fountain, climbing bougainvillea, sunlit stucco walls."),
    ("architecture_library", "A square grand library interior with tall shelves, spiral staircases, reading tables, golden lamps."),
    ("text_sign_bakery", "A square close-up illustration of a clean white bakery sign on a brick wall. The sign says SWEET BUNS in large readable dark letters, with two croissants below."),
    ("text_sign_station", "A square realistic photo of a subway station sign that clearly reads PLATFORM 7, yellow tiles, commuters blurred in background."),
    ("text_poster_moon", "A square vintage travel poster with large readable text VISIT THE MOON, rocket silhouette, retro colors."),
    ("text_menu_cafe", "A square cafe chalkboard menu with readable handwritten text COFFEE TEA CAKE, cups and pastries on a counter."),
    ("text_neon_arcade", "A square night street scene with a neon sign that clearly reads ARCADE OPEN, rain reflections, cyberpunk style."),
    ("diagram_cycle", "A square clean educational diagram showing the water cycle with arrows, clouds, rain, river, sun, and readable labels."),
    ("diagram_engine", "A square technical cutaway diagram of a small electric motor with colored coils, rotor, magnets, and clear callout labels."),
    ("diagram_garden", "A square illustrated garden layout map with paths, vegetable beds, greenhouse, pond, and small labels."),
    ("diagram_spacecraft", "A square blueprint-style diagram of a compact spacecraft with numbered modules, solar panels, and docking port."),
    ("diagram_brain", "A square medical-style diagram of a human brain with highlighted regions and clean annotation lines."),
    ("count_mugs", "A square children's book illustration of exactly three ceramic mugs on a wooden table: red left, blue center, yellow right; a green apple sits in front of the blue mug."),
    ("count_balloons", "A square cheerful illustration of exactly seven balloons tied to a park bench: two red, two blue, two yellow, one green."),
    ("count_birds", "A square realistic scene of exactly five white birds standing on a wooden pier, evenly spaced, foggy lake background."),
    ("count_coins", "A square macro photo of exactly nine old coins arranged in a 3 by 3 grid on dark cloth, sharp engraved details."),
    ("count_candles", "A square birthday cake with exactly twelve lit candles arranged in a circle, colorful frosting, dark party background."),
    ("spatial_blocks", "A square render of a red cube behind a transparent glass sphere, a blue cylinder to the left, and a yellow cone on top of a black platform."),
    ("spatial_room", "A square minimal room with a green chair under a window, a round table in front of it, and a blue vase behind the table."),
    ("spatial_shelf", "A square bookshelf scene where a silver trophy is between two red books, with a small clock above them."),
    ("spatial_toytrain", "A square toy train curving around a miniature tree, with a tiny station inside the track loop and mountains painted behind."),
    ("spatial_kitchen", "A square kitchen counter with a knife to the right of a cutting board, tomatoes on the board, and a bowl behind them."),
    ("material_glass", "A square still life of transparent glass marbles in a crystal bowl, caustic light patterns on a white table."),
    ("material_metal", "A square macro photo of brushed copper pipes with water droplets, warm industrial lighting."),
    ("material_fabric", "A square close-up of folded silk scarves in emerald, gold, and burgundy, visible woven texture."),
    ("material_wood", "A square detailed photo of carved walnut wood panels with intricate floral patterns and warm varnish."),
    ("material_ice", "A square macro scene of ice cubes with trapped air bubbles on a black surface, cold blue lighting."),
    ("action_ballet", "A square dynamic photo of a ballet dancer mid-leap in an empty theater, flowing white costume, spotlight."),
    ("action_cyclist", "A square sports photo of a cyclist racing through mud, droplets frozen in air, forest trail background."),
    ("action_chef", "A square kitchen action shot of a chef tossing vegetables in a flaming wok, dramatic motion and sparks."),
    ("action_surfer", "A square ocean photo of a surfer inside a curling wave, turquoise water, sunlight through spray."),
    ("action_painter", "A square studio scene of a painter splashing bright colors onto a large canvas, energetic motion."),
    ("night_market", "A square busy night market with lanterns, food stalls, steam, colorful signs, and crowds under warm lights."),
    ("night_forest", "A square moonlit forest path with glowing mushrooms and mist, fantasy realism, deep blue shadows."),
    ("night_train", "A square cinematic train platform at midnight, single train arriving, wet pavement, overhead lights."),
    ("night_harbor", "A square harbor at night with boats, reflections, distant city lights, calm water."),
    ("night_camp", "A square camping scene with a small tent and campfire under the Milky Way, silhouettes of pine trees."),
    ("fantasy_castle", "A square fantasy castle floating above clouds with waterfalls falling into the sky, sunrise glow."),
    ("fantasy_dragon", "A square epic fantasy scene of a young dragon curled around a glowing crystal in a cavern."),
    ("fantasy_library", "A square magical library with floating books, spiral shelves, glowing runes, and a tiny reading desk."),
    ("fantasy_forest", "A square enchanted forest with giant luminous flowers and a narrow stone path, fairy-tale atmosphere."),
    ("fantasy_ship", "A square fantasy airship sailing through pink clouds with brass propellers and cloth sails."),
    ("surreal_clock", "A square surreal realistic scene of a melting clock draped over a marble statue in a quiet gallery."),
    ("surreal_door", "A square surreal desert scene with a red door standing alone, opening into a rainy city street."),
    ("surreal_fish", "A square surreal image of colorful fish swimming through the air above a dining table set for dinner."),
    ("surreal_teacup", "A square surreal teacup containing a tiny stormy ocean with a miniature lighthouse."),
    ("surreal_train", "A square surreal train emerging from a bookshelf, smoke turning into paper birds."),
    ("medical_lab", "A square realistic laboratory scene with microscope, sample tubes, blue gloves, and sterile lighting."),
    ("medical_plant", "A square scientific macro photo of plant cells under a microscope, green chloroplast-like structures, labeled style."),
    ("medical_robot", "A square clean hospital room with a compact medical robot delivering supplies, soft daylight, realistic."),
    ("medical_xray", "A square radiology workstation with an x-ray image on screen and a doctor taking notes, clinical lighting."),
    ("medical_vaccine", "A square close-up of vaccine vials and syringes on a stainless tray, cool blue lab light."),
    ("vehicle_car", "A square glossy red electric sports car parked on a mountain road at sunset, cinematic reflections."),
    ("vehicle_bicycle", "A square photo of a vintage bicycle leaning against a yellow wall with flowers in the basket."),
    ("vehicle_boat", "A square small wooden boat tied to a misty lake dock, early morning, calm water."),
    ("vehicle_plane", "A square view of a small propeller plane on a grassy runway, clouds and sun rays behind it."),
    ("vehicle_rover", "A square Mars rover driving over red rocks with Earth visible as a tiny blue dot in the sky."),
    ("fashion_dress", "A square fashion editorial photo of a flowing emerald dress on a mannequin in a minimalist studio."),
    ("fashion_sunglasses", "A square product photo of translucent amber sunglasses casting shadows on a peach background."),
    ("fashion_boots", "A square close-up of rugged leather boots with mud on the soles, autumn leaves around them."),
    ("fashion_hat", "A square portrait-style product image of a wide-brim black hat on a wooden stand, soft side light."),
    ("fashion_watch", "A square macro fashion photo of a gold bracelet watch beside pearl earrings on satin fabric."),
    ("sports_basketball", "A square action photo of a basketball player dunking in an outdoor court at sunset, dust in air."),
    ("sports_tennis", "A square tennis player hitting a forehand on a clay court, ball frozen near racket strings."),
    ("sports_ski", "A square skier carving through powder snow, mountain slope, bright blue sky."),
    ("sports_boxing", "A square dramatic boxing gym scene with gloves hanging from a rope, sweat, warm overhead light."),
    ("sports_climbing", "A square rock climber reaching for a hold on an indoor climbing wall, colorful holds, chalk dust."),
    ("minimal_red", "A square minimalist composition with a single red sphere on a white floor, long shadow, clean studio."),
    ("minimal_blue", "A square minimalist scene with three blue paper rectangles standing upright, precise spacing, soft light."),
    ("minimal_leaf", "A square minimal macro image of one green leaf on a beige background with a single water droplet."),
    ("minimal_stone", "A square zen still life with two smooth gray stones and one black feather on white sand."),
    ("minimal_line", "A square abstract minimalist artwork of a thin black line crossing a pale yellow circle."),
]

prompts = {name: prompt for name, prompt in _cases}

