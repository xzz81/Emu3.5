#!/usr/bin/env python3
"""Build target+distractor images where distractor has the same shape but different color."""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw


OUT_DIR = Path("assets/ume_image_read_same_shape_distractor_seed30")
SIZE = 512
SHAPE_SIZE = 104
BG = (148, 148, 148)
OUTLINE = (30, 30, 30)

POSITIONS = {
    "upper-left": (64, 64),
    "upper-right": (SIZE - 64 - SHAPE_SIZE, 64),
    "lower-left": (64, SIZE - 64 - SHAPE_SIZE),
    "lower-right": (SIZE - 64 - SHAPE_SIZE, SIZE - 64 - SHAPE_SIZE),
}

OPPOSITE = {
    "upper-left": "lower-right",
    "upper-right": "lower-left",
    "lower-left": "upper-right",
    "lower-right": "upper-left",
}

TARGETS = [
    ("red circle", "circle", (218, 38, 38), "blue circle", (35, 95, 215)),
    ("blue square", "square", (35, 95, 215), "yellow square", (238, 204, 42)),
    ("yellow triangle", "triangle", (238, 204, 42), "red triangle", (218, 38, 38)),
]

POS_PREFIX = {
    "upper-left": "ul",
    "upper-right": "ur",
    "lower-left": "ll",
    "lower-right": "lr",
}


def draw_shape(draw: ImageDraw.ImageDraw, kind: str, xy: tuple[int, int, int, int], fill: tuple[int, int, int]) -> None:
    if kind == "circle":
        draw.ellipse(xy, fill=fill, outline=OUTLINE, width=4)
    elif kind == "square":
        draw.rectangle(xy, fill=fill, outline=OUTLINE, width=4)
    elif kind == "triangle":
        x0, y0, x1, y1 = xy
        points = [(x0 + (x1 - x0) // 2, y0), (x0, y1), (x1, y1)]
        draw.polygon(points, fill=fill, outline=OUTLINE)
        draw.line(points + [points[0]], fill=OUTLINE, width=4)
    else:
        raise ValueError(f"unknown shape kind: {kind}")


def bbox(position: str) -> tuple[int, int, int, int]:
    x0, y0 = POSITIONS[position]
    return x0, y0, x0 + SHAPE_SIZE, y0 + SHAPE_SIZE


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for object_phrase, kind, color, distractor_phrase, distractor_color in TARGETS:
        color_name, noun = object_phrase.split()
        for position_phrase in POSITIONS:
            distractor_position = OPPOSITE[position_phrase]
            key = f"{POS_PREFIX[position_phrase]}_{color_name}_{noun}"
            image = Image.new("RGB", (SIZE, SIZE), BG)
            draw = ImageDraw.Draw(image)
            draw_shape(draw, kind, bbox(distractor_position), distractor_color)
            draw_shape(draw, kind, bbox(position_phrase), color)
            image_path = OUT_DIR / f"{key}.png"
            image.save(image_path)
            x0, y0, x1, y1 = bbox(position_phrase)
            dx0, dy0, dx1, dy1 = bbox(distractor_position)
            rows.append(
                {
                    "sample_suffix": key,
                    "object": object_phrase,
                    "target_position": position_phrase,
                    "reference_image": str(image_path),
                    "distractor_object": distractor_phrase,
                    "distractor_position": distractor_position,
                    "ref_x0": x0,
                    "ref_y0": y0,
                    "ref_x1": x1,
                    "ref_y1": y1,
                    "distractor_x0": dx0,
                    "distractor_y0": dy0,
                    "distractor_x1": dx1,
                    "distractor_y1": dy1,
                }
            )
    with (OUT_DIR / "reference_layout_metadata.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] wrote {len(rows)} same-shape distractor images to {OUT_DIR}")


if __name__ == "__main__":
    main()
