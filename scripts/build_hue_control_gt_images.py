#!/usr/bin/env python3
"""Build exactly matched GT hue-control icon pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


COLORS = {
    "red": (226, 42, 50),
    "blue": (47, 166, 226),
    "green": (35, 166, 86),
    "purple": (137, 88, 190),
    "yellow": (255, 218, 43),
    "cyan": (73, 205, 199),
    "orange": (241, 139, 39),
    "magenta": (224, 38, 180),
}


PAIR_SPECS = [
    ("gt_hue_circle_red_vs_blue", "circle", "red", "blue"),
    ("gt_hue_square_green_vs_purple", "square", "green", "purple"),
    ("gt_hue_triangle_yellow_vs_cyan", "triangle", "yellow", "cyan"),
    ("gt_hue_diamond_orange_vs_magenta", "diamond", "orange", "magenta"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/hue_control_gt_images_seed70")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--scale", type=int, default=4)
    return parser.parse_args()


def polygon_for_shape(shape: str, size: int) -> list[tuple[int, int]]:
    cx = cy = size // 2
    r = int(size * 0.28)
    if shape == "square":
        return [(cx - r, cy - r), (cx + r, cy - r), (cx + r, cy + r), (cx - r, cy + r)]
    if shape == "triangle":
        return [(cx, cy - r), (cx + int(r * 0.92), cy + int(r * 0.72)), (cx - int(r * 0.92), cy + int(r * 0.72))]
    if shape == "diamond":
        return [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
    raise ValueError(f"unknown polygon shape {shape}")


def draw_icon(shape: str, color_name: str, size: int, scale: int) -> Image.Image:
    canvas_size = size * scale
    image = Image.new("RGB", (canvas_size, canvas_size), "white")
    draw = ImageDraw.Draw(image)
    fill = COLORS[color_name]
    outline = (20, 24, 28)
    width = max(4, int(size * 0.018)) * scale
    margin = int(size * 0.22) * scale
    if shape == "circle":
        box = (margin, margin, canvas_size - margin, canvas_size - margin)
        draw.ellipse(box, fill=fill, outline=outline, width=width)
    else:
        points = [(x * scale, y * scale) for x, y in polygon_for_shape(shape, size)]
        draw.polygon(points, fill=fill)
        draw.line(points + [points[0]], fill=outline, width=width, joint="curve")
    return image.resize((size, size), Image.Resampling.LANCZOS)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for pair_id, shape, color_a, color_b in PAIR_SPECS:
        for variant, color in (("a", color_a), ("b", color_b)):
            image_id = f"{pair_id}__{variant}"
            path = image_dir / f"{image_id}.png"
            draw_icon(shape, color, args.size, args.scale).save(path)
            rows.append(
                {
                    "image_id": image_id,
                    "pair_id": pair_id,
                    "variant": variant,
                    "shape": shape,
                    "fill_color": color,
                    "allowed_colors": f"{color};black;white",
                    "local_path": str(path),
                    "gt_description": f"a centered {color} {shape} with a black outline on a white background",
                }
            )
    (out_dir / "manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"[INFO] wrote {len(rows)} GT hue-control images to {out_dir}")


if __name__ == "__main__":
    main()
