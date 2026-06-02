#!/usr/bin/env python3
"""Create reproducible text-band masked variants of the real poster GT images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


# Fractions are x0, y0, x1, y1. They intentionally cover common poster
# tagline/title/credits bands while preserving the main visual figure area.
MASK_SPECS = {
    2: [(0.0, 0.00, 1.0, 0.18), (0.0, 0.66, 1.0, 1.00)],
    3: [(0.0, 0.00, 1.0, 0.22), (0.0, 0.64, 1.0, 1.00)],
    7: [(0.0, 0.00, 1.0, 0.22), (0.0, 0.66, 1.0, 1.00)],
    9: [(0.0, 0.00, 1.0, 0.17), (0.0, 0.64, 1.0, 1.00)],
    10: [(0.0, 0.00, 1.0, 0.20), (0.0, 0.62, 1.0, 1.00)],
    14: [(0.0, 0.00, 1.0, 0.18), (0.0, 0.56, 1.0, 1.00)],
    15: [(0.0, 0.00, 1.0, 0.30), (0.0, 0.66, 1.0, 1.00)],
    17: [(0.0, 0.00, 1.0, 0.12), (0.0, 0.63, 1.0, 1.00)],
    19: [(0.0, 0.00, 1.0, 0.12), (0.0, 0.64, 1.0, 1.00)],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="../research_logs/modal_aphasia_posters/gt_posters_wikipedia_20260531/manifest.json",
    )
    parser.add_argument(
        "--out-dir",
        default="../research_logs/modal_aphasia_posters/gt_posters_text_band_masked_seed75",
    )
    return parser.parse_args()


def rect_pixels(spec: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = spec
    return (
        int(round(x0 * width)),
        int(round(y0 * height)),
        int(round(x1 * width)),
        int(round(y1 * height)),
    )


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    out_rows = []
    for row in rows:
        poster_id = int(row["poster_id"])
        image_path = Path(row["local_path"])
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        draw = ImageDraw.Draw(image)
        specs = MASK_SPECS.get(poster_id, [(0.0, 0.0, 1.0, 0.18), (0.0, 0.65, 1.0, 1.0)])
        pixel_rects = []
        for spec in specs:
            rect = rect_pixels(spec, width, height)
            pixel_rects.append(rect)
            draw.rectangle(rect, fill=(128, 128, 128))
        out_path = out_dir / image_path.name
        image.save(out_path)
        out_rows.append(
            {
                **row,
                "original_local_path": row["local_path"],
                "local_path": str(out_path),
                "mask_rects_fraction": specs,
                "mask_rects_pixels": pixel_rects,
                "mask_color_rgb": [128, 128, 128],
            }
        )
    (out_dir / "manifest.json").write_text(json.dumps(out_rows, indent=2), encoding="utf-8")
    print(f"[INFO] wrote {len(out_rows)} masked posters to {out_dir}")


if __name__ == "__main__":
    main()
