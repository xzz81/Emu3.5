#!/usr/bin/env python3
"""Build deliverable panels for entropy and readback attention maps."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

EOI = 151853


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--entropy-field", default="u_tok_full")
    parser.add_argument("--top-frac", type=float, default=0.2)
    return parser.parse_args()


def heat_color(x):
    x = float(np.clip(x, 0.0, 1.0))
    if x < 0.5:
        t = x * 2.0
        return int(35 * (1 - t)), int(70 * t), int(255 * (1 - t) + 40 * t)
    t = (x - 0.5) * 2.0
    return int(255 * t + 40 * (1 - t)), int(70 * (1 - t)), int(20 * (1 - t))


def to_heatmap(arr, size=(320, 320)):
    arr = np.asarray(arr, dtype=np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    norm = (arr - lo) / (hi - lo + 1e-8)
    rgb = np.zeros((*arr.shape, 3), dtype=np.uint8)
    for y in range(arr.shape[0]):
        for x in range(arr.shape[1]):
            rgb[y, x] = heat_color(norm[y, x])
    return Image.fromarray(rgb).resize(size, Image.Resampling.NEAREST)


def load_entropy_map(trace_path: Path, field: str):
    records = []
    for line in trace_path.open(encoding="utf-8"):
        row = json.loads(line)
        records.append(row)
        if int(row.get("token_id", -1)) == EOI:
            break
    visual = [r for r in records if r.get("token_type") == "visual"]
    side = int(round(len(visual) ** 0.5))
    if side * side != len(visual):
        raise ValueError(f"{trace_path} has {len(visual)} visual tokens, not a square grid")
    return np.asarray([float(r[field]) for r in visual], dtype=np.float32).reshape(side, side)


def sample_id_from_image_path(image_path: Path):
    stem = image_path.stem
    if stem.endswith("_image_00"):
        return stem[: -len("_image_00")]
    return stem


def top_entropy_overlay(image: Image.Image, entropy_map: np.ndarray, top_frac: float, size=(320, 320)):
    base = image.convert("RGB").resize(size)
    flat = entropy_map.reshape(-1)
    k = max(1, int(round(len(flat) * top_frac)))
    mask = np.zeros_like(flat, dtype=np.uint8)
    mask[np.argsort(flat)[-k:]] = 1
    mask = mask.reshape(entropy_map.shape)
    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    cell_w = size[0] / mask.shape[1]
    cell_h = size[1] / mask.shape[0]
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if mask[y, x]:
                draw.rectangle(
                    (x * cell_w, y * cell_h, (x + 1) * cell_w, (y + 1) * cell_h),
                    fill=(255, 0, 0, 105),
                )
    return Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")


def label_cell(image, label, sublabel=None, size=(320, 366)):
    cell = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(cell)
    draw.text((8, 8), label, fill=(0, 0, 0))
    if sublabel:
        draw.text((8, 26), sublabel, fill=(70, 70, 70))
    cell.paste(image.resize((size[0], size[0])), (0, size[1] - size[0]))
    return cell


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    relation_dir = run_dir / "relation_analysis"
    out_dir = run_dir / "deliverables"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    with (relation_dir / "summary_best_layers.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            summary[row["sample_id"]] = row

    panels = []
    metric_rows = []
    for image_path in sorted((run_dir / "decoded").glob("*.png")):
        sample_id = sample_id_from_image_path(image_path)
        image = Image.open(image_path).convert("RGB")
        entropy_map = load_entropy_map(run_dir / "entropy_traces" / f"{sample_id}_entropy.jsonl", args.entropy_field)
        layer_paths = sorted(relation_dir.glob(f"{sample_id}_layer_*_attention.npy"))
        attention_maps = [np.load(path) for path in layer_paths]
        if not attention_maps:
            continue
        last_layer = len(attention_maps) - 1
        best_layer = int(summary[sample_id]["best_abs_corr_layer"])
        mean_map = np.mean(np.stack(attention_maps, axis=0), axis=0)

        original = image.resize((320, 320))
        entropy_top20 = top_entropy_overlay(image, entropy_map, args.top_frac)
        last_attention = to_heatmap(attention_maps[last_layer])
        best_attention = to_heatmap(attention_maps[best_layer])
        mean_attention = to_heatmap(mean_map)

        original.save(out_dir / f"{sample_id}_original.png")
        entropy_top20.save(out_dir / f"{sample_id}_top20_entropy_overlay.png")
        last_attention.save(out_dir / f"{sample_id}_attention_last_layer.png")
        best_attention.save(out_dir / f"{sample_id}_attention_best_corr_layer.png")
        mean_attention.save(out_dir / f"{sample_id}_attention_mean_all_layers.png")

        row = summary[sample_id]
        metric_rows.append({
            "sample_id": sample_id,
            "best_corr_layer": best_layer,
            "pearson": row["pearson_entropy_attention"],
            "spearman": row["spearman_entropy_attention"],
            "top20_overlap_fraction": row["top20_overlap_fraction"],
            "top20_attention_enrichment": row["top20_attention_enrichment"],
        })

        cells = [
            label_cell(original, "original", sample_id),
            label_cell(entropy_top20, "top 20% generation entropy", args.entropy_field),
            label_cell(last_attention, "attention heatmap", f"last layer {last_layer}"),
            label_cell(best_attention, "attention heatmap", f"best corr layer {best_layer}, r={float(row['pearson_entropy_attention']):+.3f}"),
            label_cell(mean_attention, "attention heatmap", "mean over all layers"),
        ]
        panel = Image.new("RGB", (cells[0].width * len(cells), cells[0].height), "white")
        for idx, cell in enumerate(cells):
            panel.paste(cell, (idx * cell.width, 0))
        panel_path = out_dir / f"{sample_id}_deliverable_panel.png"
        panel.save(panel_path)
        panels.append(panel)

    if panels:
        overview = Image.new("RGB", (panels[0].width, panels[0].height * len(panels)), "white")
        for idx, panel in enumerate(panels):
            overview.paste(panel, (0, idx * panel.height))
        overview.save(out_dir / "all_samples_deliverable_overview.png")

    with (out_dir / "entropy_attention_relation_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)
    print(f"[INFO] deliverables saved to {out_dir}")


if __name__ == "__main__":
    main()
