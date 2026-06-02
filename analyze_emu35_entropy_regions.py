#!/usr/bin/env python3
"""Analyze per-token entropy regions for Emu3.5 synthetic concept generations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ATTRS = ("color", "pattern", "position", "shape")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--top-frac", type=float, default=0.20)
    parser.add_argument("--grid-size", type=int, default=24)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--max-overlays", type=int, default=24)
    return parser.parse_args()


def load_jsonl_rows(bench_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(bench_dir.glob("image_memory_worker*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line:
                rows.append(json.loads(line))
    return rows


def entropy_grid(path: Path, grid_size: int) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    grid = np.full((grid_size, grid_size), np.nan, dtype=np.float32)
    for step in payload["steps"]:
        if step.get("phase") != "visual":
            continue
        idx = step.get("visual_index")
        if idx is None:
            continue
        idx = int(idx)
        if not 0 <= idx < grid_size * grid_size:
            continue
        grid[idx // grid_size, idx % grid_size] = float(step["entropy"])
    return grid


def downsample_mask(mask: np.ndarray, grid_size: int) -> np.ndarray:
    h, w = mask.shape
    cell_h = h // grid_size
    cell_w = w // grid_size
    mask = mask[: grid_size * cell_h, : grid_size * cell_w]
    return mask.reshape(grid_size, cell_h, grid_size, cell_w).mean(axis=(1, 3))


def foreground_mask(image_path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(Image.open(image_path).convert("RGB"))
    # Synthetic foreground shape is dark/black; use a conservative threshold.
    fg = np.all(image < 80, axis=-1)
    padded = np.pad(fg, 1, mode="edge")
    neighbors = [
        padded[:-2, 1:-1],
        padded[2:, 1:-1],
        padded[1:-1, :-2],
        padded[1:-1, 2:],
        padded[:-2, :-2],
        padded[:-2, 2:],
        padded[2:, :-2],
        padded[2:, 2:],
    ]
    eroded = fg.copy()
    dilated = fg.copy()
    for item in neighbors:
        eroded &= item
        dilated |= item
    boundary = dilated & ~eroded
    return fg, boundary


def position_mask(position: str, grid_size: int) -> np.ndarray:
    mask = np.zeros((grid_size, grid_size), dtype=bool)
    r0, r1 = (0, grid_size // 2) if "top" in position else (grid_size // 2, grid_size)
    c0, c1 = (0, grid_size // 2) if "left" in position else (grid_size // 2, grid_size)
    mask[r0:r1, c0:c1] = True
    return mask


def top_mask(grid: np.ndarray, top_frac: float) -> np.ndarray:
    values = grid[np.isfinite(grid)]
    threshold = np.quantile(values, 1.0 - top_frac)
    return np.isfinite(grid) & (grid >= threshold)


def safe_mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def overlap_fraction(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    denom = float(mask_a.sum())
    if denom == 0:
        return float("nan")
    return float((mask_a & mask_b).sum() / denom)


def save_heatmap(array: np.ndarray, path: Path, title: str, vmax: float | None = None) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.6), dpi=180)
    im = ax.imshow(array, cmap="magma", vmin=0, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_overlay(image_path: Path, grid: np.ndarray, top: np.ndarray, out_path: Path) -> None:
    image = Image.open(image_path).convert("RGB").resize((384, 384))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), dpi=160)
    axes[0].imshow(image)
    axes[0].set_title("generated")
    axes[1].imshow(image)
    axes[1].imshow(grid, cmap="magma", alpha=0.55, extent=(0, 384, 384, 0))
    axes[1].set_title("entropy")
    axes[2].imshow(image)
    axes[2].imshow(top, cmap="Reds", alpha=0.45, extent=(0, 384, 384, 0), vmin=0, vmax=1)
    axes[2].set_title("top 20%")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    bench_dir = Path(args.bench_dir)
    out_dir = Path(args.output_dir) if args.output_dir else bench_dir / "entropy_region_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = out_dir / "overlays"
    overlay_dir.mkdir(exist_ok=True)

    rows = load_jsonl_rows(bench_dir)
    sample_rows = []
    heatmaps = []
    top_masks = []
    overlay_count = 0

    for row in rows:
        entropy_path = row.get("entropy_path")
        image_path = row.get("image_path")
        if not entropy_path or not image_path:
            continue
        entropy_path = Path(entropy_path)
        image_path = Path(image_path)
        if not entropy_path.exists() or not image_path.exists():
            continue

        grid = entropy_grid(entropy_path, args.grid_size)
        top = top_mask(grid, args.top_frac)
        fg, boundary = foreground_mask(image_path)
        fg_grid = downsample_mask(fg, args.grid_size) >= 0.05
        boundary_grid = downsample_mask(boundary, args.grid_size) >= 0.01
        pos_grid = position_mask(row["position"], args.grid_size)
        heatmaps.append(grid)
        top_masks.append(top.astype(np.float32))

        selected = grid[top]
        sample = {
            "sample_id": row["sample_id"],
            "split": row["split"],
            "image_path": row["image_path"],
            "entropy_path": row["entropy_path"],
            "mean_entropy": float(np.nanmean(grid)),
            "top20_mean_entropy": float(np.nanmean(selected)),
            "top20_min_entropy": float(np.nanmin(selected)),
            "top20_foreground_overlap": overlap_fraction(top, fg_grid),
            "top20_boundary_overlap": overlap_fraction(top, boundary_grid),
            "top20_target_position_overlap": overlap_fraction(top, pos_grid),
            "foreground_top20_coverage": overlap_fraction(fg_grid, top),
            "boundary_top20_coverage": overlap_fraction(boundary_grid, top),
            "target_position_top20_coverage": overlap_fraction(pos_grid, top),
            **{attr: row[attr] for attr in ATTRS},
            **{f"grading_correct_{attr}": bool(row.get(f"grading_correct_{attr}")) for attr in ATTRS},
            "grading_all_correct": bool(row.get("grading_all_correct")),
        }
        sample_rows.append(sample)

        if overlay_count < args.max_overlays:
            save_overlay(image_path, grid, top, overlay_dir / f"{row['sample_id']}.png")
            overlay_count += 1

    fieldnames = list(sample_rows[0].keys())
    with (out_dir / "sample_entropy_region_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sample_rows)

    heat = np.stack(heatmaps)
    top_stack = np.stack(top_masks)
    mean_heat = np.nanmean(heat, axis=0)
    top_freq = np.nanmean(top_stack, axis=0)
    np.save(out_dir / "mean_entropy_grid.npy", mean_heat)
    np.save(out_dir / "top20_frequency_grid.npy", top_freq)
    save_heatmap(mean_heat, out_dir / "mean_entropy_grid.png", "Mean visual-token entropy")
    save_heatmap(top_freq, out_dir / "top20_frequency_grid.png", "Top-20% entropy frequency", vmax=1.0)

    aggregate = {
        "num_samples": len(sample_rows),
        "top_frac": args.top_frac,
        "mean_entropy": safe_mean([row["mean_entropy"] for row in sample_rows]),
        "mean_top20_foreground_overlap": safe_mean([row["top20_foreground_overlap"] for row in sample_rows]),
        "mean_top20_boundary_overlap": safe_mean([row["top20_boundary_overlap"] for row in sample_rows]),
        "mean_top20_target_position_overlap": safe_mean([row["top20_target_position_overlap"] for row in sample_rows]),
        "by_correctness": {},
    }
    for key in ("grading_all_correct",) + tuple(f"grading_correct_{attr}" for attr in ATTRS):
        aggregate["by_correctness"][key] = {}
        for value in (True, False):
            selected = [row for row in sample_rows if row[key] is value]
            aggregate["by_correctness"][key][str(value).lower()] = {
                "n": len(selected),
                "mean_entropy": safe_mean([row["mean_entropy"] for row in selected]),
                "top20_foreground_overlap": safe_mean([row["top20_foreground_overlap"] for row in selected]),
                "top20_boundary_overlap": safe_mean([row["top20_boundary_overlap"] for row in selected]),
                "top20_target_position_overlap": safe_mean([row["top20_target_position_overlap"] for row in selected]),
            }
    (out_dir / "region_summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
