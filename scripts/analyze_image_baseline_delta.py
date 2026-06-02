#!/usr/bin/env python3
"""Compare image-space edge/variance baselines with mask DeltaNLL maps."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--mask-dir-name", required=True)
    parser.add_argument("--out-dir-name", default=None)
    return parser.parse_args()


def write_csv(rows, path: Path):
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sample_id_from_image_path(image_path: Path):
    stem = image_path.stem
    if stem.endswith("_image_00"):
        return stem[: -len("_image_00")]
    return stem


def rankdata(x):
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def compare_maps(signal_map, delta_map):
    x = np.asarray(signal_map, dtype=np.float32).reshape(-1)
    y = np.asarray(delta_map, dtype=np.float32).reshape(-1)
    k = max(1, int(round(0.2 * len(x))))
    top_x = set(np.argsort(x)[-k:].tolist())
    top_y = set(np.argsort(y)[-k:].tolist())
    rest_idx = [i for i in range(len(y)) if i not in top_x]
    top_vals = y[list(top_x)]
    rest_vals = y[rest_idx]
    return {
        "pearson": corr(x, y),
        "spearman": corr(rankdata(x), rankdata(y)),
        "top20_overlap": len(top_x & top_y) / k,
        "delta_top20": float(top_vals.mean()),
        "delta_rest80": float(np.asarray(rest_vals).mean()),
        "top20_minus_rest": float(top_vals.mean() - np.asarray(rest_vals).mean()),
    }


def convolve2d_same(arr, kernel):
    arr = np.asarray(arr, dtype=np.float32)
    kernel = np.asarray(kernel, dtype=np.float32)
    pad_y = kernel.shape[0] // 2
    pad_x = kernel.shape[1] // 2
    padded = np.pad(arr, ((pad_y, pad_y), (pad_x, pad_x)), mode="edge")
    out = np.zeros_like(arr, dtype=np.float32)
    for y in range(arr.shape[0]):
        for x in range(arr.shape[1]):
            out[y, x] = float((padded[y : y + kernel.shape[0], x : x + kernel.shape[1]] * kernel).sum())
    return out


def image_feature_maps(image_path: Path):
    rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    kx = np.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.asarray([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    gx = convolve2d_same(gray, kx)
    gy = convolve2d_same(gray, ky)
    edge = np.sqrt(gx * gx + gy * gy)
    rgb_var = ((rgb - rgb.mean(axis=2, keepdims=True)) ** 2).mean(axis=2)
    luminance = gray
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    return {
        "edge": edge,
        "rgb_variance": rgb_var,
        "luminance": luminance,
        "chroma": chroma,
    }


def block_mean(arr, grid: int):
    arr = np.asarray(arr, dtype=np.float32)
    h, w = arr.shape
    rows = []
    for row in range(grid):
        vals = []
        y0, y1 = row * h // grid, (row + 1) * h // grid
        for col in range(grid):
            x0, x1 = col * w // grid, (col + 1) * w // grid
            vals.append(float(arr[y0:y1, x0:x1].mean()))
        rows.append(vals)
    return np.asarray(rows, dtype=np.float32)


def load_cell_maps(mask_dir: Path):
    rows = list(csv.DictReader((mask_dir / "mask_cell_metrics.csv").open(encoding="utf-8")))
    by_sample = {}
    for row in rows:
        sid = row["sample_id"]
        by_sample.setdefault(sid, []).append(row)
    maps = {}
    for sid, sample_rows in by_sample.items():
        grid = max(max(int(r["row"]), int(r["col"])) for r in sample_rows) + 1
        fields = ("coarse_ume", "delta_nll", "delta_entropy")
        sample_maps = {field: np.zeros((grid, grid), dtype=np.float32) for field in fields}
        for r in sample_rows:
            y = int(r["row"])
            x = int(r["col"])
            for field in fields:
                sample_maps[field][y, x] = float(r[field])
        maps[sid] = sample_maps
    return maps


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    mask_dir = run_dir / args.mask_dir_name
    out_dir_name = args.out_dir_name or f"{args.mask_dir_name}_image_baselines"
    out_dir = run_dir / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cell_maps = load_cell_maps(mask_dir)
    image_paths = {sample_id_from_image_path(p): p for p in sorted((run_dir / "decoded").glob("*.png"))}

    summary_rows = []
    cell_rows = []
    for sample_id, maps in sorted(cell_maps.items()):
        if sample_id not in image_paths:
            raise FileNotFoundError(f"missing decoded image for {sample_id} in {run_dir / 'decoded'}")
        grid = maps["delta_nll"].shape[0]
        features = {
            name: block_mean(arr, grid)
            for name, arr in image_feature_maps(image_paths[sample_id]).items()
        }
        for row in range(grid):
            for col in range(grid):
                cell = {
                    "sample_id": sample_id,
                    "row": row,
                    "col": col,
                    "coarse_ume": float(maps["coarse_ume"][row, col]),
                    "delta_nll": float(maps["delta_nll"][row, col]),
                    "delta_entropy": float(maps["delta_entropy"][row, col]),
                }
                for name, fmap in features.items():
                    cell[name] = float(fmap[row, col])
                cell_rows.append(cell)

        row = {"sample_id": sample_id, "grid": grid}
        for signal_name, signal_map in {"ume": maps["coarse_ume"], **features}.items():
            for delta_name in ("delta_nll", "delta_entropy"):
                stats = compare_maps(signal_map, maps[delta_name])
                prefix = f"{signal_name}_{delta_name}"
                row[f"{prefix}_pearson"] = stats["pearson"]
                row[f"{prefix}_spearman"] = stats["spearman"]
                row[f"{prefix}_top20_overlap"] = stats["top20_overlap"]
                row[f"{prefix}_top20_minus_rest"] = stats["top20_minus_rest"]
        summary_rows.append(row)

    write_csv(cell_rows, out_dir / "image_baseline_cell_metrics.csv")
    write_csv(summary_rows, out_dir / "summary_image_baselines.csv")
    print(f"[INFO] image baseline analysis saved to {out_dir}")


if __name__ == "__main__":
    main()
