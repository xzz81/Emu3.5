#!/usr/bin/env python3
"""Debias UME against post-hoc foreground/object-region covariates.

This is intended for simple object-on-background placement probes. It estimates a
foreground mask from color distance to the image border background, aggregates
mask statistics to each mask cell, and compares UME after residualizing foreground
and image/position covariates.
"""

from __future__ import annotations

import argparse
import csv
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image


SIGNALS = (
    "ume",
    "ume_resid_foreground",
    "ume_resid_image",
    "ume_resid_position",
    "ume_resid_image_position",
    "ume_resid_image_foreground",
    "ume_resid_position_foreground",
    "ume_resid_image_position_foreground",
    "foreground_fraction",
    "foreground_luminance_contrast",
    "foreground_edge",
    "foreground_distance",
    "luminance",
    "edge",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--run-mask",
        action="append",
        required=True,
        help="Pair in the form RUN_DIR:MASK_DIR_NAME.",
    )
    parser.add_argument("--save-foreground-masks", action="store_true")
    return parser.parse_args()


def write_csv(rows, path: Path):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path):
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def standardize(x):
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean()) / (x.std() + 1e-12)


def residualize(y, columns):
    y = np.asarray(y, dtype=np.float64)
    if not columns:
        return y - y.mean()
    x = np.column_stack([np.ones(len(y)), *[standardize(col) for col in columns]])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return y - x @ beta


def compare_signal(signal, delta):
    x = np.asarray(signal, dtype=np.float64)
    y = np.asarray(delta, dtype=np.float64)
    k = max(1, int(round(0.2 * len(x))))
    top_x = set(np.argsort(x)[-k:].tolist())
    top_y = set(np.argsort(y)[-k:].tolist())
    rest_idx = [i for i in range(len(y)) if i not in top_x]
    return {
        "pearson": corr(x, y),
        "spearman": corr(rankdata(x), rankdata(y)),
        "top20_overlap": len(top_x & top_y) / k,
        "top20_minus_rest": float(y[list(top_x)].mean() - y[rest_idx].mean()),
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


def edge_map(gray):
    kx = np.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.asarray([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    gx = convolve2d_same(gray, kx)
    gy = convolve2d_same(gray, ky)
    return np.sqrt(gx * gx + gy * gy)


def connected_components(mask):
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or seen[y, x]:
                continue
            q = deque([(y, x)])
            seen[y, x] = True
            pixels = []
            while q:
                cy, cx = q.popleft()
                pixels.append((cy, cx))
                for ny in (cy - 1, cy, cy + 1):
                    for nx in (cx - 1, cx, cx + 1):
                        if ny == cy and nx == cx:
                            continue
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            q.append((ny, nx))
            comps.append(pixels)
    return comps


def estimate_foreground(image_path: Path):
    rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    h, w = gray.shape
    border = max(8, min(h, w) // 24)
    border_pixels = np.concatenate(
        [
            rgb[:border].reshape(-1, 3),
            rgb[-border:].reshape(-1, 3),
            rgb[:, :border].reshape(-1, 3),
            rgb[:, -border:].reshape(-1, 3),
        ],
        axis=0,
    )
    bg = np.median(border_pixels, axis=0)
    dist = np.linalg.norm(rgb - bg[None, None, :], axis=2)
    med = float(np.median(dist))
    mad = float(np.median(np.abs(dist - med)))
    threshold = max(med + 5.0 * mad, float(np.quantile(dist, 0.92)), 0.060)
    bg_lum = float(0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2])
    lum_contrast = np.abs(gray - bg_lum)
    # Avoid smooth background gradients by requiring colorfulness or a stronger
    # luminance deviation in addition to RGB distance.
    raw_mask = (dist > threshold) & ((chroma > 0.035) | (lum_contrast > 0.090))

    comps = connected_components(raw_mask)
    min_area = max(24, int(0.0002 * h * w))
    max_area = int(0.35 * h * w)
    kept = [comp for comp in comps if min_area <= len(comp) <= max_area]
    edge = edge_map(gray)

    def component_score(comp):
        ys = np.asarray([p[0] for p in comp], dtype=np.int64)
        xs = np.asarray([p[1] for p in comp], dtype=np.int64)
        mean_dist = float(dist[ys, xs].mean())
        mean_chroma = float(chroma[ys, xs].mean())
        mean_edge = float(edge[ys, xs].mean())
        touches = int(ys.min() <= 2) + int(xs.min() <= 2) + int(ys.max() >= h - 3) + int(xs.max() >= w - 3)
        border_penalty = 0.45 if touches >= 2 else 1.0
        return len(comp) * (mean_dist + 0.5 * mean_chroma + 0.25 * mean_edge) * border_penalty

    kept.sort(key=component_score, reverse=True)

    # Preserve split object parts/highlights but avoid admitting broad background bands.
    max_components = 4
    mask = np.zeros((h, w), dtype=bool)
    for comp in kept[:max_components]:
        ys = [p[0] for p in comp]
        xs = [p[1] for p in comp]
        bbox_area = (max(ys) - min(ys) + 1) * (max(xs) - min(xs) + 1)
        if bbox_area > 0.45 * h * w:
            continue
        touches = int(min(ys) <= 2) + int(min(xs) <= 2) + int(max(ys) >= h - 3) + int(max(xs) >= w - 3)
        mean_chroma = float(chroma[ys, xs].mean())
        mean_edge = float(edge[ys, xs].mean())
        if touches >= 2 and mean_chroma < 0.030 and mean_edge < 0.025:
            continue
        for y, x in comp:
            mask[y, x] = True

    return {
        "mask": mask.astype(np.float32),
        "distance": dist.astype(np.float32),
        "edge": edge.astype(np.float32),
        "luminance_contrast": lum_contrast.astype(np.float32),
        "background_luminance": bg_lum,
        "foreground_area_fraction": float(mask.mean()),
        "foreground_threshold": threshold,
    }


def block_mean(arr, grid):
    arr = np.asarray(arr, dtype=np.float32)
    h, w = arr.shape
    out = np.zeros((grid, grid), dtype=np.float32)
    for row in range(grid):
        y0, y1 = row * h // grid, (row + 1) * h // grid
        for col in range(grid):
            x0, x1 = col * w // grid, (col + 1) * w // grid
            out[row, col] = float(arr[y0:y1, x0:x1].mean())
    return out


def build_signals(rows, fg_maps):
    rows = sorted(rows, key=lambda r: (int(r["row"]), int(r["col"])))
    grid = max(max(int(r["row"]), int(r["col"])) for r in rows) + 1
    row_pos = np.asarray([int(r["row"]) for r in rows], dtype=np.float64)
    col_pos = np.asarray([int(r["col"]) for r in rows], dtype=np.float64)
    center = (grid - 1) / 2.0
    dist_center = np.sqrt((row_pos - center) ** 2 + (col_pos - center) ** 2)

    ume = np.asarray([float(r["coarse_ume"]) for r in rows], dtype=np.float64)
    edge = np.asarray([float(r["edge"]) for r in rows], dtype=np.float64)
    rgb_variance = np.asarray([float(r["rgb_variance"]) for r in rows], dtype=np.float64)
    luminance = np.asarray([float(r["luminance"]) for r in rows], dtype=np.float64)
    chroma = np.asarray([float(r["chroma"]) for r in rows], dtype=np.float64)

    fg_fraction_map = block_mean(fg_maps["mask"], grid)
    fg_lum_map = block_mean(fg_maps["luminance_contrast"] * fg_maps["mask"], grid)
    fg_edge_map = block_mean(fg_maps["edge"] * fg_maps["mask"], grid)
    fg_dist_map = block_mean(fg_maps["distance"] * fg_maps["mask"], grid)
    fg_fraction = fg_fraction_map.reshape(-1).astype(np.float64)
    fg_lum = fg_lum_map.reshape(-1).astype(np.float64)
    fg_edge = fg_edge_map.reshape(-1).astype(np.float64)
    fg_dist = fg_dist_map.reshape(-1).astype(np.float64)
    fg_area = np.full_like(fg_fraction, float(fg_maps["foreground_area_fraction"]))

    image_columns = [edge, rgb_variance, luminance, chroma]
    position_columns = [row_pos, col_pos, dist_center]
    foreground_columns = [fg_fraction, fg_lum, fg_edge, fg_dist, fg_area]
    return rows, grid, {
        "ume": ume,
        "ume_resid_foreground": residualize(ume, foreground_columns),
        "ume_resid_image": residualize(ume, image_columns),
        "ume_resid_position": residualize(ume, position_columns),
        "ume_resid_image_position": residualize(ume, image_columns + position_columns),
        "ume_resid_image_foreground": residualize(ume, image_columns + foreground_columns),
        "ume_resid_position_foreground": residualize(ume, position_columns + foreground_columns),
        "ume_resid_image_position_foreground": residualize(
            ume, image_columns + position_columns + foreground_columns
        ),
        "foreground_fraction": fg_fraction,
        "foreground_luminance_contrast": fg_lum,
        "foreground_edge": fg_edge,
        "foreground_distance": fg_dist,
        "luminance": luminance,
        "edge": edge,
    }, fg_fraction_map


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    sample_rows = []
    cell_rows = []
    fg_audit_rows = []

    for spec in args.run_mask:
        run_text, mask_name = spec.split(":", 1)
        run_dir = Path(run_text)
        cell_path = run_dir / f"{mask_name}_image_baselines" / "image_baseline_cell_metrics.csv"
        rows = read_rows(cell_path)
        by_sample = {}
        for row in rows:
            by_sample.setdefault(row["sample_id"], []).append(row)
        image_paths = {sample_id_from_image_path(p): p for p in sorted((run_dir / "decoded").glob("*.png"))}

        for sample_id, group in sorted(by_sample.items()):
            if sample_id not in image_paths:
                raise FileNotFoundError(f"missing decoded image for {sample_id}")
            fg = estimate_foreground(image_paths[sample_id])
            sorted_rows, grid, signals, fg_fraction_map = build_signals(group, fg)
            delta_nll = np.asarray([float(r["delta_nll"]) for r in sorted_rows], dtype=np.float64)
            delta_entropy = np.asarray([float(r["delta_entropy"]) for r in sorted_rows], dtype=np.float64)

            summary = {
                "run_id": run_dir.name,
                "mask_dir_name": mask_name,
                "grid": grid,
                "sample_id": sample_id,
                "foreground_area_fraction": fg["foreground_area_fraction"],
                "foreground_threshold": fg["foreground_threshold"],
                "background_luminance": fg["background_luminance"],
            }
            for signal_name in SIGNALS:
                for delta_name, delta in (("delta_nll", delta_nll), ("delta_entropy", delta_entropy)):
                    for key, value in compare_signal(signals[signal_name], delta).items():
                        summary[f"{signal_name}_{delta_name}_{key}"] = value
            sample_rows.append(summary)

            for idx, row in enumerate(sorted_rows):
                out = {
                    "run_id": run_dir.name,
                    "mask_dir_name": mask_name,
                    "grid": grid,
                    "sample_id": sample_id,
                    "row": row["row"],
                    "col": row["col"],
                    "delta_nll": row["delta_nll"],
                    "delta_entropy": row["delta_entropy"],
                }
                for signal_name in SIGNALS:
                    out[signal_name] = float(signals[signal_name][idx])
                cell_rows.append(out)

            fg_audit_rows.append(
                {
                    "run_id": run_dir.name,
                    "mask_dir_name": mask_name,
                    "sample_id": sample_id,
                    "foreground_area_fraction": fg["foreground_area_fraction"],
                    "foreground_threshold": fg["foreground_threshold"],
                    "background_luminance": fg["background_luminance"],
                    "max_cell_foreground_fraction": float(fg_fraction_map.max()),
                    "mean_cell_foreground_fraction": float(fg_fraction_map.mean()),
                }
            )
            if args.save_foreground_masks:
                mask_img = Image.fromarray((fg["mask"] * 255).astype(np.uint8), mode="L")
                mask_dir = out_dir / "foreground_masks"
                mask_dir.mkdir(parents=True, exist_ok=True)
                mask_img.save(mask_dir / f"{sample_id}_foreground_mask.png")

    write_csv(sample_rows, out_dir / "summary_foreground_object_debias.csv")
    write_csv(cell_rows, out_dir / "foreground_object_debias_cell_metrics.csv")
    write_csv(fg_audit_rows, out_dir / "foreground_mask_audit.csv")
    print(f"[INFO] wrote {len(sample_rows)} sample rows and {len(cell_rows)} cell rows to {out_dir}")


if __name__ == "__main__":
    main()
