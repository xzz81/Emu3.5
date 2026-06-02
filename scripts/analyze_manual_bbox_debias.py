#!/usr/bin/env python3
"""Debias UME against manually annotated object bounding boxes."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


SIGNALS = (
    "ume",
    "ume_resid_bbox",
    "ume_resid_image",
    "ume_resid_position",
    "ume_resid_image_position",
    "ume_resid_image_bbox",
    "ume_resid_position_bbox",
    "ume_resid_image_position_bbox",
    "bbox_fraction",
    "bbox_luminance_contrast",
    "bbox_edge",
    "bbox_rgb_distance",
    "luminance",
    "edge",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-mask", action="append", required=True)
    parser.add_argument("--save-bbox-masks", action="store_true")
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


def load_bboxes(path: Path):
    bboxes = {}
    for row in read_rows(path):
        bboxes[row["sample_id"]] = {
            "x0": int(row["x0"]),
            "y0": int(row["y0"]),
            "x1": int(row["x1"]),
            "y1": int(row["y1"]),
            "bbox_note": row.get("bbox_note", ""),
        }
    return bboxes


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


def bbox_maps(image_path: Path, bbox):
    rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
    h, w = rgb.shape[:2]
    x0 = max(0, min(w, int(bbox["x0"])))
    x1 = max(0, min(w, int(bbox["x1"])))
    y0 = max(0, min(h, int(bbox["y0"])))
    y1 = max(0, min(h, int(bbox["y1"])))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"invalid bbox for {image_path}: {bbox}")

    mask = np.zeros((h, w), dtype=np.float32)
    mask[y0:y1, x0:x1] = 1.0
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
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
    bg_lum = float(0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2])
    dist = np.linalg.norm(rgb - bg[None, None, :], axis=2)
    lum_contrast = np.abs(gray - bg_lum)
    edge = edge_map(gray)
    return {
        "mask": mask,
        "luminance_contrast": lum_contrast.astype(np.float32),
        "rgb_distance": dist.astype(np.float32),
        "edge": edge.astype(np.float32),
        "bbox_area_fraction": float(mask.mean()),
        "background_luminance": bg_lum,
    }


def build_signals(rows, maps):
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

    bbox_fraction = block_mean(maps["mask"], grid).reshape(-1).astype(np.float64)
    bbox_lum = block_mean(maps["luminance_contrast"] * maps["mask"], grid).reshape(-1).astype(np.float64)
    bbox_edge = block_mean(maps["edge"] * maps["mask"], grid).reshape(-1).astype(np.float64)
    bbox_dist = block_mean(maps["rgb_distance"] * maps["mask"], grid).reshape(-1).astype(np.float64)
    bbox_area = np.full_like(bbox_fraction, float(maps["bbox_area_fraction"]))

    image_columns = [edge, rgb_variance, luminance, chroma]
    position_columns = [row_pos, col_pos, dist_center]
    bbox_columns = [bbox_fraction, bbox_lum, bbox_edge, bbox_dist, bbox_area]
    return rows, grid, {
        "ume": ume,
        "ume_resid_bbox": residualize(ume, bbox_columns),
        "ume_resid_image": residualize(ume, image_columns),
        "ume_resid_position": residualize(ume, position_columns),
        "ume_resid_image_position": residualize(ume, image_columns + position_columns),
        "ume_resid_image_bbox": residualize(ume, image_columns + bbox_columns),
        "ume_resid_position_bbox": residualize(ume, position_columns + bbox_columns),
        "ume_resid_image_position_bbox": residualize(ume, image_columns + position_columns + bbox_columns),
        "bbox_fraction": bbox_fraction,
        "bbox_luminance_contrast": bbox_lum,
        "bbox_edge": bbox_edge,
        "bbox_rgb_distance": bbox_dist,
        "luminance": luminance,
        "edge": edge,
    }


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    bboxes = load_bboxes(Path(args.bbox_csv))
    sample_rows = []
    cell_rows = []
    audit_rows = []

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
            if sample_id not in bboxes:
                raise KeyError(f"missing bbox for {sample_id}")
            if sample_id not in image_paths:
                raise FileNotFoundError(f"missing decoded image for {sample_id}")
            maps = bbox_maps(image_paths[sample_id], bboxes[sample_id])
            sorted_rows, grid, signals = build_signals(group, maps)
            delta_nll = np.asarray([float(r["delta_nll"]) for r in sorted_rows], dtype=np.float64)
            delta_entropy = np.asarray([float(r["delta_entropy"]) for r in sorted_rows], dtype=np.float64)

            summary = {
                "run_id": run_dir.name,
                "mask_dir_name": mask_name,
                "grid": grid,
                "sample_id": sample_id,
                "bbox_area_fraction": maps["bbox_area_fraction"],
                "background_luminance": maps["background_luminance"],
                "bbox_note": bboxes[sample_id].get("bbox_note", ""),
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

            audit = {"sample_id": sample_id, **bboxes[sample_id]}
            audit["bbox_area_fraction"] = maps["bbox_area_fraction"]
            audit["background_luminance"] = maps["background_luminance"]
            audit_rows.append(audit)

            if args.save_bbox_masks:
                mask_dir = out_dir / "bbox_masks"
                mask_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray((maps["mask"] * 255).astype(np.uint8)).save(
                    mask_dir / f"{sample_id}_bbox_mask.png"
                )

    write_csv(sample_rows, out_dir / "summary_manual_bbox_debias.csv")
    write_csv(cell_rows, out_dir / "manual_bbox_debias_cell_metrics.csv")
    write_csv(audit_rows, out_dir / "manual_bbox_audit.csv")
    print(f"[INFO] wrote {len(sample_rows)} sample rows and {len(cell_rows)} cell rows to {out_dir}")


if __name__ == "__main__":
    main()
