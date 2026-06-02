#!/usr/bin/env python3
"""Debias UME against image and position features for selected mask runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


SIGNALS = (
    "ume",
    "ume_resid_luminance",
    "ume_resid_edge",
    "ume_resid_image",
    "ume_resid_position",
    "ume_resid_image_position",
    "luminance",
    "edge",
    "rgb_variance",
    "chroma",
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


def build_signals(rows):
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
    image_columns = [edge, rgb_variance, luminance, chroma]
    position_columns = [row_pos, col_pos, dist_center]
    return rows, grid, {
        "ume": ume,
        "ume_resid_luminance": residualize(ume, [luminance]),
        "ume_resid_edge": residualize(ume, [edge]),
        "ume_resid_image": residualize(ume, image_columns),
        "ume_resid_position": residualize(ume, position_columns),
        "ume_resid_image_position": residualize(ume, image_columns + position_columns),
        "luminance": luminance,
        "edge": edge,
        "rgb_variance": rgb_variance,
        "chroma": chroma,
    }


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    sample_rows = []
    cell_rows = []

    for spec in args.run_mask:
        run_text, mask_name = spec.split(":", 1)
        run_dir = Path(run_text)
        cell_path = run_dir / f"{mask_name}_image_baselines" / "image_baseline_cell_metrics.csv"
        rows = read_rows(cell_path)
        by_sample = {}
        for row in rows:
            by_sample.setdefault(row["sample_id"], []).append(row)
        for sample_id, group in sorted(by_sample.items()):
            sorted_rows, grid, signals = build_signals(group)
            delta_nll = np.asarray([float(r["delta_nll"]) for r in sorted_rows], dtype=np.float64)
            delta_entropy = np.asarray([float(r["delta_entropy"]) for r in sorted_rows], dtype=np.float64)
            summary = {
                "run_id": run_dir.name,
                "mask_dir_name": mask_name,
                "grid": grid,
                "sample_id": sample_id,
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

    write_csv(sample_rows, out_dir / "summary_debiased_ume.csv")
    write_csv(cell_rows, out_dir / "debiased_ume_cell_metrics.csv")
    print(f"[INFO] wrote {len(sample_rows)} sample rows and {len(cell_rows)} cell rows to {out_dir}")


if __name__ == "__main__":
    main()
