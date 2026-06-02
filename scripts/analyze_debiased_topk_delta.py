#!/usr/bin/env python3
"""Evaluate edge/image/position-debiased top-k visual distribution features."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


BASE_SIGNALS = (
    "topk_embedding_spread",
    "topk_entropy",
    "topk_prob_mass",
    "top1_prob",
    "sampled_code_topk_prob",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/emu3p5-main")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-id-contains", default="topk64_batch2")
    parser.add_argument("--mask-dir-name", default="mask_delta_analysis_topk64")
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
    top_vals = y[list(top_x)]
    rest_vals = y[rest_idx]
    return {
        "pearson": corr(x, y),
        "spearman": corr(rankdata(x), rankdata(y)),
        "top20_overlap": len(top_x & top_y) / k,
        "top20_minus_rest": float(top_vals.mean() - np.asarray(rest_vals).mean()),
    }


def load_rows(path: Path):
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def key(row):
    return row["sample_id"], int(row["row"]), int(row["col"])


def find_topk_runs(root: Path, run_id_contains: str, mask_dir_name: str):
    for path in sorted(root.glob(f"*/ume_trace_runs/*/{mask_dir_name}_topk_code_spread/topk_code_spread_cell_metrics.csv")):
        run_dir = path.parents[1]
        if run_id_contains and run_id_contains not in run_dir.name:
            continue
        image_path = run_dir / f"{mask_dir_name}_image_baselines" / "image_baseline_cell_metrics.csv"
        if not image_path.exists():
            continue
        task = run_dir.parent.parent.name
        run_id = run_dir.name
        yield task, run_id, path, image_path


def merged_rows(topk_path: Path, image_path: Path):
    image_by_key = {key(row): row for row in load_rows(image_path)}
    rows = []
    for row in load_rows(topk_path):
        merged = dict(row)
        image_row = image_by_key[key(row)]
        for field in ("edge", "rgb_variance", "luminance", "chroma"):
            merged[field] = image_row[field]
        rows.append(merged)
    return rows


def sample_groups(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["sample_id"], []).append(row)
    return groups


def build_signals(rows):
    rows = sorted(rows, key=lambda r: (int(r["row"]), int(r["col"])))
    grid = max(max(int(r["row"]), int(r["col"])) for r in rows) + 1
    row_pos = np.asarray([int(r["row"]) for r in rows], dtype=np.float64)
    col_pos = np.asarray([int(r["col"]) for r in rows], dtype=np.float64)
    center = (grid - 1) / 2.0
    dist_center = np.sqrt((row_pos - center) ** 2 + (col_pos - center) ** 2)

    edge = np.asarray([float(r["edge"]) for r in rows], dtype=np.float64)
    rgb_variance = np.asarray([float(r["rgb_variance"]) for r in rows], dtype=np.float64)
    luminance = np.asarray([float(r["luminance"]) for r in rows], dtype=np.float64)
    chroma = np.asarray([float(r["chroma"]) for r in rows], dtype=np.float64)
    image_columns = [edge, rgb_variance, luminance, chroma]
    position_columns = [row_pos, col_pos, dist_center]

    signals = {
        "ume": np.asarray([float(r["coarse_ume"]) for r in rows], dtype=np.float64),
        "edge": edge,
    }
    for name in BASE_SIGNALS:
        raw = np.asarray([float(r[name]) for r in rows], dtype=np.float64)
        signals[name] = raw
        signals[f"{name}_low"] = -raw
        signals[f"{name}_resid_edge"] = residualize(raw, [edge])
        signals[f"{name}_resid_image"] = residualize(raw, image_columns)
        signals[f"{name}_resid_position"] = residualize(raw, position_columns)
        signals[f"{name}_resid_image_position"] = residualize(raw, image_columns + position_columns)

    delta_nll = np.asarray([float(r["delta_nll"]) for r in rows], dtype=np.float64)
    delta_entropy = np.asarray([float(r["delta_entropy"]) for r in rows], dtype=np.float64)
    return grid, signals, delta_nll, delta_entropy, rows


def main():
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    sample_rows = []
    cell_rows = []

    for task, run_id, topk_path, image_path in find_topk_runs(root, args.run_id_contains, args.mask_dir_name):
        for sample_id, group in sample_groups(merged_rows(topk_path, image_path)).items():
            grid, signals, delta_nll, delta_entropy, sorted_rows = build_signals(group)
            summary = {
                "task": task,
                "run_id": run_id,
                "grid": grid,
                "sample_id": sample_id,
            }
            for signal_name, signal in signals.items():
                for delta_name, delta in (("delta_nll", delta_nll), ("delta_entropy", delta_entropy)):
                    stats = compare_signal(signal, delta)
                    prefix = f"{signal_name}_{delta_name}"
                    for key_name, value in stats.items():
                        summary[f"{prefix}_{key_name}"] = value
            sample_rows.append(summary)

            for idx, row in enumerate(sorted_rows):
                out = {
                    "task": task,
                    "run_id": run_id,
                    "grid": grid,
                    "sample_id": sample_id,
                    "row": row["row"],
                    "col": row["col"],
                    "delta_nll": row["delta_nll"],
                    "delta_entropy": row["delta_entropy"],
                }
                for signal_name, signal in signals.items():
                    out[signal_name] = float(signal[idx])
                cell_rows.append(out)

    write_csv(sample_rows, out_dir / "summary_debiased_topk.csv")
    write_csv(cell_rows, out_dir / "debiased_topk_cell_metrics.csv")
    print(f"[INFO] wrote {len(sample_rows)} sample rows and {len(cell_rows)} cell rows to {out_dir}")


if __name__ == "__main__":
    main()
