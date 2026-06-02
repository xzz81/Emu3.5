#!/usr/bin/env python3
"""Analyze counterfactual T2I prompt pairs against visual-token entropy maps."""

from __future__ import annotations

import argparse
import csv
import importlib as imp
import json
import math
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--score-field", default="ume_full", choices=["ume", "ume_full"])
    parser.add_argument("--top-frac", default=0.2, type=float)
    parser.add_argument("--shuffle-count", default=1024, type=int)
    parser.add_argument("--shuffle-seed", default=24000, type=int)
    return parser.parse_args()


def load_cfg(path: str):
    cfg_name = Path(path).stem
    cfg_package = Path(path).parent.__str__().replace("/", ".")
    return imp.import_module(f".{cfg_name}", package=cfg_package)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fnum(value: object, default: float = 0.0) -> float:
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_visual_map(trace_path: Path, score_field: str) -> np.ndarray:
    rows = [r for r in read_jsonl(trace_path) if r.get("token_type") == "visual"]
    scores = np.array([fnum(r.get(score_field, r.get("ume"))) for r in rows], dtype=np.float32)
    if scores.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    width = int(round(math.sqrt(scores.size)))
    if width * width != scores.size:
        raise ValueError(f"{trace_path} has non-square visual token count {scores.size}")
    return scores.reshape(width, width)


def load_image(path: Path, size: tuple[int, int]) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize(size, Image.Resampling.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def rank_top_mask(values: np.ndarray, frac: float) -> np.ndarray:
    flat = values.reshape(-1)
    if flat.size == 0:
        return np.zeros_like(values, dtype=bool)
    k = max(1, int(math.ceil(flat.size * frac)))
    threshold = np.partition(flat, flat.size - k)[flat.size - k]
    return values >= threshold


def stable_seed(*parts: object) -> int:
    text = "::".join(str(part) for part in parts)
    return zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF


def random_overlap_stats(
    target_mask: np.ndarray,
    candidate_count: int,
    observed_overlap: float,
    shuffle_count: int,
    seed: int,
) -> dict:
    target_flat = target_mask.reshape(-1)
    n = int(target_flat.size)
    target_count = int(target_flat.sum())
    candidate_count = max(0, min(int(candidate_count), n))
    if n == 0 or target_count == 0 or candidate_count == 0 or shuffle_count <= 0:
        return {
            "shuffle_mean": 0.0,
            "shuffle_p95": 0.0,
            "shuffle_p_value": 1.0,
            "over_shuffle_p95": 0,
        }
    rng = np.random.default_rng(seed)
    values = np.empty(shuffle_count, dtype=np.float32)
    for idx in range(shuffle_count):
        picks = rng.choice(n, size=candidate_count, replace=False)
        values[idx] = float(target_flat[picks].sum()) / target_count
    p_value = (float(np.sum(values >= observed_overlap)) + 1.0) / (shuffle_count + 1.0)
    p95 = float(np.quantile(values, 0.95))
    return {
        "shuffle_mean": float(values.mean()),
        "shuffle_p95": p95,
        "shuffle_p_value": p_value,
        "over_shuffle_p95": int(observed_overlap > p95),
    }


def add_shuffle_columns(row: dict, prefix: str, stats: dict) -> None:
    row[f"{prefix}_shuffle_mean"] = stats["shuffle_mean"]
    row[f"{prefix}_shuffle_p95"] = stats["shuffle_p95"]
    row[f"{prefix}_minus_shuffle_mean"] = row[prefix] - stats["shuffle_mean"]
    row[f"{prefix}_minus_shuffle_p95"] = row[prefix] - stats["shuffle_p95"]
    row[f"{prefix}_shuffle_p_value"] = stats["shuffle_p_value"]
    row[f"{prefix}_over_shuffle_p95"] = stats["over_shuffle_p95"]


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    x = a.reshape(-1).astype(np.float64)
    y = b.reshape(-1).astype(np.float64)
    if x.size == 0 or y.size == 0 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def rankdata(values: np.ndarray) -> np.ndarray:
    flat = values.reshape(-1)
    order = np.argsort(flat, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(flat.size, dtype=np.float64)
    return ranks.reshape(values.shape)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return pearson(rankdata(a), rankdata(b))


def edge_map(image: np.ndarray) -> np.ndarray:
    gray = image.mean(axis=2)
    gy = np.zeros_like(gray)
    gx = np.zeros_like(gray)
    gy[1:, :] = np.abs(gray[1:, :] - gray[:-1, :])
    gx[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1])
    return gx + gy


def quadrant_counts(mask: np.ndarray) -> str:
    h, w = mask.shape
    parts = {
        "top_left": mask[: h // 2, : w // 2],
        "top_right": mask[: h // 2, w // 2 :],
        "bottom_left": mask[h // 2 :, : w // 2],
        "bottom_right": mask[h // 2 :, w // 2 :],
    }
    counts = Counter({key: int(value.sum()) for key, value in parts.items()})
    return ";".join(f"{key}:{value}" for key, value in counts.items())


def mask_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum()) / float(union)


def sample_row(run_dir: Path, sample_id: str, meta: dict, score_field: str) -> dict:
    trace_path = run_dir / "entropy_traces" / f"{sample_id}_entropy.jsonl"
    img_path = run_dir / "decoded" / f"{sample_id}_image_00.png"
    entropy = load_visual_map(trace_path, score_field)
    image_std = 0.0
    edge_mean = 0.0
    if img_path.exists():
        size = (int(entropy.shape[1]), int(entropy.shape[0])) if entropy.size else (16, 16)
        image = load_image(img_path, size)
        image_std = float(image.std())
        edge_mean = float(edge_map(image).mean())
    return {
        "sample_id": sample_id,
        "pair_id": meta["pair_id"],
        "factor": meta["factor"],
        "variant": meta["variant"],
        "counterfactual_value": meta["counterfactual_value"],
        "decoded_image_exists": int(img_path.exists()),
        "visual_token_count": int(entropy.size),
        "grid_width": int(entropy.shape[1]) if entropy.size else 0,
        "mean_entropy": float(entropy.mean()) if entropy.size else 0.0,
        "max_entropy": float(entropy.max()) if entropy.size else 0.0,
        "image_rgb_std": image_std,
        "image_edge_mean": edge_mean,
        "low_contrast_flag": int(image_std < 0.08 and edge_mean < 0.04),
    }


def analyze_pair(
    run_dir: Path,
    pair_id: str,
    items: list[tuple[str, dict]],
    score_field: str,
    top_frac: float,
    shuffle_count: int,
    shuffle_seed: int,
) -> dict:
    by_variant = {meta["variant"]: (sample_id, meta) for sample_id, meta in items}
    if "a" not in by_variant or "b" not in by_variant:
        raise ValueError(f"incomplete pair {pair_id}")
    sample_a, meta_a = by_variant["a"]
    sample_b, meta_b = by_variant["b"]
    trace_dir = run_dir / "entropy_traces"
    decoded_dir = run_dir / "decoded"
    ent_a = load_visual_map(trace_dir / f"{sample_a}_entropy.jsonl", score_field)
    ent_b = load_visual_map(trace_dir / f"{sample_b}_entropy.jsonl", score_field)
    if ent_a.shape != ent_b.shape or ent_a.size == 0:
        raise ValueError(f"bad entropy map shapes for {pair_id}: {ent_a.shape}, {ent_b.shape}")
    h, w = ent_a.shape
    img_a = load_image(decoded_dir / f"{sample_a}_image_00.png", (w, h))
    img_b = load_image(decoded_dir / f"{sample_b}_image_00.png", (w, h))
    image_a_std = float(img_a.std())
    image_b_std = float(img_b.std())
    image_a_edge = float(edge_map(img_a).mean())
    image_b_edge = float(edge_map(img_b).mean())
    low_contrast_a = int(image_a_std < 0.08 and image_a_edge < 0.04)
    low_contrast_b = int(image_b_std < 0.08 and image_b_edge < 0.04)
    image_delta = np.abs(img_a - img_b).mean(axis=2)
    entropy_mean = (ent_a + ent_b) / 2.0
    entropy_delta = np.abs(ent_a - ent_b)
    edge_a = edge_map(img_a)
    edge_b = edge_map(img_b)
    edge_mean = (edge_a + edge_b) / 2.0
    edge_delta = np.abs(edge_a - edge_b)

    image_top = rank_top_mask(image_delta, top_frac)
    entropy_top = rank_top_mask(entropy_mean, top_frac)
    delta_top = rank_top_mask(entropy_delta, top_frac)
    edge_top = rank_top_mask(edge_mean, top_frac)
    edge_a_top = rank_top_mask(edge_a, top_frac)
    edge_b_top = rank_top_mask(edge_b, top_frac)

    image_top_count = int(image_top.sum())
    overlap_entropy = int(np.logical_and(image_top, entropy_top).sum())
    overlap_delta = int(np.logical_and(image_top, delta_top).sum())
    overlap_edge = int(np.logical_and(image_top, edge_top).sum())
    outside = ~image_top
    row = {
        "pair_id": pair_id,
        "factor": meta_a["factor"],
        "value_a": meta_a["counterfactual_value"],
        "value_b": meta_b["counterfactual_value"],
        "grid_width": w,
        "mean_image_delta": float(image_delta.mean()),
        "mean_edge_delta": float(edge_delta.mean()),
        "edge_delta_to_image_delta_ratio": float(edge_delta.mean() / image_delta.mean()) if image_delta.mean() > 0 else 0.0,
        "edge_top20_jaccard": mask_jaccard(edge_a_top, edge_b_top),
        "mean_entropy": float(entropy_mean.mean()),
        "mean_delta_entropy": float(entropy_delta.mean()),
        "image_a_std": image_a_std,
        "image_b_std": image_b_std,
        "image_a_edge_mean": image_a_edge,
        "image_b_edge_mean": image_b_edge,
        "low_contrast_a": low_contrast_a,
        "low_contrast_b": low_contrast_b,
        "clean_pair": int(not low_contrast_a and not low_contrast_b),
        "image_top20_entropy_overlap": overlap_entropy / image_top_count if image_top_count else 0.0,
        "image_top20_delta_entropy_overlap": overlap_delta / image_top_count if image_top_count else 0.0,
        "image_top20_edge_overlap": overlap_edge / image_top_count if image_top_count else 0.0,
        "pearson_entropy_image_delta": pearson(entropy_mean, image_delta),
        "spearman_entropy_image_delta": spearman(entropy_mean, image_delta),
        "pearson_delta_entropy_image_delta": pearson(entropy_delta, image_delta),
        "spearman_delta_entropy_image_delta": spearman(entropy_delta, image_delta),
        "pearson_edge_image_delta": pearson(edge_mean, image_delta),
        "mean_entropy_in_image_top20": float(entropy_mean[image_top].mean()) if image_top.any() else 0.0,
        "mean_entropy_outside_image_top20": float(entropy_mean[outside].mean()) if outside.any() else 0.0,
        "mean_delta_entropy_in_image_top20": float(entropy_delta[image_top].mean()) if image_top.any() else 0.0,
        "mean_delta_entropy_outside_image_top20": float(entropy_delta[outside].mean()) if outside.any() else 0.0,
        "image_delta_top20_quadrants": quadrant_counts(image_top),
        "entropy_top20_quadrants": quadrant_counts(entropy_top),
        "delta_entropy_top20_quadrants": quadrant_counts(delta_top),
    }
    for prefix, mask in [
        ("image_top20_entropy_overlap", entropy_top),
        ("image_top20_delta_entropy_overlap", delta_top),
        ("image_top20_edge_overlap", edge_top),
    ]:
        stats = random_overlap_stats(
            image_top,
            int(mask.sum()),
            float(row[prefix]),
            shuffle_count,
            stable_seed(shuffle_seed, pair_id, prefix),
        )
        add_shuffle_columns(row, prefix, stats)
    return row


def aggregate(rows: list[dict], key: str) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    out = []
    for value, group in sorted(grouped.items()):
        out.append(
            {
                key: value,
                "pair_count": len(group),
                "clean_pair_count": int(sum(int(r.get("clean_pair", 0)) for r in group)),
                "mean_image_top20_entropy_overlap": float(np.mean([r["image_top20_entropy_overlap"] for r in group])),
                "mean_image_top20_delta_entropy_overlap": float(np.mean([r["image_top20_delta_entropy_overlap"] for r in group])),
                "mean_image_top20_edge_overlap": float(np.mean([r["image_top20_edge_overlap"] for r in group])),
                "mean_edge_delta_to_image_delta_ratio": float(
                    np.mean([r["edge_delta_to_image_delta_ratio"] for r in group])
                ),
                "mean_edge_top20_jaccard": float(np.mean([r["edge_top20_jaccard"] for r in group])),
                "mean_image_top20_entropy_shuffle_p95": float(
                    np.mean([r["image_top20_entropy_overlap_shuffle_p95"] for r in group])
                ),
                "mean_image_top20_delta_entropy_shuffle_p95": float(
                    np.mean([r["image_top20_delta_entropy_overlap_shuffle_p95"] for r in group])
                ),
                "mean_image_top20_edge_shuffle_p95": float(
                    np.mean([r["image_top20_edge_overlap_shuffle_p95"] for r in group])
                ),
                "entropy_overlap_over_shuffle_p95_count": int(
                    sum(int(r["image_top20_entropy_overlap_over_shuffle_p95"]) for r in group)
                ),
                "delta_entropy_overlap_over_shuffle_p95_count": int(
                    sum(int(r["image_top20_delta_entropy_overlap_over_shuffle_p95"]) for r in group)
                ),
                "edge_overlap_over_shuffle_p95_count": int(
                    sum(int(r["image_top20_edge_overlap_over_shuffle_p95"]) for r in group)
                ),
                "mean_entropy_overlap_shuffle_p_value": float(
                    np.mean([r["image_top20_entropy_overlap_shuffle_p_value"] for r in group])
                ),
                "mean_delta_entropy_overlap_shuffle_p_value": float(
                    np.mean([r["image_top20_delta_entropy_overlap_shuffle_p_value"] for r in group])
                ),
                "mean_edge_overlap_shuffle_p_value": float(
                    np.mean([r["image_top20_edge_overlap_shuffle_p_value"] for r in group])
                ),
                "mean_pearson_entropy_image_delta": float(np.mean([r["pearson_entropy_image_delta"] for r in group])),
                "mean_pearson_delta_entropy_image_delta": float(np.mean([r["pearson_delta_entropy_image_delta"] for r in group])),
                "mean_pearson_edge_image_delta": float(np.mean([r["pearson_edge_image_delta"] for r in group])),
                "mean_entropy_in_minus_out_image_top20": float(
                    np.mean([r["mean_entropy_in_image_top20"] - r["mean_entropy_outside_image_top20"] for r in group])
                ),
                "mean_delta_entropy_in_minus_out_image_top20": float(
                    np.mean([
                        r["mean_delta_entropy_in_image_top20"] - r["mean_delta_entropy_outside_image_top20"]
                        for r in group
                    ])
                ),
            }
        )
    return out


def clean_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if int(row.get("clean_pair", 0)) == 1]


def write_report(path: Path, pair_rows: list[dict], factor_rows: list[dict]) -> None:
    clean_factor_rows = aggregate(clean_rows(pair_rows), "factor") if clean_rows(pair_rows) else []
    lines = [
        "# Counterfactual T2I Pair Report",
        "",
        "This report compares prompt-pair image-change regions against visual-token entropy maps.",
        "",
        "## Factor Summary",
        "",
    ]
    for row in factor_rows:
        lines.append(
            f"- {row['factor']}: pairs={row['pair_count']}, "
            f"clean={row['clean_pair_count']}, "
            f"image-top20/entropy-top20 overlap={row['mean_image_top20_entropy_overlap']:.3f}, "
            f"image-top20/delta-entropy-top20 overlap={row['mean_image_top20_delta_entropy_overlap']:.3f}, "
            f"edge overlap={row['mean_image_top20_edge_overlap']:.3f}, "
            f"edge-delta/image-delta={row['mean_edge_delta_to_image_delta_ratio']:.3f}, "
            f"edge-top20 Jaccard={row['mean_edge_top20_jaccard']:.3f}, "
            f"over shuffle-p95 entropy/delta/edge="
            f"{row['entropy_overlap_over_shuffle_p95_count']}/"
            f"{row['delta_entropy_overlap_over_shuffle_p95_count']}/"
            f"{row['edge_overlap_over_shuffle_p95_count']}, "
            f"Pearson entropy={row['mean_pearson_entropy_image_delta']:.3f}, "
            f"Pearson delta-entropy={row['mean_pearson_delta_entropy_image_delta']:.3f}, "
            f"Pearson edge={row['mean_pearson_edge_image_delta']:.3f}"
        )
    if clean_factor_rows:
        lines.extend(["", "## Clean-Pair Factor Summary", ""])
        for row in clean_factor_rows:
            lines.append(
                f"- {row['factor']}: clean pairs={row['pair_count']}, "
                f"image-top20/entropy-top20 overlap={row['mean_image_top20_entropy_overlap']:.3f}, "
                f"image-top20/delta-entropy-top20 overlap={row['mean_image_top20_delta_entropy_overlap']:.3f}, "
                f"edge overlap={row['mean_image_top20_edge_overlap']:.3f}, "
                f"edge-delta/image-delta={row['mean_edge_delta_to_image_delta_ratio']:.3f}, "
                f"edge-top20 Jaccard={row['mean_edge_top20_jaccard']:.3f}, "
                f"over shuffle-p95 entropy/delta/edge="
                f"{row['entropy_overlap_over_shuffle_p95_count']}/"
                f"{row['delta_entropy_overlap_over_shuffle_p95_count']}/"
                f"{row['edge_overlap_over_shuffle_p95_count']}, "
                f"Pearson entropy={row['mean_pearson_entropy_image_delta']:.3f}, "
                f"Pearson delta-entropy={row['mean_pearson_delta_entropy_image_delta']:.3f}, "
                f"Pearson edge={row['mean_pearson_edge_image_delta']:.3f}"
            )
    lines.extend(["", "## Pair Details", ""])
    for row in pair_rows:
        lines.append(
            f"- {row['pair_id']} ({row['factor']}: {row['value_a']} -> {row['value_b']}): "
            f"clean={row['clean_pair']}, "
            f"overlap entropy={row['image_top20_entropy_overlap']:.3f}, "
            f"entropy shuffle-p95={row['image_top20_entropy_overlap_shuffle_p95']:.3f}, "
            f"p={row['image_top20_entropy_overlap_shuffle_p_value']:.3f}, "
            f"delta-entropy={row['image_top20_delta_entropy_overlap']:.3f}, "
            f"delta shuffle-p95={row['image_top20_delta_entropy_overlap_shuffle_p95']:.3f}, "
            f"p={row['image_top20_delta_entropy_overlap_shuffle_p_value']:.3f}, "
            f"edge={row['image_top20_edge_overlap']:.3f}, "
            f"edge shuffle-p95={row['image_top20_edge_overlap_shuffle_p95']:.3f}, "
            f"p={row['image_top20_edge_overlap_shuffle_p_value']:.3f}, "
            f"edge-delta/image-delta={row['edge_delta_to_image_delta_ratio']:.3f}, "
            f"edge-top20 Jaccard={row['edge_top20_jaccard']:.3f}, "
            f"Pearson entropy={row['pearson_entropy_image_delta']:.3f}, "
            f"delta-entropy={row['pearson_delta_entropy_image_delta']:.3f}, "
            f"edge={row['pearson_edge_image_delta']:.3f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_cfg(args.cfg)
    metadata = getattr(cfg, "pair_metadata")
    grouped: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    sample_rows = []
    for sample_id, meta in sorted(metadata.items()):
        trace_path = run_dir / "entropy_traces" / f"{sample_id}_entropy.jsonl"
        if not trace_path.exists():
            continue
        grouped[meta["pair_id"]].append((sample_id, meta))
        sample_rows.append(sample_row(run_dir, sample_id, meta, args.score_field))
    pair_rows = [
        analyze_pair(
            run_dir,
            pair_id,
            items,
            args.score_field,
            args.top_frac,
            args.shuffle_count,
            args.shuffle_seed,
        )
        for pair_id, items in sorted(grouped.items())
        if len(items) == 2
    ]
    factor_rows = aggregate(pair_rows, "factor")
    clean_pair_rows = clean_rows(pair_rows)
    clean_factor_rows = aggregate(clean_pair_rows, "factor")
    write_csv(out_dir / "counterfactual_sample_summary.csv", sample_rows)
    write_csv(out_dir / "counterfactual_pair_summary.csv", pair_rows)
    write_csv(out_dir / "counterfactual_factor_summary.csv", factor_rows)
    write_csv(out_dir / "counterfactual_clean_pair_summary.csv", clean_pair_rows)
    write_csv(out_dir / "counterfactual_clean_factor_summary.csv", clean_factor_rows)
    write_report(out_dir / "counterfactual_pair_report.md", pair_rows, factor_rows)
    print(f"[INFO] wrote counterfactual analysis for {len(pair_rows)} pairs to {out_dir}")


if __name__ == "__main__":
    main()
