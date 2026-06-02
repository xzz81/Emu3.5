#!/usr/bin/env python3
"""Summarize manual span entropy labels by prompt/input condition."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fnum(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def infer_condition(sample_id: str) -> str:
    if "__seed72_strict_visible_no_guess" in sample_id:
        return "strict_visible_no_guess"
    if "__seed72_visible_only" in sample_id:
        return "visible_only"
    if "__seed72_describe" in sample_id:
        return "describe"
    if "__greedy_visible_only" in sample_id:
        return "greedy_visible_only"
    if "__greedy_describe" in sample_id:
        return "greedy_describe"
    if "__visible_only" in sample_id:
        return "visible_only"
    if "__gt_describe" in sample_id:
        return "describe"
    if "__gt_imageonly" in sample_id:
        return "pure_image"
    if "__text_band_masked" in sample_id:
        return "text_band_masked"
    return "unknown"


def counts_to_text(counts: Counter) -> str:
    return ";".join(f"{k}:{v}" for k, v in counts.most_common())


def summarize_group(condition: str, truth_label: str, rows: list[dict]) -> dict:
    n = len(rows)
    mean_ume = sum(fnum(r["mean_ume"]) for r in rows) / n if n else 0.0
    mean_delta = sum(fnum(r["mean_ume_minus_baseline"]) for r in rows) / n if n else 0.0
    exact_span_rate = sum(1 for r in rows if fnum(r["exact_top20_fraction"]) > 0) / n if n else 0.0
    near_span_rate = sum(1 for r in rows if fnum(r["near_top20_fraction"]) > 0) / n if n else 0.0
    exact_token_fraction = sum(fnum(r["exact_top20_fraction"]) for r in rows) / n if n else 0.0
    near_token_fraction = sum(fnum(r["near_top20_fraction"]) for r in rows) / n if n else 0.0
    aspects = Counter(r["aspect"] for r in rows)
    examples = " | ".join(r["snippet"] for r in rows[:5])
    return {
        "condition": condition,
        "truth_label": truth_label,
        "span_count": n,
        "mean_ume": fmt(mean_ume),
        "mean_delta_vs_same_length": fmt(mean_delta),
        "span_has_exact_top20_rate": fmt(exact_span_rate),
        "span_has_near_top20_rate": fmt(near_span_rate),
        "mean_exact_top20_token_fraction": fmt(exact_token_fraction),
        "mean_near_top20_token_fraction": fmt(near_token_fraction),
        "aspect_counts": counts_to_text(aspects),
        "example_snippets": examples,
    }


def write_report(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Manual Span Entropy By Condition",
        "",
        "Rows are grouped by prompt/input condition and truth label.",
        "",
    ]
    for row in rows:
        lines.append(
            f"- {row['condition']} / {row['truth_label']}: n={row['span_count']}, "
            f"mean UME={row['mean_ume']}, delta={row['mean_delta_vs_same_length']}, "
            f"has_exact={row['span_has_exact_top20_rate']}, has_near={row['span_has_near_top20_rate']}, "
            f"aspects={row['aspect_counts']}"
        )
        if row["example_snippets"]:
            lines.append(f"  examples: {row['example_snippets']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = read_csv(Path(args.rows_csv))
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(infer_condition(row["sample_id"]), row["truth_label"])].append(row)
    out = [
        summarize_group(condition, truth_label, group)
        for (condition, truth_label), group in sorted(grouped.items())
    ]
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "manual_span_entropy_by_condition.csv", out)
    write_report(out_dir / "manual_span_entropy_by_condition.md", out)
    print(f"[INFO] wrote {len(out)} condition/truth rows to {out_dir}")


if __name__ == "__main__":
    main()
