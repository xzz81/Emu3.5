#!/usr/bin/env python3
"""Compare repeated real-poster GT readback prompt conditions."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path
from statistics import mean


OUTSIDE_KNOWLEDGE = {
    "released",
    "release",
    "director",
    "directed",
    "starring",
    "installment",
    "sequel",
    "trilogy",
    "franchise",
    "original",
    "theatrical",
    "christopher",
    "nolan",
    "zemeckis",
    "spielberg",
    "leonardo",
    "dicaprio",
    "keanu",
    "reeves",
    "michael",
    "fox",
    "hamill",
    "ford",
    "fisher",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition",
        action="append",
        required=True,
        help="NAME=SUMMARY_DIR=RUN_DIR. SUMMARY_DIR must contain gt_readback_repeat_* CSVs.",
    )
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


def words(text: str) -> set[str]:
    text = re.sub(r"<\|[^>]+?\|>", " ", text.lower())
    return set(re.findall(r"[a-z0-9]+", text))


def outside_hits(run_dir: Path, sample_id: str) -> set[str]:
    path = run_dir / "raw_generations" / f"{sample_id}.txt"
    if not path.exists():
        return set()
    return words(path.read_text(encoding="utf-8")) & OUTSIDE_KNOWLEDGE


def parse_condition(text: str) -> tuple[str, Path, Path]:
    parts = text.split("=", 2)
    if len(parts) != 3:
        raise SystemExit(f"--condition must be NAME=SUMMARY_DIR=RUN_DIR, got {text!r}")
    return parts[0], Path(parts[1]), Path(parts[2])


def summarize_condition(name: str, summary_dir: Path, run_dir: Path) -> dict:
    sample_rows = read_csv(summary_dir / "gt_readback_repeat_sample_summary.csv")
    group_rows = read_csv(summary_dir / "gt_readback_repeat_group_summary.csv")
    if name == "describe":
        sample_rows = [row for row in sample_rows if row.get("input_form") == "image_plus_text"]
        group_rows = [row for row in group_rows if row.get("input_form") == "image_plus_text"]
    behaviors = Counter(row["behavior"] for row in sample_rows)
    outside = [outside_hits(run_dir, row["sample_id"]) for row in sample_rows]
    top_categories = Counter()
    for row in group_rows:
        for item in row.get("aggregate_top_categories", "").split(";"):
            if ":" not in item:
                continue
            key, value = item.rsplit(":", 1)
            top_categories[key] += int(float(value))
    return {
        "condition": name,
        "sample_count": len(sample_rows),
        "poster_group_count": len(group_rows),
        "behavior_counts": ";".join(f"{k}:{v}" for k, v in behaviors.most_common()),
        "mean_word_count": mean(float(row["word_count"]) for row in sample_rows),
        "mean_trace_tokens": mean(float(row["trace_token_count"]) for row in sample_rows),
        "mean_ume": mean(float(row["mean_ume"]) for row in sample_rows),
        "mean_group_sd_ume": mean(float(row["sd_ume"]) for row in group_rows),
        "mean_content_jaccard": mean(float(row["pairwise_content_jaccard"]) for row in group_rows),
        "mean_top_category_jaccard": mean(float(row["top_category_jaccard"]) for row in group_rows),
        "outside_knowledge_hit_rate": sum(bool(x) for x in outside) / len(outside) if outside else 0.0,
        "mean_outside_knowledge_hits": mean(len(x) for x in outside) if outside else 0.0,
        "outside_knowledge_words": ";".join(k for k, _ in Counter(w for xs in outside for w in xs).most_common(20)),
        "aggregate_top_categories": ";".join(f"{k}:{v}" for k, v in top_categories.most_common()),
    }


def write_report(path: Path, rows: list[dict]) -> None:
    lines = [
        "# GT Prompt Repeat Condition Comparison",
        "",
        "This report compares repeated real-poster readback prompt conditions.",
        "",
    ]
    for row in rows:
        lines.append(f"## {row['condition']}")
        lines.append(f"- samples: {row['sample_count']}")
        lines.append(f"- behavior counts: {row['behavior_counts']}")
        lines.append(f"- mean words: {float(row['mean_word_count']):.2f}")
        lines.append(f"- mean UME: {float(row['mean_ume']):.4f}")
        lines.append(f"- mean group sd UME: {float(row['mean_group_sd_ume']):.4f}")
        lines.append(f"- mean content Jaccard: {float(row['mean_content_jaccard']):.3f}")
        lines.append(f"- mean top-category Jaccard: {float(row['mean_top_category_jaccard']):.3f}")
        lines.append(f"- outside-hit rate: {float(row['outside_knowledge_hit_rate']):.3f}")
        lines.append(f"- mean outside hits: {float(row['mean_outside_knowledge_hits']):.3f}")
        lines.append(f"- outside words: {row['outside_knowledge_words']}")
        lines.append(f"- aggregate top categories: {row['aggregate_top_categories']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    rows = [
        summarize_condition(name, summary_dir, run_dir)
        for name, summary_dir, run_dir in (parse_condition(item) for item in args.condition)
    ]
    write_csv(out_dir / "gt_prompt_repeat_condition_summary.csv", rows)
    write_report(out_dir / "gt_prompt_repeat_condition_comparison.md", rows)
    print(f"[INFO] wrote {len(rows)} condition rows to {out_dir}")


if __name__ == "__main__":
    main()
