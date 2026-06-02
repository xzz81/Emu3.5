#!/usr/bin/env python3
"""Compare repeated describe and visible-only GT readback runs."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
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
    parser.add_argument("--describe-summary-dir", required=True)
    parser.add_argument("--visible-summary-dir", required=True)
    parser.add_argument("--describe-run-dir", required=True)
    parser.add_argument("--visible-run-dir", required=True)
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


def form_metrics(sample_rows: list[dict], group_rows: list[dict], run_dir: Path, form_name: str) -> dict:
    behaviors = Counter(row["behavior"] for row in sample_rows)
    outside = [outside_hits(run_dir, row["sample_id"]) for row in sample_rows]
    return {
        "condition": form_name,
        "sample_count": len(sample_rows),
        "behavior_counts": ";".join(f"{k}:{v}" for k, v in behaviors.most_common()),
        "mean_word_count": mean(float(row["word_count"]) for row in sample_rows),
        "mean_trace_tokens": mean(float(row["trace_token_count"]) for row in sample_rows),
        "mean_ume": mean(float(row["mean_ume"]) for row in sample_rows),
        "mean_group_sd_ume": mean(float(row["sd_ume"]) for row in group_rows),
        "mean_content_jaccard": mean(float(row["pairwise_content_jaccard"]) for row in group_rows),
        "mean_top_category_jaccard": mean(float(row["top_category_jaccard"]) for row in group_rows),
        "outside_knowledge_hit_rate": sum(bool(x) for x in outside) / len(outside),
        "mean_outside_knowledge_hits": mean(len(x) for x in outside),
        "outside_knowledge_words": ";".join(k for k, _ in Counter(w for xs in outside for w in xs).most_common(20)),
    }


def write_report(path: Path, form_rows: list[dict], poster_rows: list[dict], notable: list[dict]) -> None:
    lines = [
        "# Describe vs Visible-Only Repeat Comparison",
        "",
        "This report compares seed54 `Describe this image carefully.` repeats with seed56 `visible_only` repeats.",
        "",
        "## Condition Summary",
        "",
    ]
    for row in form_rows:
        lines.append(f"### {row['condition']}")
        lines.append(f"- samples: {row['sample_count']}")
        lines.append(f"- behavior counts: {row['behavior_counts']}")
        lines.append(f"- mean words: {float(row['mean_word_count']):.2f}")
        lines.append(f"- mean trace tokens: {float(row['mean_trace_tokens']):.2f}")
        lines.append(f"- mean UME: {float(row['mean_ume']):.4f}")
        lines.append(f"- mean group sd UME: {float(row['mean_group_sd_ume']):.4f}")
        lines.append(f"- mean content Jaccard: {float(row['mean_content_jaccard']):.3f}")
        lines.append(f"- mean top-category Jaccard: {float(row['mean_top_category_jaccard']):.3f}")
        lines.append(f"- outside-knowledge hit rate: {float(row['outside_knowledge_hit_rate']):.3f}")
        lines.append(f"- outside-knowledge words: {row['outside_knowledge_words']}")
        lines.append("")

    lines.extend(["## Poster-Level Delta", ""])
    for row in poster_rows:
        lines.append(
            f"- poster {row['poster_id']}: visible_minus_describe_ume={float(row['visible_minus_describe_ume']):+.4f}, "
            f"visible_minus_describe_topcat_jaccard={float(row['visible_minus_describe_top_category_jaccard']):+.3f}, "
            f"visible_words={float(row['visible_mean_word_count']):.1f}, describe_words={float(row['describe_mean_word_count']):.1f}"
        )

    lines.extend(["", "## Notable Visible-Only Outputs", ""])
    for row in notable:
        lines.append(
            f"- `{row['sample_id']}` words={row['word_count']} outside={row['outside_knowledge_words']}: {row['raw_text_preview']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    describe_summary = Path(args.describe_summary_dir)
    visible_summary = Path(args.visible_summary_dir)
    describe_run = Path(args.describe_run_dir)
    visible_run = Path(args.visible_run_dir)
    out_dir = Path(args.out_dir)

    describe_samples = [
        row for row in read_csv(describe_summary / "gt_readback_repeat_sample_summary.csv")
        if row["input_form"] == "image_plus_text"
    ]
    describe_groups = [
        row for row in read_csv(describe_summary / "gt_readback_repeat_group_summary.csv")
        if row["input_form"] == "image_plus_text"
    ]
    visible_samples = read_csv(visible_summary / "gt_readback_repeat_sample_summary.csv")
    visible_groups = read_csv(visible_summary / "gt_readback_repeat_group_summary.csv")

    form_rows = [
        form_metrics(describe_samples, describe_groups, describe_run, "describe"),
        form_metrics(visible_samples, visible_groups, visible_run, "visible_only"),
    ]

    describe_by_poster = {row["poster_id"]: row for row in describe_groups}
    poster_rows = []
    for row in visible_groups:
        desc = describe_by_poster[row["poster_id"]]
        poster_rows.append(
            {
                "poster_id": row["poster_id"],
                "describe_mean_ume": desc["mean_ume"],
                "visible_mean_ume": row["mean_ume"],
                "visible_minus_describe_ume": float(row["mean_ume"]) - float(desc["mean_ume"]),
                "describe_top_category_jaccard": desc["top_category_jaccard"],
                "visible_top_category_jaccard": row["top_category_jaccard"],
                "visible_minus_describe_top_category_jaccard": float(row["top_category_jaccard"]) - float(desc["top_category_jaccard"]),
                "describe_mean_word_count": desc["mean_word_count"],
                "visible_mean_word_count": row["mean_word_count"],
            }
        )

    notable = []
    for row in visible_samples:
        hits = outside_hits(visible_run, row["sample_id"])
        if hits or row["behavior"] != "description" or int(float(row["word_count"])) < 60:
            notable.append(
                {
                    "sample_id": row["sample_id"],
                    "word_count": row["word_count"],
                    "outside_knowledge_words": ";".join(sorted(hits)),
                    "raw_text_preview": row["raw_text_preview"],
                }
            )
    notable = notable[:40]

    write_csv(out_dir / "describe_visible_condition_summary.csv", form_rows)
    write_csv(out_dir / "describe_visible_poster_delta.csv", poster_rows)
    write_report(out_dir / "describe_visible_repeat_comparison_report.md", form_rows, poster_rows, notable)
    print(f"[INFO] wrote comparison to {out_dir}")


if __name__ == "__main__":
    main()
