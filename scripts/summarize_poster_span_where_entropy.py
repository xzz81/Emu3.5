#!/usr/bin/env python3
"""Summarize where manually audited poster spans fall relative to top-entropy regions."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-examples", type=int, default=8)
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
    if value in ("", None):
        return 0.0
    return float(value)


def group_summary(rows: list[dict], keys: list[str]) -> list[dict]:
    grouped: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    out = []
    for values, group in sorted(grouped.items()):
        entropy_classes = Counter(row["entropy_class"] for row in group)
        row = {key: value for key, value in zip(keys, values)}
        row.update(
            {
                "span_count": len(group),
                "mean_span_ume": mean(fnum(r["mean_ume"]) for r in group),
                "mean_delta_vs_same_length": mean(fnum(r["mean_ume_minus_baseline"]) for r in group),
                "exact_top20_span_rate": mean(fnum(r["exact_top20_fraction"]) > 0 for r in group),
                "near_top20_span_rate": mean(fnum(r["near_top20_fraction"]) > 0 for r in group),
                "mean_exact_top20_fraction": mean(fnum(r["exact_top20_fraction"]) for r in group),
                "mean_near_top20_fraction": mean(fnum(r["near_top20_fraction"]) for r in group),
                "entropy_class_counts": ";".join(f"{k}:{v}" for k, v in entropy_classes.most_common()),
                "example_snippets": " | ".join(r["snippet"] for r in group[:3]),
            }
        )
        out.append(row)
    return out


def example_rows(rows: list[dict], truth_label: str, high: bool, max_examples: int) -> list[dict]:
    subset = [r for r in rows if r["truth_label"] == truth_label]
    if high:
        subset = sorted(subset, key=lambda r: (fnum(r["exact_top20_fraction"]), fnum(r["mean_ume"])), reverse=True)
    else:
        subset = sorted(subset, key=lambda r: (fnum(r["exact_top20_fraction"]), fnum(r["mean_ume"])))
    out = []
    for row in subset[:max_examples]:
        out.append(
            {
                "truth_label": row["truth_label"],
                "entropy_side": "high" if high else "low_or_not_exact_top20",
                "label_id": row["label_id"],
                "audit": row.get("audit", ""),
                "sample_id": row["sample_id"],
                "aspect": row["aspect"],
                "entropy_class": row["entropy_class"],
                "mean_ume": row["mean_ume"],
                "exact_top20_fraction": row["exact_top20_fraction"],
                "near_top20_fraction": row["near_top20_fraction"],
                "snippet": row["snippet"],
                "notes": row["notes"],
                "context": row["context"],
            }
        )
    return out


def write_report(path: Path, truth_rows: list[dict], aspect_rows: list[dict], examples: list[dict]) -> None:
    lines = [
        "# Poster Manual Span: Where Are The Top-Entropy Regions?",
        "",
        "Rows are manually audited spans from real downloaded poster readback outputs.",
        "Exact top20 means at least one token in the audited span is itself in the answer's top-20% UME tokens; near top20 uses the existing +/- neighborhood measure.",
        "",
        "## Truth Labels",
        "",
    ]
    for row in truth_rows:
        lines.append(
            f"- {row['truth_label']}: n={row['span_count']}, mean UME={float(row['mean_span_ume']):.4f}, "
            f"delta={float(row['mean_delta_vs_same_length']):+.4f}, exact={float(row['exact_top20_span_rate']):.3f}, "
            f"near={float(row['near_top20_span_rate']):.3f}, classes={row['entropy_class_counts']}"
        )
    lines.extend(["", "## Aspect x Truth", ""])
    for row in sorted(aspect_rows, key=lambda r: (r["truth_label"], -int(r["span_count"]), r["aspect"])):
        lines.append(
            f"- {row['truth_label']} / {row['aspect']}: n={row['span_count']}, "
            f"mean UME={float(row['mean_span_ume']):.4f}, exact={float(row['exact_top20_span_rate']):.3f}, "
            f"near={float(row['near_top20_span_rate']):.3f}; examples={row['example_snippets']}"
        )
    lines.extend(["", "## Representative Examples", ""])
    for row in examples:
        lines.append(
            f"- {row['truth_label']} {row['entropy_side']} / {row['aspect']} / {row['entropy_class']}: "
            f"`{row['snippet']}` mean={float(row['mean_ume']):.4f}, exact={float(row['exact_top20_fraction']):.3f}, "
            f"near={float(row['near_top20_fraction']):.3f}. {row['notes']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(Path(args.rows))
    truth_rows = group_summary(rows, ["truth_label"])
    aspect_rows = group_summary(rows, ["truth_label", "aspect"])
    audit_truth_rows = group_summary(rows, ["audit", "truth_label"])
    examples = []
    for truth in ("hallucinated", "grounded", "ambiguous"):
        examples.extend(example_rows(rows, truth, True, args.max_examples))
        examples.extend(example_rows(rows, truth, False, args.max_examples))
    write_csv(out_dir / "poster_span_where_truth_summary.csv", truth_rows)
    write_csv(out_dir / "poster_span_where_aspect_truth_summary.csv", aspect_rows)
    write_csv(out_dir / "poster_span_where_audit_truth_summary.csv", audit_truth_rows)
    write_csv(out_dir / "poster_span_where_representative_examples.csv", examples)
    write_report(out_dir / "poster_span_where_entropy_report.md", truth_rows, aspect_rows, examples)
    print(f"[INFO] wrote poster span where-entropy summary for {len(rows)} spans to {out_dir}")


if __name__ == "__main__":
    main()
