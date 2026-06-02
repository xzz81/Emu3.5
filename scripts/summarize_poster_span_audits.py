#!/usr/bin/env python3
"""Summarize poster manual span audits across deterministic and sampled runs."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="append", required=True, help="NAME=CSV_PATH")
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


def fnum(value: str) -> float:
    return float(value) if value not in {"", None} else 0.0


def parse_audits(items: list[str]) -> list[tuple[str, Path]]:
    out = []
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--audit must be NAME=CSV_PATH, got {item}")
        name, path = item.split("=", 1)
        out.append((name, Path(path)))
    return out


def summarize_group(name: str, rows: list[dict]) -> list[dict]:
    by_truth: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_truth[row["truth_label"]].append(row)
    out = []
    for truth, group in sorted(by_truth.items()):
        aspects = Counter(row["aspect"] for row in group)
        out.append(
            {
                "audit": name,
                "truth_label": truth,
                "span_count": len(group),
                "mean_span_ume": f"{mean(fnum(row['mean_ume']) for row in group):.4f}",
                "mean_delta_vs_same_length": f"{mean(fnum(row['mean_ume_minus_baseline']) for row in group):+.4f}",
                "exact_top20_span_rate": f"{mean(fnum(row['exact_top20_fraction']) > 0 for row in group):.3f}",
                "near_top20_span_rate": f"{mean(fnum(row['near_top20_fraction']) > 0 for row in group):.3f}",
                "aspect_counts": ";".join(f"{key}:{value}" for key, value in aspects.most_common()),
            }
        )
    return out


def write_report(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Poster Span Audit Summary",
        "",
        "This report combines manual span audits that compare GT-grounded, hallucinated, and ambiguous poster-answer spans against answer-token UME.",
        "",
    ]
    for row in rows:
        lines.append(
            f"- {row['audit']} / {row['truth_label']}: n={row['span_count']}, "
            f"mean UME={row['mean_span_ume']}, delta={row['mean_delta_vs_same_length']}, "
            f"exact top20 span rate={row['exact_top20_span_rate']}, near={row['near_top20_span_rate']}, "
            f"aspects={row['aspect_counts']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    all_rows = []
    for name, path in parse_audits(args.audit):
        rows = read_csv(path)
        for row in rows:
            row = dict(row)
            row["audit"] = name
            all_rows.append(row)
    summary_rows = []
    for name in sorted({row["audit"] for row in all_rows}):
        summary_rows.extend(summarize_group(name, [row for row in all_rows if row["audit"] == name]))
    summary_rows.extend(summarize_group("combined", all_rows))
    write_csv(out_dir / "poster_span_audit_truth_summary.csv", summary_rows)
    write_csv(out_dir / "poster_span_audit_all_rows.csv", all_rows)
    write_report(out_dir / "poster_span_audit_summary.md", summary_rows)
    print(f"[INFO] wrote {len(summary_rows)} summary rows and {len(all_rows)} audit rows to {out_dir}")


if __name__ == "__main__":
    main()
