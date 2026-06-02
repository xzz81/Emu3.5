#!/usr/bin/env python3
"""Summarize GT lexical mismatch/top-entropy overlap by prompt variant."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", required=True)
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
    if value in ("", None):
        return 0.0
    return float(value)


def variant(sample_id: str) -> str:
    if "__seed72_" in sample_id:
        return sample_id.rsplit("__seed72_", 1)[-1]
    suffix = sample_id.rsplit("__", 1)[-1]
    return re.sub(r"^(?:greedy_|seed\d+_)+", "", suffix)


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[variant(row["sample_id"])].append(row)
    out = []
    for key, group in sorted(grouped.items()):
        mismatch = sum(int(r["mismatch_token_count"]) for r in group)
        exact = sum(int(r["mismatch_exact_top20_count"]) for r in group)
        near = sum(int(r["mismatch_near_top20_count"]) for r in group)
        token_count = sum(int(r["token_count"]) for r in group)
        baseline = (
            sum(fnum(r["all_token_near_top20_rate"]) * int(r["token_count"]) for r in group) / token_count
            if token_count
            else 0.0
        )
        mismatch_umes = [fnum(r["mean_mismatch_ume"]) for r in group if r["mean_mismatch_ume"] != ""]
        non_mismatch_umes = [fnum(r["mean_non_mismatch_ume"]) for r in group if r["mean_non_mismatch_ume"] != ""]
        out.append(
            {
                "prompt_variant": key,
                "sample_count": len(group),
                "token_count": token_count,
                "mismatch_token_count": mismatch,
                "mismatch_tokens_per_sample": mismatch / len(group) if group else 0.0,
                "exact_top20_rate_weighted": exact / mismatch if mismatch else "",
                "near_top20_rate_weighted": near / mismatch if mismatch else "",
                "all_token_near_top20_baseline_weighted": baseline,
                "near_enrichment_vs_baseline": (near / mismatch) / baseline if mismatch and baseline else "",
                "mean_sample_mismatch_ume": mean(mismatch_umes) if mismatch_umes else "",
                "mean_sample_non_mismatch_ume": mean(non_mismatch_umes) if non_mismatch_umes else "",
                "mean_sample_mismatch_minus_non_mismatch_ume": (
                    mean(mismatch_umes) - mean(non_mismatch_umes) if mismatch_umes and non_mismatch_umes else ""
                ),
            }
        )
    return out


def write_report(path: Path, rows: list[dict]) -> None:
    lines = [
        "# GT Mismatch / Top-Entropy By Prompt Variant",
        "",
        "Automatic lexical mismatch compares candidate answer content words against the seed51 real-poster GT readback vocabulary. It is a noisy broad stress signal, not a final hallucination label.",
        "",
    ]
    for row in rows:
        lines.append(
            f"- {row['prompt_variant']}: samples={row['sample_count']}, mismatch={row['mismatch_token_count']} "
            f"({float(row['mismatch_tokens_per_sample']):.1f}/sample), exact={float(row['exact_top20_rate_weighted'] or 0):.3f}, "
            f"near={float(row['near_top20_rate_weighted'] or 0):.3f}, baseline={float(row['all_token_near_top20_baseline_weighted']):.3f}, "
            f"enrich={float(row['near_enrichment_vs_baseline'] or 0):.3f}, "
            f"mismatch UME={float(row['mean_sample_mismatch_ume'] or 0):.4f}, "
            f"non-mismatch UME={float(row['mean_sample_non_mismatch_ume'] or 0):.4f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = summarize(read_csv(Path(args.summary_csv)))
    write_csv(out_dir / "gt_mismatch_by_prompt_variant.csv", rows)
    write_report(out_dir / "gt_mismatch_by_prompt_variant.md", rows)
    print(f"[INFO] wrote mismatch-by-variant summary for {len(rows)} variants to {out_dir}")


if __name__ == "__main__":
    main()
