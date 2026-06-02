#!/usr/bin/env python3
"""Summarize repeated image-read sampling stability from span correctness rows."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        yield from csv.DictReader(f)


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


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def repeat_id(sample_id: str) -> str:
    match = re.search(r"_rep(\d+)_", sample_id)
    return match.group(1) if match else ""


def aggregate_case(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["sample_suffix"], []).append(row)
    out = []
    for suffix, group in sorted(grouped.items()):
        hits = [r for r in group if int(r["position_hit"]) == 1]
        misses = [r for r in group if int(r["position_hit"]) == 0]
        out.append(
            {
                "sample_suffix": suffix,
                "expected_object": group[0]["expected_object"],
                "expected_position": group[0]["expected_position"],
                "repeat_count": len(group),
                "position_hits": len(hits),
                "position_misses": len(misses),
                "position_fail_rate": len(misses) / len(group) if group else 0.0,
                "mean_ume_all": mean(float(r["mean_ume"]) for r in group),
                "mean_ume_hit": mean(float(r["mean_ume"]) for r in hits),
                "mean_ume_miss": mean(float(r["mean_ume"]) for r in misses),
                "color_hits": sum(int(r["color_hit"]) for r in group),
                "exact_noun_hits": sum(int(r["exact_noun_hit"]) for r in group),
                "action_scripts": sum(int(r["action_script"]) for r in group),
                "wrong_position_hits": sum(int(r["wrong_position_hit"]) for r in group),
            }
        )
    return out


def aggregate_position(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["expected_position"], []).append(row)
    out = []
    for position, group in sorted(grouped.items()):
        hits = sum(int(r["position_hit"]) for r in group)
        out.append(
            {
                "expected_position": position,
                "sample_count": len(group),
                "position_hits": hits,
                "position_misses": len(group) - hits,
                "position_fail_rate": (len(group) - hits) / len(group) if group else 0.0,
                "mean_ume": mean(float(r["mean_ume"]) for r in group),
                "wrong_position_hits": sum(int(r["wrong_position_hit"]) for r in group),
            }
        )
    return out


def aggregate_repeat(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(repeat_id(row["sample_id"]), []).append(row)
    out = []
    for rep, group in sorted(grouped.items()):
        hits = sum(int(r["position_hit"]) for r in group)
        out.append(
            {
                "repeat": rep,
                "sample_count": len(group),
                "position_hits": hits,
                "position_misses": len(group) - hits,
                "position_fail_rate": (len(group) - hits) / len(group) if group else 0.0,
                "mean_ume": mean(float(r["mean_ume"]) for r in group),
                "action_scripts": sum(int(r["action_script"]) for r in group),
            }
        )
    return out


def write_report(path: Path, rows, case_rows, position_rows, repeat_rows):
    hits = [r for r in rows if int(r["position_hit"]) == 1]
    misses = [r for r in rows if int(r["position_hit"]) == 0]
    action_scripts = [r for r in rows if int(r["action_script"]) == 1]
    wrong_positions = [r for r in rows if int(r["wrong_position_hit"]) == 1]

    lines = [
        "# Image-read Repeated Spatial Prompt Stability",
        "",
        "This report summarizes repeated stochastic samples for the spatialdescribe prompt.",
        "",
        "## Overall",
        "",
        f"- Samples: {len(rows)}",
        f"- Position hits: {len(hits)}/{len(rows)}",
        f"- Position fail rate: {len(misses) / len(rows):.4f}" if rows else "- Position fail rate: 0.0000",
        f"- Mean UME all: {mean(float(r['mean_ume']) for r in rows):.4f}",
        f"- Mean UME hit: {mean(float(r['mean_ume']) for r in hits):.4f}",
        f"- Mean UME miss: {mean(float(r['mean_ume']) for r in misses):.4f}",
        f"- Color hits: {sum(int(r['color_hit']) for r in rows)}/{len(rows)}",
        f"- Exact noun hits: {sum(int(r['exact_noun_hit']) for r in rows)}/{len(rows)}",
        f"- Action scripts: {len(action_scripts)}/{len(rows)}",
        f"- Wrong position hits: {len(wrong_positions)}/{len(rows)}",
        "",
        "## By Position",
        "",
    ]
    for row in position_rows:
        lines.append(
            f"- {row['expected_position']}: hits={row['position_hits']}/{row['sample_count']}, "
            f"fail_rate={float(row['position_fail_rate']):.4f}, mean_UME={float(row['mean_ume']):.4f}"
        )
    lines.extend(["", "## By Case", ""])
    for row in case_rows:
        lines.append(
            f"- {row['sample_suffix']} ({row['expected_object']}, {row['expected_position']}): "
            f"hits={row['position_hits']}/{row['repeat_count']}, "
            f"fail_rate={float(row['position_fail_rate']):.4f}, "
            f"mean_UME_hit={float(row['mean_ume_hit']):.4f}, "
            f"mean_UME_miss={float(row['mean_ume_miss']):.4f}"
        )
    lines.extend(["", "## By Repeat", ""])
    for row in repeat_rows:
        lines.append(
            f"- repeat {row['repeat']}: hits={row['position_hits']}/{row['sample_count']}, "
            f"fail_rate={float(row['position_fail_rate']):.4f}, mean_UME={float(row['mean_ume']):.4f}"
        )

    miss_examples = misses[:10]
    if miss_examples:
        lines.extend(["", "## Miss Examples", ""])
        for row in miss_examples:
            text = row["raw_text"][:240].replace("`", "")
            lines.append(
                f"- `{row['sample_id']}` expected {row['expected_position']}, "
                f"mean_UME={float(row['mean_ume']):.4f}: `{text}`"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = list(read_csv(Path(args.sample_csv)))
    out_dir = Path(args.out_dir)
    case_rows = aggregate_case(rows)
    position_rows = aggregate_position(rows)
    repeat_rows = aggregate_repeat(rows)
    write_csv(case_rows, out_dir / "repeat_case_stability.csv")
    write_csv(position_rows, out_dir / "repeat_position_stability.csv")
    write_csv(repeat_rows, out_dir / "repeat_index_stability.csv")
    write_report(out_dir / "image_read_repeat_stability_report.md", rows, case_rows, position_rows, repeat_rows)
    print(f"[INFO] wrote repeated stability summaries for {len(rows)} samples to {out_dir}")


if __name__ == "__main__":
    main()
