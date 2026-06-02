#!/usr/bin/env python3
"""Aggregate forced-position image-read results by distance to quadrant boundary."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-csv", required=True)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        yield from csv.DictReader(f)


def write_csv(rows, path: Path) -> None:
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


def enrich_rows(sample_rows, metadata):
    out = []
    for row in sample_rows:
        meta = metadata.get(row["sample_suffix"])
        if meta is None:
            continue
        item = dict(row)
        item["margin_px"] = meta["margin_px"]
        item["shape_size"] = meta["shape_size"]
        out.append(item)
    return out


def aggregate(rows, keys):
    grouped = {}
    for row in rows:
        grouped.setdefault(tuple(row[k] for k in keys), []).append(row)
    out = []
    for key, group in sorted(grouped.items(), key=lambda kv: tuple(int(x) if str(x).isdigit() else x for x in kv[0])):
        hits = [r for r in group if int(r["position_hit"]) == 1]
        misses = [r for r in group if int(r["position_hit"]) == 0]
        item = {k: v for k, v in zip(keys, key)}
        item.update(
            {
                "sample_count": len(group),
                "position_hits": len(hits),
                "position_misses": len(misses),
                "position_fail_rate": len(misses) / len(group) if group else 0.0,
                "wrong_position_hits": sum(int(r["wrong_position_hit"]) for r in group),
                "mean_ume_all": mean(float(r["mean_ume"]) for r in group),
                "mean_ume_hit": mean(float(r["mean_ume"]) for r in hits),
                "mean_ume_miss": mean(float(r["mean_ume"]) for r in misses),
                "mean_max_ume": mean(float(r["max_ume"]) for r in group),
            }
        )
        out.append(item)
    return out


def write_report(path: Path, rows, by_margin, by_margin_position, by_margin_mode):
    hits = [r for r in rows if int(r["position_hit"]) == 1]
    misses = [r for r in rows if int(r["position_hit"]) == 0]
    lines = [
        "# Image-read Margin Sweep Report",
        "",
        "This report aggregates forced-position answers by the object's pixel margin from the nearest quadrant boundary.",
        "",
        "## Overall",
        "",
        f"- Samples: {len(rows)}",
        f"- Position hits: {len(hits)}/{len(rows)}",
        f"- Position fail rate: {len(misses) / len(rows):.4f}" if rows else "- Position fail rate: 0.0000",
        f"- Mean UME all: {mean(float(r['mean_ume']) for r in rows):.4f}",
        f"- Mean UME hit: {mean(float(r['mean_ume']) for r in hits):.4f}",
        f"- Mean UME miss: {mean(float(r['mean_ume']) for r in misses):.4f}",
        "",
        "## By Margin",
        "",
    ]
    for row in by_margin:
        lines.append(
            f"- margin={row['margin_px']} px: hits={row['position_hits']}/{row['sample_count']}, "
            f"fail_rate={float(row['position_fail_rate']):.4f}, "
            f"mean_UME={float(row['mean_ume_all']):.4f}, "
            f"hit_UME={float(row['mean_ume_hit']):.4f}, miss_UME={float(row['mean_ume_miss']):.4f}"
        )
    if len({row.get("mode", "") for row in rows}) > 1:
        lines.extend(["", "## Margin x Mode", ""])
        for row in by_margin_mode:
            lines.append(
                f"- margin={row['margin_px']} px / {row['mode']}: "
                f"hits={row['position_hits']}/{row['sample_count']}, "
                f"fail_rate={float(row['position_fail_rate']):.4f}, mean_UME={float(row['mean_ume_all']):.4f}"
            )
    lines.extend(["", "## Margin x Position", ""])
    for row in by_margin_position:
        lines.append(
            f"- margin={row['margin_px']} px / {row['expected_position']}: "
            f"hits={row['position_hits']}/{row['sample_count']}, "
            f"fail_rate={float(row['position_fail_rate']):.4f}, mean_UME={float(row['mean_ume_all']):.4f}"
        )
    high_misses = sorted(misses, key=lambda r: float(r["mean_ume"]), reverse=True)[:12]
    if high_misses:
        lines.extend(["", "## Highest-entropy Misses", ""])
        for row in high_misses:
            text = row["raw_text"][:160].replace("`", "")
            lines.append(
                f"- `{row['sample_id']}` margin={row['margin_px']} expected {row['expected_position']}, "
                f"mean_UME={float(row['mean_ume']):.4f}: `{text}`"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    metadata = {row["sample_suffix"]: row for row in read_csv(Path(args.metadata_csv))}
    sample_rows = list(read_csv(Path(args.sample_csv)))
    rows = enrich_rows(sample_rows, metadata)
    out_dir = Path(args.out_dir)
    by_margin = aggregate(rows, ["margin_px"])
    by_margin_position = aggregate(rows, ["margin_px", "expected_position"])
    by_margin_object = aggregate(rows, ["margin_px", "expected_object"])
    by_margin_mode = aggregate(rows, ["margin_px", "mode"])
    by_margin_mode_position = aggregate(rows, ["margin_px", "mode", "expected_position"])
    write_csv(rows, out_dir / "sample_margin_rows.csv")
    write_csv(by_margin, out_dir / "margin_summary.csv")
    write_csv(by_margin_position, out_dir / "margin_position_summary.csv")
    write_csv(by_margin_object, out_dir / "margin_object_summary.csv")
    write_csv(by_margin_mode, out_dir / "margin_mode_summary.csv")
    write_csv(by_margin_mode_position, out_dir / "margin_mode_position_summary.csv")
    write_report(out_dir / "image_read_margin_sweep_report.md", rows, by_margin, by_margin_position, by_margin_mode)
    print(f"[INFO] wrote margin sweep summaries for {len(rows)} samples to {out_dir}")


if __name__ == "__main__":
    main()
