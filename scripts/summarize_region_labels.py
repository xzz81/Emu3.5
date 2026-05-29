#!/usr/bin/env python3
# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Summarize manual visual-token region labels in labeled UME traces."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def mean(values: Sequence[float]) -> float | str:
    if not values:
        return ""
    return sum(values) / len(values)


def quantile(values: Sequence[float], fraction: float) -> float | str:
    if not values:
        return ""
    ordered = sorted(values)
    idx = round((len(ordered) - 1) * fraction)
    return ordered[max(0, min(len(ordered) - 1, idx))]


def summarize_group(rows: Sequence[Mapping[str, Any]], group: Mapping[str, Any]) -> Dict[str, Any]:
    umes = [float(row["ume"]) for row in rows if row.get("ume") is not None]
    cfgs = [float(row.get("u_cfg", 0.0)) for row in rows]
    return {
        **group,
        "tokens": len(rows),
        "mean_ume": mean(umes),
        "p50_ume": quantile(umes, 0.5),
        "p90_ume": quantile(umes, 0.9),
        "max_ume": max(umes) if umes else "",
        "mean_u_cfg": mean(cfgs),
        "max_u_cfg": max(cfgs) if cfgs else "",
    }


def collect_rows(trace_dir: Path) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    region_rows: List[Dict[str, Any]] = []
    sample_rows: List[Dict[str, Any]] = []
    for path in sorted(trace_dir.glob("*_entropy.jsonl")):
        records = read_jsonl(path)
        visual = [row for row in records if row.get("token_type") == "visual"]
        sample_id = str(records[0].get("sample_id", path.name.replace("_entropy.jsonl", ""))) if records else path.name
        task = str(records[0].get("task", "unknown")) if records else "unknown"

        grouped: Dict[tuple[str, bool], List[Mapping[str, Any]]] = {}
        for row in visual:
            label = str(row.get("manual_region_label", "") or "background_or_unannotated")
            is_error = bool(row.get("is_error", False))
            grouped.setdefault((label, is_error), []).append(row)
        for (label, is_error), rows in sorted(grouped.items()):
            region_rows.append(
                summarize_group(
                    rows,
                    {
                        "task": task,
                        "sample_id": sample_id,
                        "source_trace": str(path),
                        "region_label": label,
                        "is_error": is_error,
                    },
                )
            )

        annotated = [row for row in visual if row.get("manual_region_label")]
        error = [row for row in visual if row.get("is_error")]
        sample_rows.append(
            {
                "task": task,
                "sample_id": sample_id,
                "source_trace": str(path),
                "visual_tokens": len(visual),
                "annotated_region_tokens": len(annotated),
                "error_tokens": len(error),
                "annotated_region_mean_ume": mean([float(row["ume"]) for row in annotated]),
                "error_region_mean_ume": mean([float(row["ume"]) for row in error]),
                "all_visual_mean_ume": mean([float(row["ume"]) for row in visual]),
            }
        )
    return region_rows, sample_rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt(value: Any) -> str:
    if value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(path: Path, region_rows: Sequence[Mapping[str, Any]], sample_rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Manual Region Label Summary",
        "",
        "This report summarizes visual-token regions labeled by manual token-grid bounding boxes.",
        "",
        "## Samples",
        "",
        "| task | sample_id | visual_tokens | annotated_tokens | error_tokens | annotated_mean_ume | error_mean_ume | all_visual_mean_ume |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sample_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["task"]),
                    str(row["sample_id"]),
                    str(row["visual_tokens"]),
                    str(row["annotated_region_tokens"]),
                    str(row["error_tokens"]),
                    fmt(row["annotated_region_mean_ume"]),
                    fmt(row["error_region_mean_ume"]),
                    fmt(row["all_visual_mean_ume"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Regions", ""])
    lines.extend(
        [
            "| task | sample_id | region_label | is_error | tokens | mean_ume | p90_ume | max_ume | mean_u_cfg |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in region_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["task"]),
                    str(row["sample_id"]),
                    str(row["region_label"]),
                    str(row["is_error"]),
                    str(row["tokens"]),
                    fmt(row["mean_ume"]),
                    fmt(row["p90_ume"]),
                    fmt(row["max_ume"]),
                    fmt(row["mean_u_cfg"]),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    trace_dir = Path(args.trace_dir)
    out_dir = Path(args.out_dir)
    region_rows, sample_rows = collect_rows(trace_dir)
    write_csv(out_dir / "manual_region_label_summary.csv", region_rows)
    write_csv(out_dir / "manual_region_sample_summary.csv", sample_rows)
    write_markdown(out_dir / "manual_region_label_summary.md", region_rows, sample_rows)
    print(f"[INFO] wrote {len(region_rows)} region rows and {len(sample_rows)} sample rows to {out_dir}")


if __name__ == "__main__":
    main()
