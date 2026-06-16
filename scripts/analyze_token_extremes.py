#!/usr/bin/env python3
# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Build token-level extreme-case catalogs from real region-labeled UME traces."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


CATALOG_COLUMNS = [
    "rank",
    "task",
    "sample_id",
    "trace_file",
    "step",
    "visual_row",
    "visual_col",
    "token_id",
    "token_text",
    "ume",
    "u_tok",
    "u_intra",
    "u_cfg",
    "u_mod",
    "is_error",
    "error_type",
    "manual_region_label",
    "manual_region_id",
]

SUMMARY_COLUMNS = [
    "task",
    "sample_id",
    "trace_file",
    "visual_tokens",
    "error_tokens",
    "error_rate",
    "mean_ume",
    "error_mean_ume",
    "correct_mean_ume",
    "mean_u_cfg",
    "error_mean_u_cfg",
    "correct_mean_u_cfg",
    "low_ume_error_tokens",
    "high_cfg_error_tokens",
    "false_conf_rate",
    "high_cfg_error_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", default="outputs/research_logs/region_labeled_entropy_traces")
    parser.add_argument("--out-dir", default="outputs/research_logs/token_extremes")
    parser.add_argument("--top-k", type=int, default=100)
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def as_bool(value: Any) -> bool | str:
    if value == "":
        return ""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes"}
    return bool(value)


def as_float(value: Any) -> float:
    if value == "" or value is None:
        return 0.0
    return float(value)


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


def collect_visual_rows(trace_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(trace_dir.glob("*_entropy.jsonl")):
        trace_sample = path.name.replace("_entropy.jsonl", "")
        for row in read_jsonl(path):
            if row.get("token_type") != "visual" or "is_error" not in row:
                continue
            out = {
                "task": str(row.get("task", "unknown")),
                "sample_id": str(row.get("sample_id", trace_sample)),
                "trace_file": path.name,
                "step": row.get("step", ""),
                "visual_row": row.get("visual_row", ""),
                "visual_col": row.get("visual_col", ""),
                "token_id": row.get("token_id", ""),
                "token_text": row.get("token_text", ""),
                "ume": as_float(row.get("ume", 0.0)),
                "u_tok": as_float(row.get("u_tok", 0.0)),
                "u_intra": as_float(row.get("u_intra", 0.0)),
                "u_cfg": as_float(row.get("u_cfg", 0.0)),
                "u_mod": as_float(row.get("u_mod", 0.0)),
                "is_error": as_bool(row.get("is_error", False)),
                "error_type": row.get("error_type", ""),
                "manual_region_label": row.get("manual_region_label", ""),
                "manual_region_id": row.get("manual_region_id", ""),
            }
            rows.append(out)
    return rows


def threshold_row(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    ume_values = [as_float(row.get("ume", 0.0)) for row in rows]
    cfg_values = [as_float(row.get("u_cfg", 0.0)) for row in rows]
    ume_q25 = quantile(ume_values, 0.25)
    ume_q50 = quantile(ume_values, 0.50)
    ume_q75 = quantile(ume_values, 0.75)
    cfg_q75 = quantile(cfg_values, 0.75)
    low_threshold = as_float(ume_q25)
    high_cfg_threshold = as_float(cfg_q75)
    error_rows = [row for row in rows if as_bool(row.get("is_error", False)) is True]
    return {
        "visual_tokens": len(rows),
        "error_tokens": len(error_rows),
        "ume_q25": ume_q25,
        "ume_median": ume_q50,
        "ume_q75": ume_q75,
        "u_cfg_q75": cfg_q75,
        "low_ume_error_tokens": sum(1 for row in error_rows if as_float(row.get("ume", 0.0)) <= low_threshold),
        "high_cfg_error_tokens": sum(1 for row in error_rows if as_float(row.get("u_cfg", 0.0)) >= high_cfg_threshold),
    }


def extreme_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    descending: bool,
    top_k: int,
    only_errors: bool = False,
) -> List[Dict[str, Any]]:
    candidates = [row for row in rows if not only_errors or as_bool(row.get("is_error", False)) is True]
    ordered = sorted(
        candidates,
        key=lambda row: (as_float(row.get(metric, 0.0)), str(row.get("sample_id", "")), int(row.get("step", 0) or 0)),
        reverse=descending,
    )
    out: List[Dict[str, Any]] = []
    for rank, row in enumerate(ordered[:top_k], start=1):
        out.append({"rank": rank, **dict(row)})
    return out


def group_by_sample(rows: Sequence[Mapping[str, Any]]) -> Dict[tuple[str, str, str], List[Mapping[str, Any]]]:
    grouped: Dict[tuple[str, str, str], List[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("task", "unknown")), str(row.get("sample_id", "")), str(row.get("trace_file", "")))
        grouped.setdefault(key, []).append(row)
    return grouped


def sample_summary_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    low_ume_threshold: float,
    high_cfg_threshold: float,
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for (task, sample_id, trace_file), items in sorted(group_by_sample(rows).items()):
        error_items = [row for row in items if as_bool(row.get("is_error", False)) is True]
        correct_items = [row for row in items if as_bool(row.get("is_error", False)) is not True]
        low_ume_errors = [row for row in error_items if as_float(row.get("ume", 0.0)) <= low_ume_threshold]
        high_cfg_errors = [row for row in error_items if as_float(row.get("u_cfg", 0.0)) >= high_cfg_threshold]
        output.append(
            {
                "task": task,
                "sample_id": sample_id,
                "trace_file": trace_file,
                "visual_tokens": len(items),
                "error_tokens": len(error_items),
                "error_rate": len(error_items) / len(items) if items else "",
                "mean_ume": mean([as_float(row.get("ume", 0.0)) for row in items]),
                "error_mean_ume": mean([as_float(row.get("ume", 0.0)) for row in error_items]),
                "correct_mean_ume": mean([as_float(row.get("ume", 0.0)) for row in correct_items]),
                "mean_u_cfg": mean([as_float(row.get("u_cfg", 0.0)) for row in items]),
                "error_mean_u_cfg": mean([as_float(row.get("u_cfg", 0.0)) for row in error_items]),
                "correct_mean_u_cfg": mean([as_float(row.get("u_cfg", 0.0)) for row in correct_items]),
                "low_ume_error_tokens": len(low_ume_errors),
                "high_cfg_error_tokens": len(high_cfg_errors),
                "false_conf_rate": len(low_ume_errors) / len(error_items) if error_items else "",
                "high_cfg_error_rate": len(high_cfg_errors) / len(error_items) if error_items else "",
            }
        )
    return output


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        keys: List[str] = []
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        columns = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 4) -> str:
    if value == "" or value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> List[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def catalog_table(rows: Sequence[Mapping[str, Any]], limit: int = 10) -> List[List[Any]]:
    output: List[List[Any]] = []
    for row in rows[:limit]:
        output.append(
            [
                row["rank"],
                row["task"],
                row["sample_id"],
                row["step"],
                row.get("visual_row", ""),
                row.get("visual_col", ""),
                fmt(row.get("ume", "")),
                fmt(row.get("u_cfg", "")),
                row.get("is_error", ""),
                row.get("manual_region_label", ""),
            ]
        )
    return output


def build_report(
    thresholds: Mapping[str, Any],
    high_ume: Sequence[Mapping[str, Any]],
    low_ume_error: Sequence[Mapping[str, Any]],
    high_cfg_error: Sequence[Mapping[str, Any]],
    sample_summary: Sequence[Mapping[str, Any]],
) -> str:
    high_false_conf = sorted(
        sample_summary,
        key=lambda row: (int(row.get("low_ume_error_tokens", 0) or 0), float(row.get("false_conf_rate", 0.0) or 0.0)),
        reverse=True,
    )
    high_cfg_samples = sorted(
        sample_summary,
        key=lambda row: (int(row.get("high_cfg_error_tokens", 0) or 0), float(row.get("high_cfg_error_rate", 0.0) or 0.0)),
        reverse=True,
    )
    lines: List[str] = [
        "# Token Extreme Case Catalog",
        "",
        "This catalog is generated only from real region-labeled Emu3.5 trace files. Synthetic fixtures are not included.",
        "",
        "## Thresholds",
        "",
    ]
    lines.extend(
        md_table(
            ["visual tokens", "error tokens", "UME q25", "UME median", "UME q75", "u_cfg q75", "low-UME error tokens", "high-CFG error tokens"],
            [
                [
                    thresholds["visual_tokens"],
                    thresholds["error_tokens"],
                    fmt(thresholds["ume_q25"]),
                    fmt(thresholds["ume_median"]),
                    fmt(thresholds["ume_q75"]),
                    fmt(thresholds["u_cfg_q75"]),
                    thresholds["low_ume_error_tokens"],
                    thresholds["high_cfg_error_tokens"],
                ]
            ],
        )
    )
    lines.extend(["", "## Highest UME Tokens", ""])
    lines.extend(md_table(["rank", "task", "sample", "step", "row", "col", "UME", "u_cfg", "error", "region"], catalog_table(high_ume)))
    lines.extend(["", "## Lowest UME Error Tokens", ""])
    lines.extend(md_table(["rank", "task", "sample", "step", "row", "col", "UME", "u_cfg", "error", "region"], catalog_table(low_ume_error)))
    lines.extend(["", "## Highest CFG Error Tokens", ""])
    lines.extend(md_table(["rank", "task", "sample", "step", "row", "col", "UME", "u_cfg", "error", "region"], catalog_table(high_cfg_error)))
    lines.extend(["", "## Samples With Most Low-UME Error Tokens", ""])
    lines.extend(
        md_table(
            ["task", "sample", "error tokens", "low-UME errors", "false-conf rate", "mean error UME"],
            [
                [
                    row["task"],
                    row["sample_id"],
                    row["error_tokens"],
                    row["low_ume_error_tokens"],
                    fmt(row["false_conf_rate"]),
                    fmt(row["error_mean_ume"]),
                ]
                for row in high_false_conf[:10]
            ],
        )
    )
    lines.extend(["", "## Samples With Most High-CFG Error Tokens", ""])
    lines.extend(
        md_table(
            ["task", "sample", "error tokens", "high-CFG errors", "high-CFG error rate", "mean error u_cfg"],
            [
                [
                    row["task"],
                    row["sample_id"],
                    row["error_tokens"],
                    row["high_cfg_error_tokens"],
                    fmt(row["high_cfg_error_rate"]),
                    fmt(row["error_mean_u_cfg"]),
                ]
                for row in high_cfg_samples[:10]
            ],
        )
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    trace_dir = Path(args.trace_dir)
    out_dir = Path(args.out_dir)
    rows = collect_visual_rows(trace_dir)
    if not rows:
        raise SystemExit(f"no labeled visual rows found under {trace_dir}")

    thresholds = threshold_row(rows)
    low_ume_threshold = as_float(thresholds["ume_q25"])
    high_cfg_threshold = as_float(thresholds["u_cfg_q75"])
    high_ume = extreme_rows(rows, metric="ume", descending=True, top_k=args.top_k)
    high_ume_error = extreme_rows(rows, metric="ume", descending=True, top_k=args.top_k, only_errors=True)
    low_ume_error = extreme_rows(rows, metric="ume", descending=False, top_k=args.top_k, only_errors=True)
    high_cfg_error = extreme_rows(rows, metric="u_cfg", descending=True, top_k=args.top_k, only_errors=True)
    sample_summary = sample_summary_rows(rows, low_ume_threshold=low_ume_threshold, high_cfg_threshold=high_cfg_threshold)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "token_extreme_thresholds.csv", [thresholds])
    write_csv(out_dir / "high_ume_tokens.csv", high_ume, CATALOG_COLUMNS)
    write_csv(out_dir / "high_ume_error_tokens.csv", high_ume_error, CATALOG_COLUMNS)
    write_csv(out_dir / "low_ume_error_tokens.csv", low_ume_error, CATALOG_COLUMNS)
    write_csv(out_dir / "high_cfg_error_tokens.csv", high_cfg_error, CATALOG_COLUMNS)
    write_csv(out_dir / "sample_extreme_summary.csv", sample_summary, SUMMARY_COLUMNS)
    report = build_report(thresholds, high_ume, low_ume_error, high_cfg_error, sample_summary)
    (out_dir / "token_extremes_report.md").write_text(report, encoding="utf-8")
    print(f"[INFO] collected {len(rows)} labeled visual tokens from {trace_dir}")
    print(f"[INFO] wrote token extreme catalog to {out_dir}")


if __name__ == "__main__":
    main()
