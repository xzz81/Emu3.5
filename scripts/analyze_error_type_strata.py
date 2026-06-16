#!/usr/bin/env python3
# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Stratify real region-token UME diagnostics by manual error type."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence


MetricFn = Callable[[Mapping[str, Any]], float]

METRICS: Dict[str, MetricFn] = {
    "ume_default": lambda row: as_float(row.get("ume", 0.0)),
    "inverse_ume_default": lambda row: -as_float(row.get("ume", 0.0)),
    "u_cfg": lambda row: as_float(row.get("u_cfg", 0.0)),
    "inverse_u_tok": lambda row: -as_float(row.get("u_tok", 0.0)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", default="outputs/research_logs/region_labeled_entropy_traces")
    parser.add_argument("--out-dir", default="outputs/research_logs/error_type_strata")
    parser.add_argument("--min-positive-tokens", type=int, default=50)
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


def binary_auc(scores: Sequence[float], labels: Sequence[int]) -> float | str:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return ""
    ranked = sorted(enumerate(scores), key=lambda item: item[1])
    rank_sum_pos = 0.0
    rank = 1
    idx = 0
    while idx < len(ranked):
        j = idx + 1
        while j < len(ranked) and ranked[j][1] == ranked[idx][1]:
            j += 1
        avg_rank = (rank + rank + (j - idx) - 1) / 2
        for k in range(idx, j):
            original_idx = ranked[k][0]
            if labels[original_idx] == 1:
                rank_sum_pos += avg_rank
        rank += j - idx
        idx = j
    return (rank_sum_pos - positives * (positives + 1) / 2) / (positives * negatives)


def split_labels(value: Any) -> List[str]:
    if value is None:
        return []
    labels = [part.strip() for part in str(value).split(";")]
    return [label for label in labels if label]


def collect_visual_rows(trace_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(trace_dir.glob("*_entropy.jsonl")):
        trace_sample = path.name.replace("_entropy.jsonl", "")
        for row in read_jsonl(path):
            if row.get("token_type") != "visual" or "is_error" not in row:
                continue
            out = dict(row)
            out["task"] = str(row.get("task", "unknown"))
            out["sample_id"] = str(row.get("sample_id", trace_sample))
            out["trace_file"] = path.name
            out["is_error"] = as_bool(row.get("is_error", False))
            rows.append(out)
    return rows


def global_thresholds(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    return {
        "ume_q25": float(quantile([as_float(row.get("ume", 0.0)) for row in rows], 0.25) or 0.0),
        "u_cfg_q75": float(quantile([as_float(row.get("u_cfg", 0.0)) for row in rows], 0.75) or 0.0),
    }


def label_counts(rows: Sequence[Mapping[str, Any]], label_key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        if as_bool(row.get("is_error", False)) is not True:
            continue
        for label in split_labels(row.get(label_key, "")):
            counts[label] = counts.get(label, 0) + 1
    return counts


def candidate_rows(rows: Sequence[Mapping[str, Any]], task: str | None) -> List[Mapping[str, Any]]:
    if task is None:
        return [row for row in rows if as_bool(row.get("is_error", False)) is not True]
    return [row for row in rows if str(row.get("task", "")) == task and as_bool(row.get("is_error", False)) is not True]


def positive_rows(rows: Sequence[Mapping[str, Any]], label_key: str, label: str, task: str | None) -> List[Mapping[str, Any]]:
    output: List[Mapping[str, Any]] = []
    for row in rows:
        if task is not None and str(row.get("task", "")) != task:
            continue
        if as_bool(row.get("is_error", False)) is not True:
            continue
        if label in split_labels(row.get(label_key, "")):
            output.append(row)
    return output


def summarize_stratum(
    rows: Sequence[Mapping[str, Any]],
    *,
    label_key: str,
    label: str,
    group: str,
    task: str | None,
    thresholds: Mapping[str, float],
) -> Dict[str, Any]:
    positives = positive_rows(rows, label_key, label, task)
    negatives = candidate_rows(rows, task)
    eval_rows = positives + negatives
    labels = [1] * len(positives) + [0] * len(negatives)
    metric_aucs = {metric: binary_auc([fn(row) for row in eval_rows], labels) for metric, fn in METRICS.items()}
    comparable = {key: value for key, value in metric_aucs.items() if value != ""}
    best_metric = max(comparable, key=lambda key: float(comparable[key])) if comparable else ""
    samples = {str(row.get("sample_id", "")) for row in positives}
    ume_values = [as_float(row.get("ume", 0.0)) for row in positives]
    cfg_values = [as_float(row.get("u_cfg", 0.0)) for row in positives]
    low_ume = sum(1 for row in positives if as_float(row.get("ume", 0.0)) <= thresholds["ume_q25"])
    high_cfg = sum(1 for row in positives if as_float(row.get("u_cfg", 0.0)) >= thresholds["u_cfg_q75"])
    out: Dict[str, Any] = {
        "label_key": label_key,
        "group": group,
        "task": task or "all",
        "label": label,
        "positive_tokens": len(positives),
        "negative_tokens": len(negatives),
        "positive_samples": len(samples),
        "mean_ume": mean(ume_values),
        "p50_ume": quantile(ume_values, 0.5),
        "p90_ume": quantile(ume_values, 0.9),
        "mean_u_cfg": mean(cfg_values),
        "p90_u_cfg": quantile(cfg_values, 0.9),
        "low_ume_tokens": low_ume,
        "low_ume_rate": low_ume / len(positives) if positives else "",
        "high_cfg_tokens": high_cfg,
        "high_cfg_rate": high_cfg / len(positives) if positives else "",
        "best_metric": best_metric,
        "best_auc": metric_aucs.get(best_metric, "") if best_metric else "",
    }
    out.update({f"auc_{metric}": value for metric, value in metric_aucs.items()})
    return out


def summarize_all(
    rows: Sequence[Mapping[str, Any]],
    *,
    label_key: str,
    min_positive_tokens: int,
    thresholds: Mapping[str, float],
) -> List[Dict[str, Any]]:
    counts = label_counts(rows, label_key)
    labels = sorted(label for label, count in counts.items() if count >= min_positive_tokens)
    output: List[Dict[str, Any]] = []
    for label in labels:
        output.append(summarize_stratum(rows, label_key=label_key, label=label, group="all", task=None, thresholds=thresholds))
        tasks = sorted({str(row.get("task", "unknown")) for row in rows if label in split_labels(row.get(label_key, ""))})
        for task in tasks:
            task_positive_count = sum(
                1
                for row in rows
                if str(row.get("task", "")) == task
                and as_bool(row.get("is_error", False)) is True
                and label in split_labels(row.get(label_key, ""))
            )
            if task_positive_count >= min_positive_tokens:
                output.append(summarize_stratum(rows, label_key=label_key, label=label, group="task", task=task, thresholds=thresholds))
    return output


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in headers:
                headers.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
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


def selected_rows(rows: Sequence[Mapping[str, Any]], *, label_key: str, group: str, limit: int = 12) -> List[Mapping[str, Any]]:
    filtered = [row for row in rows if row["label_key"] == label_key and row["group"] == group]
    return sorted(filtered, key=lambda row: int(row.get("positive_tokens", 0)), reverse=True)[:limit]


def build_report(error_rows: Sequence[Mapping[str, Any]], region_rows: Sequence[Mapping[str, Any]], thresholds: Mapping[str, float]) -> str:
    lines: List[str] = [
        "# Error-Type Stratified UME Diagnostics",
        "",
        "This report is generated only from real manually region-labeled Emu3.5 visual-token traces.",
        "",
        "## Global Thresholds",
        "",
    ]
    lines.extend(md_table(["UME q25", "u_cfg q75"], [[fmt(thresholds["ume_q25"]), fmt(thresholds["u_cfg_q75"])]]))
    for title, rows, label_key in [
        ("Error Type Strata", error_rows, "error_type"),
        ("Manual Region Label Strata", region_rows, "manual_region_label"),
    ]:
        lines.extend(["", f"## {title}", ""])
        table_rows = []
        for row in selected_rows(rows, label_key=label_key, group="all", limit=12):
            table_rows.append(
                [
                    row["label"],
                    row["positive_tokens"],
                    row["positive_samples"],
                    fmt(row["mean_ume"]),
                    fmt(row["low_ume_rate"]),
                    fmt(row["mean_u_cfg"]),
                    fmt(row["high_cfg_rate"]),
                    row["best_metric"],
                    fmt(row["best_auc"]),
                ]
            )
        lines.extend(
            md_table(
                ["label", "tokens", "samples", "mean UME", "low-UME rate", "mean u_cfg", "high-CFG rate", "best metric", "best AUC"],
                table_rows,
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
    thresholds = global_thresholds(rows)
    error_rows = summarize_all(rows, label_key="error_type", min_positive_tokens=args.min_positive_tokens, thresholds=thresholds)
    region_rows = summarize_all(rows, label_key="manual_region_label", min_positive_tokens=args.min_positive_tokens, thresholds=thresholds)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "error_type_strata_summary.csv", error_rows)
    write_csv(out_dir / "manual_region_label_strata_summary.csv", region_rows)
    write_csv(out_dir / "error_type_strata_thresholds.csv", [thresholds])
    (out_dir / "error_type_strata_report.md").write_text(build_report(error_rows, region_rows, thresholds), encoding="utf-8")
    print(f"[INFO] analyzed {len(rows)} real visual-token rows from {trace_dir}")
    print(f"[INFO] wrote {len(error_rows)} error-type rows and {len(region_rows)} region-label rows to {out_dir}")


if __name__ == "__main__":
    main()
