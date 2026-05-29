#!/usr/bin/env python3
# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate alternative UME component weights on real labeled traces."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


COMPONENTS = ("u_tok", "u_intra", "u_cfg", "u_mod")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", required=True, help="Region-labeled *_entropy.jsonl traces.")
    parser.add_argument("--sample-index", default="../research_logs/real_ume_sample_index.csv")
    parser.add_argument("--labels", default="../research_logs/manual_sample_judgments.jsonl")
    parser.add_argument("--out-dir", default="../research_logs/ume_weight_sweep")
    parser.add_argument("--step", type=float, default=0.25, help="Weight-grid step size; must evenly divide 1.0.")
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_bool(value: Any) -> bool | str:
    if value == "":
        return ""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes"}
    return bool(value)


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


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | str:
    if len(xs) < 2 or len(xs) != len(ys):
        return ""
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    denom_x = math.sqrt(sum(x * x for x in dx))
    denom_y = math.sqrt(sum(y * y for y in dy))
    if denom_x <= 1e-12 or denom_y <= 1e-12:
        return ""
    return sum(x * y for x, y in zip(dx, dy)) / (denom_x * denom_y)


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


def weight_grid(step: float) -> List[Dict[str, float]]:
    if step <= 0:
        raise ValueError("--step must be positive")
    units = round(1.0 / step)
    if abs(units * step - 1.0) > 1e-9:
        raise ValueError("--step must evenly divide 1.0")
    weights: List[Dict[str, float]] = []
    for a in range(units + 1):
        for b in range(units + 1 - a):
            for c in range(units + 1 - a - b):
                d = units - a - b - c
                weight = {
                    "u_tok": a / units,
                    "u_intra": b / units,
                    "u_cfg": c / units,
                    "u_mod": d / units,
                }
                weights.append(weight)
    return weights


def metric_specs(step: float) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = [
        {"metric": "ume_default", "weights": None, "source": "ume"},
        {"metric": "u_tok", "weights": {"u_tok": 1.0, "u_intra": 0.0, "u_cfg": 0.0, "u_mod": 0.0}},
        {"metric": "u_intra", "weights": {"u_tok": 0.0, "u_intra": 1.0, "u_cfg": 0.0, "u_mod": 0.0}},
        {"metric": "u_cfg", "weights": {"u_tok": 0.0, "u_intra": 0.0, "u_cfg": 1.0, "u_mod": 0.0}},
        {"metric": "u_mod", "weights": {"u_tok": 0.0, "u_intra": 0.0, "u_cfg": 0.0, "u_mod": 1.0}},
    ]
    seen = {spec["metric"] for spec in specs}
    for weights in weight_grid(step):
        name = "w_" + "_".join(f"{key.replace('u_', '')}{int(round(value * 100)):03d}" for key, value in weights.items())
        if name in seen:
            continue
        specs.append({"metric": name, "weights": weights})
        seen.add(name)
    return specs


def metric_value(row: Mapping[str, Any], spec: Mapping[str, Any]) -> float:
    if spec.get("source") == "ume":
        return float(row.get("ume", 0.0) or 0.0)
    weights = spec["weights"]
    return sum(float(weights[key]) * float(row.get(key, 0.0) or 0.0) for key in COMPONENTS)


def collect_region_rows(trace_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(trace_dir.glob("*_entropy.jsonl")):
        for row in read_jsonl(path):
            if row.get("token_type") != "visual":
                continue
            if "is_error" not in row:
                continue
            rows.append(row)
    return rows


def summarize_scores(
    rows: Sequence[Mapping[str, Any]],
    specs: Sequence[Mapping[str, Any]],
    *,
    label_key: str = "is_error",
    groups: Sequence[str] = ("all", "task"),
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    grouped: Dict[tuple[str, str], List[Mapping[str, Any]]] = {("all", "all"): list(rows)}
    if "task" in groups:
        for row in rows:
            grouped.setdefault(("task", str(row.get("task", "unknown"))), []).append(row)
    for (group_type, group), items in sorted(grouped.items()):
        labels = [1 if as_bool(row.get(label_key, False)) else 0 for row in items]
        for spec in specs:
            values = [metric_value(row, spec) for row in items]
            auc = binary_auc(values, labels)
            inv_auc = binary_auc([-value for value in values], labels)
            output.append(
                {
                    "level": "region_token",
                    "group_type": group_type,
                    "group": group,
                    "metric": spec["metric"],
                    "samples": len(items),
                    "errors": sum(labels),
                    "error_rate": sum(labels) / len(labels) if labels else "",
                    "mean_score_error": mean([value for value, label in zip(values, labels) if label == 1]),
                    "mean_score_correct": mean([value for value, label in zip(values, labels) if label == 0]),
                    "corr_score_error": pearson(values, labels),
                    "auc_score_error": auc,
                    "auc_inverse_score_error": inv_auc,
                    "best_auc_any_direction": max(v for v in (auc, inv_auc) if v != "") if auc != "" or inv_auc != "" else "",
                    "best_direction": "score_high_error" if auc != "" and (inv_auc == "" or auc >= inv_auc) else "score_low_error",
                    **weight_columns(spec),
                }
            )
    return output


def weight_columns(spec: Mapping[str, Any]) -> Dict[str, Any]:
    weights = spec.get("weights") or {}
    return {f"weight_{key}": weights.get(key, "") for key in COMPONENTS}


def label_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (str(row["task"]), str(row["run_id"]), str(row["sample_id"]))


def collect_sample_rows(sample_index: Path, labels_path: Path, specs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    labels = [
        row
        for row in read_jsonl(labels_path)
        if as_bool(row.get("include_in_analysis", False)) is True and row.get("is_error") != ""
    ]
    sample_by_key = {label_key(row): row for row in read_csv(sample_index)}
    rows: List[Dict[str, Any]] = []
    for label in labels:
        sample = sample_by_key.get(label_key(label))
        if sample is None:
            continue
        trace_path = Path(sample["trace_path"])
        if not trace_path.exists():
            continue
        visual = [row for row in read_jsonl(trace_path) if row.get("token_type") == "visual"]
        if not visual:
            continue
        row: Dict[str, Any] = {
            "task": sample["task"],
            "run_id": sample["run_id"],
            "sample_id": sample["sample_id"],
            "is_error": as_bool(label.get("is_error")),
            "quality_score": label.get("quality_score", ""),
            "visual_tokens": len(visual),
        }
        for spec in specs:
            values = [metric_value(record, spec) for record in visual]
            row[f"{spec['metric']}__mean"] = mean(values)
            row[f"{spec['metric']}__p90"] = quantile(values, 0.9)
            row[f"{spec['metric']}__max"] = max(values) if values else ""
        rows.append(row)
    return rows


def summarize_sample_scores(
    rows: Sequence[Mapping[str, Any]],
    specs: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    grouped: Dict[tuple[str, str], List[Mapping[str, Any]]] = {("all", "all"): list(rows)}
    for row in rows:
        grouped.setdefault(("task", str(row.get("task", "unknown"))), []).append(row)
    for (group_type, group), items in sorted(grouped.items()):
        labels = [1 if as_bool(row.get("is_error", False)) else 0 for row in items]
        for spec in specs:
            for agg in ("mean", "p90", "max"):
                key = f"{spec['metric']}__{agg}"
                values = [float(row[key]) for row in items if row.get(key) != ""]
                metric_labels = [label for row, label in zip(items, labels) if row.get(key) != ""]
                auc = binary_auc(values, metric_labels)
                inv_auc = binary_auc([-value for value in values], metric_labels)
                output.append(
                    {
                        "level": "sample",
                        "group_type": group_type,
                        "group": group,
                        "metric": spec["metric"],
                        "aggregation": agg,
                        "samples": len(values),
                        "errors": sum(metric_labels),
                        "error_rate": sum(metric_labels) / len(metric_labels) if metric_labels else "",
                        "mean_score_error": mean([value for value, label in zip(values, metric_labels) if label == 1]),
                        "mean_score_correct": mean([value for value, label in zip(values, metric_labels) if label == 0]),
                        "corr_score_error": pearson(values, metric_labels),
                        "auc_score_error": auc,
                        "auc_inverse_score_error": inv_auc,
                        "best_auc_any_direction": max(v for v in (auc, inv_auc) if v != "") if auc != "" or inv_auc != "" else "",
                        "best_direction": "score_high_error" if auc != "" and (inv_auc == "" or auc >= inv_auc) else "score_low_error",
                        **weight_columns(spec),
                    }
                )
    return output


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


def top_rows(rows: Sequence[Mapping[str, Any]], level: str, group: str, key: str, n: int = 10) -> List[Mapping[str, Any]]:
    filtered = [row for row in rows if row.get("level") == level and row.get("group_type") == "all" and row.get("group") == group and row.get(key) != ""]
    return sorted(filtered, key=lambda row: float(row[key]), reverse=True)[:n]


def write_markdown(path: Path, region_rows: Sequence[Mapping[str, Any]], sample_rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# UME Weight Sweep",
        "",
        "This report evaluates alternative linear combinations of UME components on real labeled traces.",
        "Synthetic fixtures are not included in these results.",
        "",
        "## Region Tokens: Best Direct AUC",
        "",
        "| metric | auc | inverse_auc | best_direction | mean_error | mean_correct | weights |",
        "| --- | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for row in top_rows(region_rows, "region_token", "all", "auc_score_error"):
        lines.append(format_result_row(row))

    lines.extend(["", "## Region Tokens: Best Any Direction", "", "| metric | auc | inverse_auc | best_direction | mean_error | mean_correct | weights |", "| --- | ---: | ---: | --- | ---: | ---: | --- |"])
    for row in top_rows(region_rows, "region_token", "all", "best_auc_any_direction"):
        lines.append(format_result_row(row))

    lines.extend(["", "## Samples: Best Direct AUC", "", "| metric | auc | inverse_auc | best_direction | mean_error | mean_correct | weights |", "| --- | ---: | ---: | --- | ---: | ---: | --- |"])
    for row in top_rows(sample_rows, "sample", "all", "auc_score_error"):
        lines.append(format_result_row(row, include_agg=True))

    lines.extend(["", "## Samples: Best Any Direction", "", "| metric | auc | inverse_auc | best_direction | mean_error | mean_correct | weights |", "| --- | ---: | ---: | --- | ---: | ---: | --- |"])
    for row in top_rows(sample_rows, "sample", "all", "best_auc_any_direction"):
        lines.append(format_result_row(row, include_agg=True))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_result_row(row: Mapping[str, Any], include_agg: bool = False) -> str:
    metric = str(row["metric"])
    if include_agg:
        metric = f"{metric}__{row.get('aggregation', '')}"
    weights = ", ".join(
        f"{key.replace('weight_', '')}={fmt(value)}"
        for key, value in row.items()
        if key.startswith("weight_") and value != ""
    )
    return (
        "| "
        + " | ".join(
            [
                metric,
                fmt(row.get("auc_score_error", "")),
                fmt(row.get("auc_inverse_score_error", "")),
                str(row.get("best_direction", "")),
                fmt(row.get("mean_score_error", "")),
                fmt(row.get("mean_score_correct", "")),
                weights,
            ]
        )
        + " |"
    )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = metric_specs(args.step)

    region_records = collect_region_rows(Path(args.trace_dir))
    region_rows = summarize_scores(region_records, specs)
    sample_records = collect_sample_rows(Path(args.sample_index), Path(args.labels), specs)
    sample_rows = summarize_sample_scores(sample_records, specs)

    write_csv(out_dir / "region_token_weight_sweep.csv", region_rows)
    write_csv(out_dir / "sample_weight_sweep.csv", sample_rows)
    write_csv(out_dir / "sample_metric_values.csv", sample_records)
    write_markdown(out_dir / "ume_weight_sweep_report.md", region_rows, sample_rows)
    print(
        f"[INFO] evaluated {len(specs)} metrics on {len(region_records)} region-token rows "
        f"and {len(sample_records)} sample rows; wrote results to {out_dir}"
    )


if __name__ == "__main__":
    main()
