#!/usr/bin/env python3
# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Join manual sample judgments with real UME sample summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


METRICS = (
    "visual_mean_ume",
    "visual_p90_ume",
    "visual_max_ume",
    "visual_mean_u_cfg",
    "visual_max_u_cfg",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-index", default="../research_logs/real_ume_sample_index.csv")
    parser.add_argument("--labels", default="../research_logs/manual_sample_judgments.jsonl")
    parser.add_argument("--out-dir", default="../research_logs")
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (str(row["task"]), str(row["run_id"]), str(row["sample_id"]))


def as_bool(value: Any) -> bool | str:
    if value == "":
        return ""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes"}
    return bool(value)


def as_float(value: Any) -> float | str:
    if value == "" or value is None:
        return ""
    return float(value)


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


def join_rows(sample_rows: Sequence[Mapping[str, str]], labels: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    sample_by_key = {key(row): row for row in sample_rows}
    rows: List[Dict[str, Any]] = []
    for label in labels:
        sample = sample_by_key.get(key(label))
        if sample is None:
            rows.append(
                {
                    "task": label["task"],
                    "run_id": label["run_id"],
                    "sample_id": label["sample_id"],
                    "matched": False,
                    "include_in_analysis": as_bool(label.get("include_in_analysis", False)),
                    "verdict": label.get("verdict", ""),
                    "quality_score": label.get("quality_score", ""),
                    "is_error": label.get("is_error", ""),
                    "reason": label.get("reason", ""),
                }
            )
            continue
        row: Dict[str, Any] = {
            "task": sample["task"],
            "run_id": sample["run_id"],
            "sample_id": sample["sample_id"],
            "matched": True,
            "include_in_analysis": as_bool(label.get("include_in_analysis", False)),
            "verdict": label.get("verdict", ""),
            "quality_score": label.get("quality_score", ""),
            "is_error": label.get("is_error", ""),
            "decoded": sample.get("decoded", ""),
            "image_complete": sample.get("image_complete", ""),
            "eos": sample.get("eos", ""),
            "visual_tokens": sample.get("visual_tokens", ""),
            "reason": label.get("reason", ""),
        }
        for metric in METRICS:
            row[metric] = sample.get(metric, "")
        rows.append(row)
    return rows


def analyzable(rows: Iterable[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    filtered = []
    for row in rows:
        if not row.get("matched"):
            continue
        if row.get("include_in_analysis") is not True:
            continue
        if row.get("is_error") == "":
            continue
        filtered.append(row)
    return filtered


def summary_rows(joined: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    groups: Dict[str, List[Mapping[str, Any]]] = {"all": analyzable(joined)}
    for row in groups["all"]:
        groups.setdefault(str(row["task"]), []).append(row)
    for group, items in sorted(groups.items()):
        if not items:
            continue
        labels = [1 if as_bool(row["is_error"]) else 0 for row in items]
        qualities = [float(row["quality_score"]) for row in items if row.get("quality_score") != ""]
        base = {
            "group": group,
            "samples": len(items),
            "errors": sum(labels),
            "successes": len(items) - sum(labels),
            "error_rate": sum(labels) / len(items) if items else "",
            "mean_quality": sum(qualities) / len(qualities) if qualities else "",
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in items if row.get(metric) != ""]
            metric_labels = [label for row, label in zip(items, labels) if row.get(metric) != ""]
            base[f"mean_{metric}"] = sum(values) / len(values) if values else ""
            base[f"corr_{metric}_error"] = pearson(values, metric_labels)
            base[f"auc_{metric}_error"] = binary_auc(values, metric_labels)
        rows.append(base)
    return rows


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


def write_markdown(path: Path, joined: Sequence[Mapping[str, Any]], summaries: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Manual Sample Judgment Analysis",
        "",
        "This report joins real sample-level UME summaries with manually written judgments in `manual_sample_judgments.jsonl`.",
        "Only rows with `include_in_analysis=true` and binary `is_error` are used for correlation/AUC.",
        "",
        "## Summary",
        "",
        "| group | samples | errors | error_rate | mean_quality | corr visual_mean_ume | auc visual_mean_ume | corr visual_p90_ume | auc visual_p90_ume |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["group"]),
                    str(row["samples"]),
                    str(row["errors"]),
                    fmt(row["error_rate"]),
                    fmt(row["mean_quality"]),
                    fmt(row["corr_visual_mean_ume_error"]),
                    fmt(row["auc_visual_mean_ume_error"]),
                    fmt(row["corr_visual_p90_ume_error"]),
                    fmt(row["auc_visual_p90_ume_error"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Joined Labels", ""])
    lines.extend(
        [
            "| task | run_id | sample_id | verdict | include | error | quality | visual_mean_ume | visual_p90_ume | reason |",
            "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in joined:
        reason = str(row.get("reason", "")).replace("|", "/")
        if len(reason) > 120:
            reason = reason[:117] + "..."
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["task"]),
                    str(row["run_id"]),
                    str(row["sample_id"]),
                    str(row.get("verdict", "")),
                    "yes" if row.get("include_in_analysis") is True else "",
                    str(row.get("is_error", "")),
                    fmt(as_float(row.get("quality_score", ""))),
                    fmt(as_float(row.get("visual_mean_ume", ""))),
                    fmt(as_float(row.get("visual_p90_ume", ""))),
                    reason,
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    sample_rows = read_csv(Path(args.sample_index))
    labels = read_jsonl(Path(args.labels))
    joined = join_rows(sample_rows, labels)
    summaries = summary_rows(joined)
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "manual_sample_judgment_joined.csv", joined)
    write_csv(out_dir / "manual_sample_judgment_summary.csv", summaries)
    write_markdown(out_dir / "manual_sample_judgment_analysis.md", joined, summaries)
    matched = sum(1 for row in joined if row.get("matched"))
    analyzed = len(analyzable(joined))
    print(f"[INFO] joined {matched}/{len(joined)} labels; analyzed {analyzed} binary labels")


if __name__ == "__main__":
    main()
