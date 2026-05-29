#!/usr/bin/env python3
# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Leave-one-cluster robustness checks for real UME/error AUC metrics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_auc_uncertainty import collect_sample_rows, label_key, region_metric_functions, sample_metric_values
from analyze_ume_weight_sweep import as_bool, binary_auc, mean, read_csv, read_jsonl


ScoreFn = Callable[[Mapping[str, Any]], float]


REGION_METRICS = (
    "ume_default",
    "inverse_ume_default",
    "u_cfg",
    "inverse_u_tok",
)

SAMPLE_METRICS = (
    "mean_ume_default",
    "inverse_mean_ume_default",
    "p90_ume_default",
    "inverse_p90_ume_default",
    "mean_u_cfg",
    "inverse_mean_u_cfg",
    "p90_u_cfg",
    "inverse_p90_u_cfg",
    "inverse_p90_best_all_mix",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", required=True, help="Region-labeled *_entropy.jsonl traces.")
    parser.add_argument("--sample-index", default="../research_logs/real_ume_sample_index.csv")
    parser.add_argument("--labels", default="../research_logs/manual_sample_judgments.jsonl")
    parser.add_argument("--out-dir", default="../research_logs/auc_robustness")
    return parser.parse_args()


def read_sample_index(path: Path) -> List[Dict[str, str]]:
    return read_csv(path)


def sample_run_lookup(sample_index: Path, labels_path: Path) -> Dict[tuple[str, str], str]:
    lookup: Dict[tuple[str, str], str] = {}
    for row in read_jsonl(labels_path):
        if "run_id" not in row:
            continue
        key = (str(row["task"]), str(row["sample_id"]))
        lookup.setdefault(key, str(row["run_id"]))
    index_candidates: Dict[tuple[str, str], set[str]] = {}
    for row in read_sample_index(sample_index):
        key = (str(row["task"]), str(row["sample_id"]))
        index_candidates.setdefault(key, set()).add(str(row["run_id"]))
    for key, run_ids in index_candidates.items():
        if key not in lookup and len(run_ids) == 1:
            lookup[key] = next(iter(run_ids))
    return lookup


def collect_region_rows(trace_dir: Path, sample_index: Path, labels_path: Path) -> List[Dict[str, Any]]:
    run_lookup = sample_run_lookup(sample_index, labels_path)
    rows: List[Dict[str, Any]] = []
    for path in sorted(trace_dir.glob("*_entropy.jsonl")):
        trace_cluster = path.name.replace("_entropy.jsonl", "")
        for row in read_jsonl(path):
            if row.get("token_type") != "visual":
                continue
            if "is_error" not in row:
                continue
            task = str(row.get("task", "unknown"))
            sample_id = str(row.get("sample_id", trace_cluster))
            run_id = run_lookup.get((task, sample_id), "unknown_run")
            rows.append(
                {
                    **row,
                    "task": task,
                    "run_id": run_id,
                    "cluster_id": f"{task}::{run_id}::{sample_id}",
                    "trace_cluster": trace_cluster,
                }
            )
    return rows


def sample_score_functions() -> Dict[str, ScoreFn]:
    return {metric: (lambda row, metric=metric: float(row.get(metric, 0.0) or 0.0)) for metric in SAMPLE_METRICS}


def auc_for(rows: Sequence[Mapping[str, Any]], score_fn: ScoreFn) -> float | str:
    labels = [1 if as_bool(row.get("is_error", False)) else 0 for row in rows]
    scores = [score_fn(row) for row in rows]
    return binary_auc(scores, labels)


def grouped_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, List[Mapping[str, Any]]]:
    groups: Dict[str, List[Mapping[str, Any]]] = {"all": list(rows)}
    for row in rows:
        groups.setdefault(str(row.get("task", "unknown")), []).append(row)
    return groups


def summarize_leave_one(
    rows: Sequence[Mapping[str, Any]],
    metric_fns: Mapping[str, ScoreFn],
    *,
    level: str,
    omit_key: str,
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for group, group_rows_ in sorted(grouped_rows(rows).items()):
        omit_values = sorted({str(row.get(omit_key, "")) for row in group_rows_ if row.get(omit_key, "") != ""})
        labels = [1 if as_bool(row.get("is_error", False)) else 0 for row in group_rows_]
        for metric, score_fn in metric_fns.items():
            baseline = auc_for(group_rows_, score_fn)
            output.append(
                {
                    "level": level,
                    "group": group,
                    "metric": metric,
                    "omit_type": "none",
                    "omitted_id": "",
                    "rows": len(group_rows_),
                    "clusters": len({row.get("cluster_id", "") for row in group_rows_}),
                    "errors": sum(labels),
                    "auc": baseline,
                    "delta_from_full": "",
                    "mean_score_error": mean(
                        [score_fn(row) for row, label in zip(group_rows_, labels) if label == 1]
                    ),
                    "mean_score_correct": mean(
                        [score_fn(row) for row, label in zip(group_rows_, labels) if label == 0]
                    ),
                }
            )
            for omitted in omit_values:
                kept = [row for row in group_rows_ if str(row.get(omit_key, "")) != omitted]
                kept_labels = [1 if as_bool(row.get("is_error", False)) else 0 for row in kept]
                auc = auc_for(kept, score_fn)
                delta = "" if auc == "" or baseline == "" else float(auc) - float(baseline)
                output.append(
                    {
                        "level": level,
                        "group": group,
                        "metric": metric,
                        "omit_type": omit_key,
                        "omitted_id": omitted,
                        "rows": len(kept),
                        "clusters": len({row.get("cluster_id", "") for row in kept}),
                        "errors": sum(kept_labels),
                        "auc": auc,
                        "delta_from_full": delta,
                        "mean_score_error": mean(
                            [score_fn(row) for row, label in zip(kept, kept_labels) if label == 1]
                        ),
                        "mean_score_correct": mean(
                            [score_fn(row) for row, label in zip(kept, kept_labels) if label == 0]
                        ),
                    }
                )
    return output


def summarize_variation(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    keys = sorted({(row["level"], row["group"], row["metric"], row["omit_type"]) for row in rows if row["omit_type"] != "none"})
    baseline_by_key = {
        (row["level"], row["group"], row["metric"]): row
        for row in rows
        if row["omit_type"] == "none" and row["auc"] != ""
    }
    for level, group, metric, omit_type in keys:
        baseline = baseline_by_key.get((level, group, metric))
        items = [
            row
            for row in rows
            if row["level"] == level
            and row["group"] == group
            and row["metric"] == metric
            and row["omit_type"] == omit_type
            and row["auc"] != ""
        ]
        if baseline is None or not items:
            continue
        aucs = [float(row["auc"]) for row in items]
        deltas = [float(row["delta_from_full"]) for row in items if row["delta_from_full"] != ""]
        worst = max(items, key=lambda row: abs(float(row["delta_from_full"])) if row["delta_from_full"] != "" else -1)
        output.append(
            {
                "level": level,
                "group": group,
                "metric": metric,
                "omit_type": omit_type,
                "baseline_auc": baseline["auc"],
                "leave_one_count": len(items),
                "min_auc": min(aucs),
                "max_auc": max(aucs),
                "mean_auc": mean(aucs),
                "max_abs_delta": max(abs(delta) for delta in deltas) if deltas else "",
                "worst_omitted_id": worst["omitted_id"],
                "worst_auc": worst["auc"],
                "worst_delta": worst["delta_from_full"],
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


def write_markdown(path: Path, summary_rows: Sequence[Mapping[str, Any]]) -> None:
    headline_keys = {
        ("region_token", "all", "inverse_u_tok", "run_id"),
        ("region_token", "all", "u_cfg", "run_id"),
        ("region_token", "x2i", "u_cfg", "run_id"),
        ("region_token", "t2i", "inverse_u_tok", "run_id"),
        ("sample", "all", "inverse_p90_best_all_mix", "run_id"),
        ("sample", "t2i", "inverse_p90_u_cfg", "run_id"),
        ("sample", "x2i", "inverse_p90_ume_default", "run_id"),
    }
    lines = [
        "# AUC Robustness Analysis",
        "",
        "Leave-one-run and leave-one-sample checks computed only from real traced outputs and manual labels.",
        "",
        "## Headline Metrics",
        "",
        "| level | group | metric | omit_type | baseline_auc | leave_one_count | min_auc | max_auc | max_abs_delta | worst_omitted_id | worst_auc |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in summary_rows:
        key = (row["level"], row["group"], row["metric"], row["omit_type"])
        if key not in headline_keys:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["level"]),
                    str(row["group"]),
                    str(row["metric"]),
                    str(row["omit_type"]),
                    fmt(row["baseline_auc"]),
                    str(row["leave_one_count"]),
                    fmt(row["min_auc"]),
                    fmt(row["max_auc"]),
                    fmt(row["max_abs_delta"]),
                    str(row["worst_omitted_id"]),
                    fmt(row["worst_auc"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Full Summary",
            "",
            "| level | group | metric | omit_type | baseline_auc | min_auc | max_auc | mean_auc | max_abs_delta | worst_omitted_id | worst_delta |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
        ]
    )
    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["level"]),
                    str(row["group"]),
                    str(row["metric"]),
                    str(row["omit_type"]),
                    fmt(row["baseline_auc"]),
                    fmt(row["min_auc"]),
                    fmt(row["max_auc"]),
                    fmt(row["mean_auc"]),
                    fmt(row["max_abs_delta"]),
                    str(row["worst_omitted_id"]),
                    fmt(row["worst_delta"]),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    trace_dir = Path(args.trace_dir)
    sample_index = Path(args.sample_index)
    labels = Path(args.labels)
    out_dir = Path(args.out_dir)

    region_fns = {name: region_metric_functions()[name] for name in REGION_METRICS}
    sample_fns = sample_score_functions()
    region_rows = collect_region_rows(trace_dir, sample_index, labels)
    sample_rows = collect_sample_rows(sample_index, labels)

    detail_rows: List[Dict[str, Any]] = []
    for omit_key in ("run_id", "cluster_id"):
        detail_rows.extend(summarize_leave_one(region_rows, region_fns, level="region_token", omit_key=omit_key))
        detail_rows.extend(summarize_leave_one(sample_rows, sample_fns, level="sample", omit_key=omit_key))
    summary_rows = summarize_variation(detail_rows)

    write_csv(out_dir / "auc_robustness_detail.csv", detail_rows)
    write_csv(out_dir / "auc_robustness_summary.csv", summary_rows)
    write_markdown(out_dir / "auc_robustness_report.md", summary_rows)
    print(
        f"[INFO] wrote {len(detail_rows)} detail rows and {len(summary_rows)} summary rows "
        f"using {len(region_rows)} region-token rows and {len(sample_rows)} sample rows to {out_dir}"
    )


if __name__ == "__main__":
    main()
