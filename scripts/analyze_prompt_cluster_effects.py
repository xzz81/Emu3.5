#!/usr/bin/env python3
# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Prompt-cluster sensitivity checks for real UME/error metrics.

This script treats sample ids ending in ``_rN`` as repeated samples from the
same prompt group. It reports both ordinary sample/token AUC and prompt-group
collapsed AUC so repeated prompts do not silently receive extra weight.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
import sys
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_auc_uncertainty import region_metric_functions
from analyze_ume_weight_sweep import as_bool, binary_auc, mean, read_csv, read_jsonl


ScoreFn = Callable[[Mapping[str, Any]], float]

REPEAT_SUFFIX = re.compile(r"^(?P<base>.+)_r(?P<repeat>[0-9]+)$")

SAMPLE_METRICS = {
    "visual_mean_ume": lambda row: float(row.get("visual_mean_ume", 0.0) or 0.0),
    "inverse_visual_mean_ume": lambda row: -float(row.get("visual_mean_ume", 0.0) or 0.0),
    "visual_p90_ume": lambda row: float(row.get("visual_p90_ume", 0.0) or 0.0),
    "inverse_visual_p90_ume": lambda row: -float(row.get("visual_p90_ume", 0.0) or 0.0),
    "visual_mean_u_cfg": lambda row: float(row.get("visual_mean_u_cfg", 0.0) or 0.0),
    "inverse_visual_mean_u_cfg": lambda row: -float(row.get("visual_mean_u_cfg", 0.0) or 0.0),
}

REGION_METRICS = {
    "inverse_ume_default": region_metric_functions()["inverse_ume_default"],
    "u_cfg": region_metric_functions()["u_cfg"],
    "inverse_u_tok": region_metric_functions()["inverse_u_tok"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-index", default="../research_logs/real_ume_sample_index.csv")
    parser.add_argument("--labels", default="../research_logs/manual_sample_judgments.jsonl")
    parser.add_argument("--trace-dir", default="../research_logs/region_labeled_entropy_traces")
    parser.add_argument("--out-dir", default="../research_logs/prompt_cluster_effects")
    return parser.parse_args()


def prompt_base(sample_id: str) -> str:
    match = REPEAT_SUFFIX.match(sample_id)
    return match.group("base") if match else sample_id


def prompt_group_id(task: str, run_id: str, sample_id: str) -> str:
    return "::".join([task, run_id, prompt_base(sample_id)])


def label_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (str(row["task"]), str(row["run_id"]), str(row["sample_id"]))


def read_labels(path: Path) -> Dict[tuple[str, str, str], Dict[str, Any]]:
    labels: Dict[tuple[str, str, str], Dict[str, Any]] = {}
    for row in read_jsonl(path):
        if as_bool(row.get("include_in_analysis", False)) is not True:
            continue
        if row.get("is_error") == "":
            continue
        labels[label_key(row)] = row
    return labels


def collect_sample_rows(sample_index: Path, labels_path: Path) -> List[Dict[str, Any]]:
    labels = read_labels(labels_path)
    rows: List[Dict[str, Any]] = []
    for sample in read_csv(sample_index):
        key = label_key(sample)
        label = labels.get(key)
        if label is None:
            continue
        row = {
            "task": sample["task"],
            "run_id": sample["run_id"],
            "sample_id": sample["sample_id"],
            "prompt_base": prompt_base(sample["sample_id"]),
            "prompt_group": prompt_group_id(sample["task"], sample["run_id"], sample["sample_id"]),
            "is_error": as_bool(label["is_error"]),
            "verdict": label.get("verdict", ""),
        }
        for metric in ("visual_mean_ume", "visual_p90_ume", "visual_mean_u_cfg"):
            row[metric] = float(sample.get(metric, 0.0) or 0.0)
        rows.append(row)
    return rows


def sample_run_lookup(labels_path: Path, sample_index: Path) -> Dict[tuple[str, str], str]:
    candidates: Dict[tuple[str, str], set[str]] = {}
    for row in read_jsonl(labels_path):
        if "run_id" in row:
            candidates.setdefault((str(row.get("task", "")), str(row.get("sample_id", ""))), set()).add(str(row["run_id"]))
    for row in read_csv(sample_index):
        candidates.setdefault((str(row.get("task", "")), str(row.get("sample_id", ""))), set()).add(str(row["run_id"]))
    return {key: next(iter(values)) for key, values in candidates.items() if len(values) == 1}


def collect_region_rows(trace_dir: Path, sample_index: Path, labels_path: Path) -> List[Dict[str, Any]]:
    run_lookup = sample_run_lookup(labels_path, sample_index)
    rows: List[Dict[str, Any]] = []
    for path in sorted(trace_dir.glob("*_entropy.jsonl")):
        trace_sample = path.name.replace("_entropy.jsonl", "")
        for row in read_jsonl(path):
            if row.get("token_type") != "visual" or "is_error" not in row:
                continue
            task = str(row.get("task", "unknown"))
            sample_id = str(row.get("sample_id", trace_sample))
            run_id = str(row.get("run_id") or run_lookup.get((task, sample_id), "unknown_run"))
            rows.append(
                {
                    **row,
                    "task": task,
                    "run_id": run_id,
                    "sample_id": sample_id,
                    "prompt_base": prompt_base(sample_id),
                    "prompt_group": prompt_group_id(task, run_id, sample_id),
                    "is_error": as_bool(row.get("is_error", False)),
                }
            )
    return rows


def group_rows(rows: Sequence[Mapping[str, Any]], key: str) -> Dict[str, List[Mapping[str, Any]]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row[key]), []).append(row)
    return grouped


def task_groups(rows: Sequence[Mapping[str, Any]]) -> Dict[str, List[Mapping[str, Any]]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {"all": list(rows)}
    for row in rows:
        grouped.setdefault(str(row.get("task", "unknown")), []).append(row)
    return grouped


def auc_for(rows: Sequence[Mapping[str, Any]], score_fn: ScoreFn) -> float | str:
    labels = [1 if as_bool(row.get("is_error", False)) else 0 for row in rows]
    scores = [score_fn(row) for row in rows]
    return binary_auc(scores, labels)


def collapsed_prompt_rows(rows: Sequence[Mapping[str, Any]], metric_fns: Mapping[str, ScoreFn]) -> List[Dict[str, Any]]:
    collapsed: List[Dict[str, Any]] = []
    for prompt_group, items in sorted(group_rows(rows, "prompt_group").items()):
        errors = sum(1 if as_bool(row.get("is_error", False)) else 0 for row in items)
        task = str(items[0].get("task", "unknown"))
        run_id = str(items[0].get("run_id", "unknown_run"))
        out: Dict[str, Any] = {
            "task": task,
            "run_id": run_id,
            "prompt_group": prompt_group,
            "prompt_base": str(items[0].get("prompt_base", "")),
            "samples_or_tokens": len(items),
            "errors": errors,
            "error_rate": errors / len(items) if items else "",
            "is_error": errors > 0,
            "any_error": errors > 0,
        }
        for metric, score_fn in metric_fns.items():
            out[metric] = mean([score_fn(row) for row in items])
        collapsed.append(out)
    return collapsed


def auc_summary(
    rows: Sequence[Mapping[str, Any]],
    metric_fns: Mapping[str, ScoreFn],
    *,
    level: str,
    collapsed: bool,
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    eval_rows = collapsed_prompt_rows(rows, metric_fns) if collapsed else list(rows)
    for group, items in sorted(task_groups(eval_rows).items()):
        labels = [1 if as_bool(row.get("is_error", False)) else 0 for row in items]
        for metric, score_fn in metric_fns.items():
            metric_fn = (lambda row, metric=metric: float(row.get(metric, 0.0) or 0.0)) if collapsed else score_fn
            scores = [metric_fn(row) for row in items]
            output.append(
                {
                    "level": level,
                    "aggregation": "prompt_group" if collapsed else "sample_or_token",
                    "group": group,
                    "metric": metric,
                    "rows": len(items),
                    "prompt_groups": len({row.get("prompt_group", "") for row in items}),
                    "errors": sum(labels),
                    "error_rate": sum(labels) / len(labels) if labels else "",
                    "mean_score_error": mean([score for score, label in zip(scores, labels) if label == 1]),
                    "mean_score_correct": mean([score for score, label in zip(scores, labels) if label == 0]),
                    "auc": binary_auc(scores, labels),
                }
            )
    return output


def leave_one_prompt_summary(
    rows: Sequence[Mapping[str, Any]],
    metric_fns: Mapping[str, ScoreFn],
    *,
    level: str,
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for group, items in sorted(task_groups(rows).items()):
        prompt_groups = sorted({str(row["prompt_group"]) for row in items})
        for metric, score_fn in metric_fns.items():
            baseline = auc_for(items, score_fn)
            aucs: List[float] = []
            worst_id = ""
            worst_auc: float | str = ""
            worst_delta: float | str = ""
            for prompt_group in prompt_groups:
                kept = [row for row in items if str(row["prompt_group"]) != prompt_group]
                auc = auc_for(kept, score_fn)
                if auc == "" or baseline == "":
                    continue
                delta = float(auc) - float(baseline)
                aucs.append(float(auc))
                if worst_delta == "" or abs(delta) > abs(float(worst_delta)):
                    worst_id = prompt_group
                    worst_auc = float(auc)
                    worst_delta = delta
            output.append(
                {
                    "level": level,
                    "group": group,
                    "metric": metric,
                    "baseline_auc": baseline,
                    "prompt_groups": len(prompt_groups),
                    "min_auc": min(aucs) if aucs else "",
                    "max_auc": max(aucs) if aucs else "",
                    "mean_auc": mean(aucs),
                    "max_abs_delta": max(abs(float(auc) - float(baseline)) for auc in aucs) if aucs and baseline != "" else "",
                    "worst_omitted_prompt_group": worst_id,
                    "worst_auc": worst_auc,
                    "worst_delta": worst_delta,
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


def write_report(path: Path, auc_rows: Sequence[Mapping[str, Any]], leave_rows: Sequence[Mapping[str, Any]]) -> None:
    interesting = {
        ("sample", "sample_or_token", "all", "visual_mean_ume"),
        ("sample", "prompt_group", "all", "visual_mean_ume"),
        ("sample", "prompt_group", "t2i", "visual_mean_ume"),
        ("sample", "prompt_group", "x2i", "visual_mean_ume"),
        ("region_token", "sample_or_token", "all", "inverse_u_tok"),
        ("region_token", "prompt_group", "all", "inverse_u_tok"),
        ("region_token", "prompt_group", "x2i", "u_cfg"),
    }
    lines = [
        "# Prompt-Cluster Effects",
        "",
        "This report uses only real traced samples and manual labels. Sample ids ending in `_rN` are collapsed into a prompt group for prompt-level checks.",
        "",
        "## Selected AUC Rows",
        "",
        "| level | aggregation | group | metric | rows | prompt_groups | errors | auc | mean_error | mean_correct |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in auc_rows:
        key = (str(row["level"]), str(row["aggregation"]), str(row["group"]), str(row["metric"]))
        if key not in interesting:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["level"]),
                    str(row["aggregation"]),
                    str(row["group"]),
                    str(row["metric"]),
                    str(row["rows"]),
                    str(row["prompt_groups"]),
                    str(row["errors"]),
                    fmt(row["auc"]),
                    fmt(row["mean_score_error"]),
                    fmt(row["mean_score_correct"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Leave-One Prompt Group",
            "",
            "| level | group | metric | baseline_auc | prompt_groups | min_auc | max_auc | max_abs_delta | worst_omitted_prompt_group |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    leave_interesting = {
        ("sample", "all", "visual_mean_ume"),
        ("sample", "t2i", "visual_mean_ume"),
        ("sample", "x2i", "visual_mean_ume"),
        ("region_token", "all", "inverse_u_tok"),
        ("region_token", "x2i", "u_cfg"),
    }
    for row in leave_rows:
        key = (str(row["level"]), str(row["group"]), str(row["metric"]))
        if key not in leave_interesting:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["level"]),
                    str(row["group"]),
                    str(row["metric"]),
                    fmt(row["baseline_auc"]),
                    str(row["prompt_groups"]),
                    fmt(row["min_auc"]),
                    fmt(row["max_auc"]),
                    fmt(row["max_abs_delta"]),
                    str(row["worst_omitted_prompt_group"]),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    sample_index = Path(args.sample_index)
    labels = Path(args.labels)
    trace_dir = Path(args.trace_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_rows = collect_sample_rows(sample_index, labels)
    region_rows = collect_region_rows(trace_dir, sample_index, labels)
    sample_group_rows = collapsed_prompt_rows(sample_rows, SAMPLE_METRICS)
    region_group_rows = collapsed_prompt_rows(region_rows, REGION_METRICS)

    auc_rows = []
    auc_rows.extend(auc_summary(sample_rows, SAMPLE_METRICS, level="sample", collapsed=False))
    auc_rows.extend(auc_summary(sample_rows, SAMPLE_METRICS, level="sample", collapsed=True))
    auc_rows.extend(auc_summary(region_rows, REGION_METRICS, level="region_token", collapsed=False))
    auc_rows.extend(auc_summary(region_rows, REGION_METRICS, level="region_token", collapsed=True))

    leave_rows = []
    leave_rows.extend(leave_one_prompt_summary(sample_rows, SAMPLE_METRICS, level="sample"))
    leave_rows.extend(leave_one_prompt_summary(region_rows, REGION_METRICS, level="region_token"))

    write_csv(out_dir / "sample_prompt_groups.csv", sample_group_rows)
    write_csv(out_dir / "region_prompt_groups.csv", region_group_rows)
    write_csv(out_dir / "prompt_cluster_auc_summary.csv", auc_rows)
    write_csv(out_dir / "prompt_cluster_leave_one_summary.csv", leave_rows)
    write_report(out_dir / "prompt_cluster_effects_report.md", auc_rows, leave_rows)

    print(
        f"[INFO] wrote prompt-cluster effects for {len(sample_rows)} sample rows, "
        f"{len(sample_group_rows)} sample prompt groups, {len(region_rows)} region rows, "
        f"and {len(region_group_rows)} region prompt groups to {out_dir}"
    )


if __name__ == "__main__":
    main()
