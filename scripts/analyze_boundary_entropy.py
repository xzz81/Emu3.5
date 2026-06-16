#!/usr/bin/env python3
# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Analyze real UME around image-generation structure/boundary tokens."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


EVENT_NAMES = {
    "<|image start|>": "image_start",
    "<|image token|>": "image_token",
    "<|extra_200|>": "row_break",
    "<|image end|>": "image_end",
    "<|extra_101|>": "ess",
    "<|extra_204|>": "eos",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", default="outputs/research_logs/region_labeled_entropy_traces")
    parser.add_argument("--sample-index", default="data/real_ume/real_ume_sample_index.csv")
    parser.add_argument("--labels", default="data/real_ume/manual_sample_judgments.jsonl")
    parser.add_argument("--out-dir", default="outputs/research_logs/boundary_entropy")
    parser.add_argument("--window", type=int, default=8)
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


def visual_window(records: Sequence[Mapping[str, Any]], event_index: int, direction: str, window: int) -> List[Mapping[str, Any]]:
    rows: List[Mapping[str, Any]] = []
    if direction == "prev":
        idx_range = range(event_index - 1, -1, -1)
    elif direction == "next":
        idx_range = range(event_index + 1, len(records))
    else:
        raise ValueError(f"unknown direction {direction}")
    for idx in idx_range:
        row = records[idx]
        if row.get("token_type") == "visual":
            rows.append(row)
            if len(rows) >= window:
                break
    if direction == "prev":
        rows.reverse()
    return rows


def window_stats(rows: Sequence[Mapping[str, Any]], prefix: str) -> Dict[str, Any]:
    umes = [as_float(row.get("ume", 0.0)) for row in rows]
    cfgs = [as_float(row.get("u_cfg", 0.0)) for row in rows]
    errors = [1 if as_bool(row.get("is_error", False)) is True else 0 for row in rows]
    return {
        f"{prefix}_visual_tokens": len(rows),
        f"{prefix}_mean_ume": mean(umes),
        f"{prefix}_p90_ume": quantile(umes, 0.9),
        f"{prefix}_mean_u_cfg": mean(cfgs),
        f"{prefix}_error_tokens": sum(errors),
        f"{prefix}_error_rate": sum(errors) / len(errors) if errors else "",
    }


def event_rows_for_trace(path: Path, records: Sequence[Mapping[str, Any]], window: int) -> List[Dict[str, Any]]:
    sample_id = str(records[0].get("sample_id", path.name.replace("_entropy.jsonl", ""))) if records else path.name
    task = str(records[0].get("task", "unknown")) if records else "unknown"
    rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(records):
        token_text = str(row.get("token_text", ""))
        if row.get("token_type") != "structure" and token_text not in EVENT_NAMES:
            continue
        event = EVENT_NAMES.get(token_text, "other_structure")
        prev_visual = visual_window(records, idx, "prev", window)
        next_visual = visual_window(records, idx, "next", window)
        rows.append(
            {
                "task": task,
                "sample_id": sample_id,
                "trace_file": path.name,
                "event": event,
                "token_text": token_text,
                "step": row.get("step", ""),
                "segment": row.get("segment", ""),
                "u_mod": as_float(row.get("u_mod", 0.0)),
                "ume": as_float(row.get("ume", 0.0)),
                "u_tok": as_float(row.get("u_tok", 0.0)),
                "u_cfg": as_float(row.get("u_cfg", 0.0)),
                **window_stats(prev_visual, "prev"),
                **window_stats(next_visual, "next"),
            }
        )
    return rows


def collect_event_rows(trace_dir: Path, window: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(trace_dir.glob("*_entropy.jsonl")):
        rows.extend(event_rows_for_trace(path, read_jsonl(path), window))
    return rows


def group_rows(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> Dict[tuple[str, ...], List[Mapping[str, Any]]]:
    grouped: Dict[tuple[str, ...], List[Mapping[str, Any]]] = {}
    for row in rows:
        key = tuple(str(row.get(key, "")) for key in keys)
        grouped.setdefault(key, []).append(row)
    return grouped


def summarize_event_groups(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for (task, event), items in sorted(group_rows(rows, ["task", "event"]).items()):
        output.append(
            {
                "task": task,
                "event": event,
                "events": len(items),
                "mean_u_mod": mean([as_float(row.get("u_mod", 0.0)) for row in items]),
                "max_u_mod": max([as_float(row.get("u_mod", 0.0)) for row in items]) if items else "",
                "mean_ume": mean([as_float(row.get("ume", 0.0)) for row in items]),
                "prev_mean_ume": mean([as_float(row.get("prev_mean_ume", 0.0)) for row in items if row.get("prev_mean_ume", "") != ""]),
                "next_mean_ume": mean([as_float(row.get("next_mean_ume", 0.0)) for row in items if row.get("next_mean_ume", "") != ""]),
                "prev_mean_u_cfg": mean([as_float(row.get("prev_mean_u_cfg", 0.0)) for row in items if row.get("prev_mean_u_cfg", "") != ""]),
                "next_mean_u_cfg": mean([as_float(row.get("next_mean_u_cfg", 0.0)) for row in items if row.get("next_mean_u_cfg", "") != ""]),
                "prev_error_rate": mean([as_float(row.get("prev_error_rate", 0.0)) for row in items if row.get("prev_error_rate", "") != ""]),
                "next_error_rate": mean([as_float(row.get("next_error_rate", 0.0)) for row in items if row.get("next_error_rate", "") != ""]),
            }
        )
    return output


def label_lookup(labels_path: Path) -> Dict[tuple[str, str], bool | str]:
    lookup: Dict[tuple[str, str], bool | str] = {}
    collisions: set[tuple[str, str]] = set()
    for row in read_jsonl(labels_path):
        if as_bool(row.get("include_in_analysis", False)) is not True:
            continue
        if row.get("is_error", "") == "":
            continue
        key = (str(row.get("task", "")), str(row.get("sample_id", "")))
        if key in lookup and lookup[key] != as_bool(row.get("is_error", "")):
            collisions.add(key)
        lookup[key] = as_bool(row.get("is_error", ""))
    for key in collisions:
        lookup.pop(key, None)
    return lookup


def sample_summary_rows(event_rows: Sequence[Mapping[str, Any]], labels_path: Path) -> List[Dict[str, Any]]:
    labels = label_lookup(labels_path)
    output: List[Dict[str, Any]] = []
    for (task, sample_id), items in sorted(group_rows(event_rows, ["task", "sample_id"]).items()):
        by_event = group_rows(items, ["event"])
        first_events = by_event.get(("image_token",), []) or by_event.get(("image_start",), [])
        image_end = by_event.get(("image_end",), [])
        row_breaks = by_event.get(("row_break",), [])
        label = labels.get((task, sample_id), "")
        output.append(
            {
                "task": task,
                "sample_id": sample_id,
                "is_error": label,
                "boundary_events": len(items),
                "row_breaks": len(row_breaks),
                "image_end_events": len(image_end),
                "first_visual_mean_ume": mean([as_float(row.get("next_mean_ume", 0.0)) for row in first_events if row.get("next_mean_ume", "") != ""]),
                "first_visual_mean_u_cfg": mean([as_float(row.get("next_mean_u_cfg", 0.0)) for row in first_events if row.get("next_mean_u_cfg", "") != ""]),
                "row_head_mean_ume": mean([as_float(row.get("next_mean_ume", 0.0)) for row in row_breaks if row.get("next_mean_ume", "") != ""]),
                "row_tail_mean_ume": mean([as_float(row.get("prev_mean_ume", 0.0)) for row in row_breaks if row.get("prev_mean_ume", "") != ""]),
                "last_visual_mean_ume": mean([as_float(row.get("prev_mean_ume", 0.0)) for row in image_end if row.get("prev_mean_ume", "") != ""]),
                "last_visual_mean_u_cfg": mean([as_float(row.get("prev_mean_u_cfg", 0.0)) for row in image_end if row.get("prev_mean_u_cfg", "") != ""]),
            }
        )
    return output


def summarize_sample_auc(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    metrics = ["first_visual_mean_ume", "first_visual_mean_u_cfg", "row_head_mean_ume", "row_tail_mean_ume", "last_visual_mean_ume", "last_visual_mean_u_cfg"]
    output: List[Dict[str, Any]] = []
    groups = {"all": list(rows)}
    for row in rows:
        groups.setdefault(str(row.get("task", "unknown")), []).append(row)
    for group, items in sorted(groups.items()):
        labeled = [row for row in items if row.get("is_error", "") != ""]
        labels = [1 if as_bool(row.get("is_error", False)) is True else 0 for row in labeled]
        for metric in metrics:
            eval_rows = [row for row in labeled if row.get(metric, "") != ""]
            eval_labels = [1 if as_bool(row.get("is_error", False)) is True else 0 for row in eval_rows]
            values = [as_float(row.get(metric, 0.0)) for row in eval_rows]
            output.append(
                {
                    "group": group,
                    "metric": metric,
                    "samples": len(eval_rows),
                    "errors": sum(eval_labels),
                    "mean_error": mean([value for value, label in zip(values, eval_labels) if label == 1]),
                    "mean_correct": mean([value for value, label in zip(values, eval_labels) if label == 0]),
                    "auc_metric_error": binary_auc(values, eval_labels),
                    "auc_inverse_metric_error": binary_auc([-value for value in values], eval_labels),
                }
            )
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


def build_report(event_summary: Sequence[Mapping[str, Any]], sample_auc: Sequence[Mapping[str, Any]]) -> str:
    lines: List[str] = [
        "# Boundary Entropy Diagnostics",
        "",
        "This report uses real Emu3.5 traces to summarize structure-token entropy and adjacent visual-token windows. Structure tokens are often decoding-constrained, so adjacent windows carry most of the useful signal.",
        "",
        "## Boundary Events",
        "",
    ]
    lines.extend(
        md_table(
            ["task", "event", "events", "mean u_mod", "max u_mod", "prev mean UME", "next mean UME", "prev err rate", "next err rate"],
            [
                [
                    row["task"],
                    row["event"],
                    row["events"],
                    fmt(row["mean_u_mod"]),
                    fmt(row["max_u_mod"]),
                    fmt(row["prev_mean_ume"]),
                    fmt(row["next_mean_ume"]),
                    fmt(row["prev_error_rate"]),
                    fmt(row["next_error_rate"]),
                ]
                for row in event_summary
            ],
        )
    )
    lines.extend(["", "## Sample-Level Boundary AUC", ""])
    selected = [
        row
        for row in sample_auc
        if row["group"] in {"all", "t2i", "x2i"}
        and row["metric"] in {"first_visual_mean_ume", "first_visual_mean_u_cfg", "row_head_mean_ume", "last_visual_mean_ume", "last_visual_mean_u_cfg"}
    ]
    lines.extend(
        md_table(
            ["group", "metric", "samples", "errors", "mean error", "mean correct", "AUC metric", "AUC inverse"],
            [
                [
                    row["group"],
                    row["metric"],
                    row["samples"],
                    row["errors"],
                    fmt(row["mean_error"]),
                    fmt(row["mean_correct"]),
                    fmt(row["auc_metric_error"]),
                    fmt(row["auc_inverse_metric_error"]),
                ]
                for row in selected
            ],
        )
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    trace_dir = Path(args.trace_dir)
    out_dir = Path(args.out_dir)
    event_rows = collect_event_rows(trace_dir, args.window)
    event_summary = summarize_event_groups(event_rows)
    sample_rows = sample_summary_rows(event_rows, Path(args.labels))
    sample_auc = summarize_sample_auc(sample_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "boundary_event_rows.csv", event_rows)
    write_csv(out_dir / "boundary_event_summary.csv", event_summary)
    write_csv(out_dir / "boundary_sample_summary.csv", sample_rows)
    write_csv(out_dir / "boundary_sample_auc.csv", sample_auc)
    (out_dir / "boundary_entropy_report.md").write_text(build_report(event_summary, sample_auc), encoding="utf-8")
    print(f"[INFO] analyzed {len(event_rows)} boundary events from {trace_dir}")
    print(f"[INFO] wrote boundary diagnostics to {out_dir}")


if __name__ == "__main__":
    main()
