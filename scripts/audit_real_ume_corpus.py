#!/usr/bin/env python3
# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Audit the real UME corpus, labels, and region-labeled traces."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


EXPECTED_PHASE1_TASKS = ("t2i", "x2i", "interleaved/story/howto")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-index", default="data/real_ume/real_ume_sample_index.csv")
    parser.add_argument("--labels", default="data/real_ume/manual_sample_judgments.jsonl")
    parser.add_argument("--region-trace-dir", default="outputs/research_logs/region_labeled_entropy_traces")
    parser.add_argument("--out-dir", default="outputs/research_logs/corpus_audit")
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes"}
    return bool(value)


def key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (str(row["task"]), str(row["run_id"]), str(row["sample_id"]))


def short_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return (str(row["task"]), str(row["sample_id"]))


def group_count(rows: Iterable[Mapping[str, Any]], field: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        value = str(row.get(field, ""))
        counts[value] = counts.get(value, 0) + 1
    return counts


def summarize_task_rows(sample_rows: Sequence[Mapping[str, str]], labels: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    labels_by_key = {key(row): row for row in labels}
    rows: List[Dict[str, Any]] = []
    for task in sorted({row["task"] for row in sample_rows}):
        task_rows = [row for row in sample_rows if row["task"] == task]
        task_labels = [labels_by_key[key(row)] for row in task_rows if key(row) in labels_by_key]
        binary = [row for row in task_labels if row.get("include_in_analysis") is True and row.get("is_error") != ""]
        rows.append(
            {
                "task": task,
                "runs": len({row["run_id"] for row in task_rows}),
                "samples": len(task_rows),
                "decoded": sum(1 for row in task_rows if as_bool(row.get("decoded", False))),
                "complete_images": sum(1 for row in task_rows if as_bool(row.get("image_complete", False))),
                "manual_labels": len(task_labels),
                "binary_labels": len(binary),
                "error_labels": sum(1 for row in binary if as_bool(row.get("is_error", False))),
                "visual_tokens": sum(int(row.get("visual_tokens") or 0) for row in task_rows),
            }
        )
    return rows


def find_duplicate_short_keys(sample_rows: Sequence[Mapping[str, str]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str], List[Mapping[str, str]]] = {}
    for row in sample_rows:
        grouped.setdefault(short_key(row), []).append(row)
    duplicates = []
    for (task, sample_id), rows in sorted(grouped.items()):
        run_ids = sorted({row["run_id"] for row in rows})
        if len(run_ids) <= 1:
            continue
        duplicates.append(
            {
                "task": task,
                "sample_id": sample_id,
                "runs": ";".join(run_ids),
                "rows": len(rows),
            }
        )
    return duplicates


def collect_label_audit(sample_rows: Sequence[Mapping[str, str]], labels: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    sample_keys = {key(row) for row in sample_rows}
    label_keys = {key(row) for row in labels}
    matched = sorted(sample_keys & label_keys)
    label_missing_sample = sorted(label_keys - sample_keys)
    sample_missing_label = sorted(sample_keys - label_keys)
    included = [row for row in labels if row.get("include_in_analysis") is True and row.get("is_error") != ""]
    return {
        "sample_rows": len(sample_rows),
        "label_rows": len(labels),
        "matched_labels": len(matched),
        "label_missing_sample": label_missing_sample,
        "sample_missing_label": sample_missing_label,
        "included_binary_labels": len(included),
        "included_errors": sum(1 for row in included if as_bool(row.get("is_error", False))),
    }


def collect_region_trace_audit(trace_dir: Path, labels: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    label_by_short: Dict[tuple[str, str], List[Mapping[str, Any]]] = {}
    for row in labels:
        label_by_short.setdefault(short_key(row), []).append(row)

    trace_files = sorted(trace_dir.glob("*_entropy.jsonl"))
    sample_rows: List[Dict[str, Any]] = []
    total_records = 0
    visual_tokens = 0
    error_tokens = 0
    ambiguous_run_maps: List[Dict[str, str]] = []
    missing_label_maps: List[Dict[str, str]] = []

    for path in trace_files:
        records = read_jsonl(path)
        if not records:
            continue
        total_records += len(records)
        first = records[0]
        task = str(first.get("task", "unknown"))
        sample_id = str(first.get("sample_id", path.name.replace("_entropy.jsonl", "")))
        labels_for_sample = label_by_short.get((task, sample_id), [])
        run_id = ""
        if len(labels_for_sample) == 1:
            run_id = str(labels_for_sample[0].get("run_id", ""))
        elif len(labels_for_sample) > 1:
            ambiguous_run_maps.append({"task": task, "sample_id": sample_id, "trace_file": str(path)})
        else:
            missing_label_maps.append({"task": task, "sample_id": sample_id, "trace_file": str(path)})
        visual = [row for row in records if row.get("token_type") == "visual"]
        errors = [row for row in visual if as_bool(row.get("is_error", False))]
        annotated = [row for row in visual if row.get("manual_region_label")]
        visual_tokens += len(visual)
        error_tokens += len(errors)
        sample_rows.append(
            {
                "task": task,
                "run_id": run_id,
                "sample_id": sample_id,
                "trace_file": str(path),
                "records": len(records),
                "visual_tokens": len(visual),
                "annotated_tokens": len(annotated),
                "error_tokens": len(errors),
                "matched_label_candidates": len(labels_for_sample),
            }
        )

    return {
        "trace_files": len(trace_files),
        "sample_rows": sample_rows,
        "total_records": total_records,
        "visual_tokens": visual_tokens,
        "error_tokens": error_tokens,
        "ambiguous_run_maps": ambiguous_run_maps,
        "missing_label_maps": missing_label_maps,
    }


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


def fmt_list(items: Sequence[Any], limit: int = 20) -> List[str]:
    if not items:
        return ["- none"]
    lines = []
    for item in items[:limit]:
        if isinstance(item, tuple):
            lines.append("- `" + " / ".join(str(part) for part in item) + "`")
        elif isinstance(item, Mapping):
            lines.append("- `" + " / ".join(f"{k}={v}" for k, v in item.items()) + "`")
        else:
            lines.append(f"- `{item}`")
    if len(items) > limit:
        lines.append(f"- ... {len(items) - limit} more")
    return lines


def write_markdown(
    path: Path,
    task_rows: Sequence[Mapping[str, Any]],
    label_audit: Mapping[str, Any],
    duplicates: Sequence[Mapping[str, Any]],
    region_audit: Mapping[str, Any],
) -> None:
    present_tasks = {row["task"] for row in task_rows}
    missing_phase1 = [task for task in EXPECTED_PHASE1_TASKS if task not in present_tasks]
    lines = [
        "# Real UME Corpus Audit",
        "",
        "This audit checks local real UME outputs, manual labels, and region-labeled traces.",
        "Synthetic tests are not counted as research evidence here.",
        "",
        "## Task Coverage",
        "",
        "| task | runs | samples | decoded | complete_images | manual_labels | binary_labels | error_labels | visual_tokens |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in task_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["task"]),
                    str(row["runs"]),
                    str(row["samples"]),
                    str(row["decoded"]),
                    str(row["complete_images"]),
                    str(row["manual_labels"]),
                    str(row["binary_labels"]),
                    str(row["error_labels"]),
                    str(row["visual_tokens"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "Phase-1 planned tasks without real local coverage in the current corpus:"])
    lines.extend(fmt_list(missing_phase1))

    lines.extend(
        [
            "",
            "## Label Audit",
            "",
            f"- Sample index rows: `{label_audit['sample_rows']}`",
            f"- Manual label rows: `{label_audit['label_rows']}`",
            f"- Matched manual labels: `{label_audit['matched_labels']}`",
            f"- Included binary labels: `{label_audit['included_binary_labels']}`",
            f"- Included error labels: `{label_audit['included_errors']}`",
            "",
            "Labels missing from sample index:",
        ]
    )
    lines.extend(fmt_list(label_audit["label_missing_sample"]))
    lines.extend(["", "Sample-index rows without manual labels:"])
    lines.extend(fmt_list(label_audit["sample_missing_label"]))
    lines.extend(["", "Duplicate `(task, sample_id)` keys across runs:"])
    lines.extend(fmt_list(duplicates))

    lines.extend(
        [
            "",
            "## Region Trace Audit",
            "",
            f"- Region-labeled trace files: `{region_audit['trace_files']}`",
            f"- Region-labeled sample rows: `{len(region_audit['sample_rows'])}`",
            f"- Total token records: `{region_audit['total_records']}`",
            f"- Visual tokens: `{region_audit['visual_tokens']}`",
            f"- Error tokens: `{region_audit['error_tokens']}`",
            "",
            "Region traces with ambiguous label-to-run mapping:",
        ]
    )
    lines.extend(fmt_list(region_audit["ambiguous_run_maps"]))
    lines.extend(["", "Region traces without a matching manual label:"])
    lines.extend(fmt_list(region_audit["missing_label_maps"]))

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- T2I and X2I have real local coverage with decoded images, traces, and manual labels.",
            "- Interleaved/story/howto remains a coverage gap because the corresponding main Emu3.5 model is not locally cached.",
            "- Several early smoke/incomplete or duplicate exploratory runs remain in the sample index, but are not included in binary manual-label analysis unless explicitly labeled.",
            "- Region-labeled traces are suitable for region-token AUC analysis; sample-level X2I remains limited by small-N labels.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    sample_index = Path(args.sample_index)
    labels_path = Path(args.labels)
    region_trace_dir = Path(args.region_trace_dir)
    out_dir = Path(args.out_dir)

    sample_rows = read_csv(sample_index)
    labels = read_jsonl(labels_path)
    task_rows = summarize_task_rows(sample_rows, labels)
    label_audit = collect_label_audit(sample_rows, labels)
    duplicates = find_duplicate_short_keys(sample_rows)
    region_audit = collect_region_trace_audit(region_trace_dir, labels)

    write_csv(out_dir / "task_coverage.csv", task_rows)
    write_csv(out_dir / "duplicate_sample_ids.csv", duplicates)
    write_csv(out_dir / "region_trace_samples.csv", region_audit["sample_rows"])
    write_markdown(out_dir / "corpus_audit_report.md", task_rows, label_audit, duplicates, region_audit)
    print(
        f"[INFO] audited {len(sample_rows)} sample rows, {len(labels)} labels, "
        f"{region_audit['trace_files']} region traces; wrote report to {out_dir}"
    )


if __name__ == "__main__":
    main()
