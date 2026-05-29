#!/usr/bin/env python3
# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Build a real-data coverage report against the phase-1 UME plan."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


PHASE1_TARGETS = [
    {
        "plan_item": "t2i_attribute_constraints",
        "task": "t2i",
        "target_samples": 100,
        "first_pass_target": 20,
        "evidence": "real_ume_sample_index.csv task=t2i",
    },
    {
        "plan_item": "x2i_reference_preservation",
        "task": "x2i",
        "target_samples": 100,
        "first_pass_target": 20,
        "evidence": "real_ume_sample_index.csv task=x2i",
    },
    {
        "plan_item": "interleaved_story",
        "task": "interleaved/story",
        "target_samples": 50,
        "first_pass_target": 10,
        "evidence": "requires BAAI/Emu3.5 main interleaved model",
    },
    {
        "plan_item": "visual_guidance_howto",
        "task": "visual_guidance/howto",
        "target_samples": 50,
        "first_pass_target": 10,
        "evidence": "requires BAAI/Emu3.5 main interleaved model",
    },
    {
        "plan_item": "ambiguous_conflict_prompts",
        "task": "mixed",
        "target_samples": 50,
        "first_pass_target": 10,
        "evidence": "keyword bucket across real T2I/X2I run_ids",
    },
]

CONFLICT_KEYWORDS = (
    "contradiction",
    "stress",
    "negation",
    "occlusion",
    "boundary",
    "patch",
    "transparent",
    "blank",
    "empty",
    "hidden",
    "covered",
    "and_blue",
    "word",
)

BUCKET_RULES = [
    ("t2i_attr", "t2i", ("attr",)),
    ("t2i_count_text", "t2i", ("count_text",)),
    ("t2i_spatial_geometry", "t2i", ("spatial", "geometry")),
    ("t2i_contradiction_negation", "t2i", ("contradiction", "negation", "stress", "occlusion")),
    ("t2i_boundary", "t2i", ("boundary",)),
    ("t2i_repeatability", "t2i", ("repeatability",)),
    ("x2i_food_reference", "x2i", ("ref_count_text", "ref_spatial_count", "ref_text_object", "ref_contradiction", "ref_suite", "ref_stress")),
    ("x2i_astronaut_reference", "x2i", ("astronaut", "boundary", "patch", "repeatability", "suit_detail")),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="../research_logs")
    parser.add_argument("--out-dir", default="../research_logs/phase1_coverage")
    parser.add_argument("--hf-cache", default=str(Path.home() / ".cache" / "huggingface" / "hub"))
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


def key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (str(row["task"]), str(row["run_id"]), str(row["sample_id"]))


def binary_label_rows(labels: Iterable[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    return [row for row in labels if row.get("include_in_analysis") is True and row.get("is_error") != ""]


def task_counts(sample_rows: Sequence[Mapping[str, str]], labels: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    labels_by_key = {key(row): row for row in labels}
    counts: Dict[str, Dict[str, Any]] = {}
    for task in sorted({row["task"] for row in sample_rows}):
        rows = [row for row in sample_rows if row["task"] == task]
        matched_labels = [labels_by_key[key(row)] for row in rows if key(row) in labels_by_key]
        binary = binary_label_rows(matched_labels)
        counts[task] = {
            "runs": len({row["run_id"] for row in rows}),
            "samples": len(rows),
            "decoded": sum(1 for row in rows if as_bool(row.get("decoded", ""))),
            "complete_images": sum(1 for row in rows if as_bool(row.get("image_complete", ""))),
            "manual_labels": len(matched_labels),
            "binary_labels": len(binary),
            "errors": sum(1 for row in binary if as_bool(row.get("is_error", False))),
        }
    return counts


def conflict_rows(sample_rows: Sequence[Mapping[str, str]], labels: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    labels_by_key = {key(row): row for row in labels}
    rows = [
        row
        for row in sample_rows
        if any(keyword in f"{row.get('run_id', '')} {row.get('sample_id', '')}".lower() for keyword in CONFLICT_KEYWORDS)
    ]
    matched_labels = [labels_by_key[key(row)] for row in rows if key(row) in labels_by_key]
    binary = binary_label_rows(matched_labels)
    return {
        "runs": len({row["run_id"] for row in rows}),
        "samples": len(rows),
        "decoded": sum(1 for row in rows if as_bool(row.get("decoded", ""))),
        "complete_images": sum(1 for row in rows if as_bool(row.get("image_complete", ""))),
        "manual_labels": len(matched_labels),
        "binary_labels": len(binary),
        "errors": sum(1 for row in binary if as_bool(row.get("is_error", False))),
    }


def model_cache_has_snapshot(hf_cache: Path, repo_cache_name: str) -> bool:
    snapshots = hf_cache / repo_cache_name / "snapshots"
    if not snapshots.exists():
        return False
    return any(path.is_file() for path in snapshots.rglob("*"))


def progress_rows(
    sample_rows: Sequence[Mapping[str, str]],
    labels: Sequence[Mapping[str, Any]],
    hf_cache: Path,
) -> List[Dict[str, Any]]:
    counts_by_task = task_counts(sample_rows, labels)
    conflict = conflict_rows(sample_rows, labels)
    main_model_cached = model_cache_has_snapshot(hf_cache, "models--BAAI--Emu3.5")
    rows: List[Dict[str, Any]] = []
    for spec in PHASE1_TARGETS:
        task = str(spec["task"])
        if task == "mixed":
            counts = conflict
            blocker = ""
        elif task in counts_by_task:
            counts = counts_by_task[task]
            blocker = ""
        else:
            counts = {
                "runs": 0,
                "samples": 0,
                "decoded": 0,
                "complete_images": 0,
                "manual_labels": 0,
                "binary_labels": 0,
                "errors": 0,
            }
            blocker = "" if main_model_cached else "BAAI/Emu3.5 main model is not cached locally"
        target = int(spec["target_samples"])
        first_pass = int(spec["first_pass_target"])
        samples = int(counts["samples"])
        rows.append(
            {
                "plan_item": spec["plan_item"],
                "task": task,
                "target_samples": target,
                "first_pass_target": first_pass,
                "samples": samples,
                "decoded": counts["decoded"],
                "complete_images": counts["complete_images"],
                "binary_labels": counts["binary_labels"],
                "errors": counts["errors"],
                "first_pass_complete": samples >= first_pass,
                "full_target_complete": samples >= target,
                "remaining_to_first_pass": max(0, first_pass - samples),
                "remaining_to_full_target": max(0, target - samples),
                "completion_fraction": samples / target if target else "",
                "blocker": blocker,
                "evidence": spec["evidence"],
            }
        )
    return rows


def match_bucket(row: Mapping[str, str]) -> str:
    run_id = str(row.get("run_id", "")).lower()
    task = str(row.get("task", ""))
    for bucket, bucket_task, needles in BUCKET_RULES:
        if task != bucket_task:
            continue
        if any(needle in run_id for needle in needles):
            return bucket
    return f"{task}_other"


def bucket_rows(sample_rows: Sequence[Mapping[str, str]], labels: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    labels_by_key = {key(row): row for row in labels}
    grouped: Dict[str, List[Mapping[str, str]]] = {}
    for row in sample_rows:
        grouped.setdefault(match_bucket(row), []).append(row)
    rows: List[Dict[str, Any]] = []
    for bucket, items in sorted(grouped.items()):
        matched_labels = [labels_by_key[key(row)] for row in items if key(row) in labels_by_key]
        binary = binary_label_rows(matched_labels)
        rows.append(
            {
                "bucket": bucket,
                "task": items[0]["task"] if items else "",
                "runs": len({row["run_id"] for row in items}),
                "samples": len(items),
                "binary_labels": len(binary),
                "errors": sum(1 for row in binary if as_bool(row.get("is_error", False))),
            }
        )
    return rows


def next_action_rows(progress: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in progress:
        if row.get("blocker"):
            priority = "blocked"
            action = f"Cache/load dependency before running {row['plan_item']}: {row['blocker']}"
        elif int(row["remaining_to_full_target"]) > 0:
            priority = "next"
            action = f"Add real {row['task']} runs; remaining full-target samples: {row['remaining_to_full_target']}"
        else:
            priority = "done"
            action = "Full target reached; continue only for robustness or new strata."
        rows.append(
            {
                "plan_item": row["plan_item"],
                "priority": priority,
                "remaining_to_full_target": row["remaining_to_full_target"],
                "action": action,
            }
        )
    return rows


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> List[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(cell) for cell in row) + " |")
    return lines


def write_report(
    path: Path,
    progress: Sequence[Mapping[str, Any]],
    buckets: Sequence[Mapping[str, Any]],
    actions: Sequence[Mapping[str, Any]],
) -> None:
    lines = [
        "# Phase-1 Coverage Plan",
        "",
        "This report compares the current real local UME corpus against the phase-1 experiment targets in `统一熵研究方案.md`.",
        "Synthetic fixtures are excluded; all counts are read from real run indices, manual labels, and real trace artifacts.",
        "",
        "## Target Progress",
        "",
    ]
    lines.extend(
        markdown_table(
            [
                "plan item",
                "task",
                "target",
                "first pass",
                "samples",
                "binary labels",
                "errors",
                "completion",
                "remaining",
                "blocker",
            ],
            [
                [
                    row["plan_item"],
                    row["task"],
                    row["target_samples"],
                    row["first_pass_target"],
                    row["samples"],
                    row["binary_labels"],
                    row["errors"],
                    row["completion_fraction"],
                    row["remaining_to_full_target"],
                    row["blocker"],
                ]
                for row in progress
            ],
        )
    )
    lines.extend(["", "## Corpus Buckets", ""])
    lines.extend(
        markdown_table(
            ["bucket", "task", "runs", "samples", "binary labels", "errors"],
            [[row["bucket"], row["task"], row["runs"], row["samples"], row["binary_labels"], row["errors"]] for row in buckets],
        )
    )
    lines.extend(["", "## Next Actions", ""])
    lines.extend(
        markdown_table(
            ["plan item", "priority", "remaining", "action"],
            [[row["plan_item"], row["priority"], row["remaining_to_full_target"], row["action"]] for row in actions],
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir)
    hf_cache = Path(args.hf_cache).expanduser()

    sample_rows = read_csv(log_dir / "real_ume_sample_index.csv")
    labels = read_jsonl(log_dir / "manual_sample_judgments.jsonl")
    progress = progress_rows(sample_rows, labels, hf_cache)
    buckets = bucket_rows(sample_rows, labels)
    actions = next_action_rows(progress)

    write_csv(out_dir / "phase1_coverage_progress.csv", progress)
    write_csv(out_dir / "phase1_coverage_buckets.csv", buckets)
    write_csv(out_dir / "phase1_next_actions.csv", actions)
    write_report(out_dir / "phase1_coverage_plan.md", progress, buckets, actions)
    print(f"[INFO] wrote phase-1 coverage plan to {out_dir}")


if __name__ == "__main__":
    main()
