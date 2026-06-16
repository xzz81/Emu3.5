#!/usr/bin/env python3
"""Audit manual annotation completion for semantic-entropy extractor validation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SLOTS = ["object_1", "color_1", "object_2", "color_2", "relation", "background"]
VOCAB = {
    "object_1": {"cube", "sphere", "cone", "unknown"},
    "object_2": {"cube", "sphere", "cone", "unknown"},
    "color_1": {"red", "blue", "green", "yellow", "unknown"},
    "color_2": {"red", "blue", "green", "yellow", "unknown"},
    "relation": {
        "object_1_left_of_object_2",
        "object_1_right_of_object_2",
        "object_1_above_object_2",
        "object_1_below_object_2",
        "unknown",
    },
    "background": {"white", "other", "unknown"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extractor-validation-dir", default="outputs/semantic_entropy_umm/extractor_validation")
    parser.add_argument("--out-dir", default="outputs/semantic_entropy_umm/manual_annotation_progress")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0].keys()) if rows else [])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def completion_status(completed: int, required: int, invalid: int, missing_rows: int, duplicate_rows: int) -> str:
    if required == 0:
        return "MISSING_INPUT"
    if missing_rows or duplicate_rows:
        return "ID_MISMATCH"
    if invalid:
        return "INVALID_VALUES"
    if completed == required:
        return "READY_TO_VALIDATE"
    if completed == 0:
        return "EMPTY"
    return "IN_PROGRESS"


def inspect_rows(manifest: list[dict[str, str]], manual: list[dict[str, str]]) -> dict[str, Any]:
    manifest_ids = [row["annotation_id"] for row in manifest]
    manifest_by_id = {row["annotation_id"]: row for row in manifest}
    manual_ids = [row.get("annotation_id", "") for row in manual]
    manual_by_id: dict[str, dict[str, str]] = {}
    duplicate_ids = []
    for row in manual:
        annotation_id = row.get("annotation_id", "")
        if annotation_id in manual_by_id:
            duplicate_ids.append(annotation_id)
        manual_by_id[annotation_id] = row

    missing_ids = [annotation_id for annotation_id in manifest_ids if annotation_id not in manual_by_id]
    extra_ids = sorted(set(manual_ids) - set(manifest_ids))
    slot_counts = {slot: Counter() for slot in SLOTS}
    route_counts: dict[str, Counter] = defaultdict(Counter)
    batch_counts: dict[str, Counter] = defaultdict(Counter)
    invalid_values = []
    missing_cells = []
    completed_slots = 0
    completed_rows = 0

    for position, annotation_id in enumerate(manifest_ids):
        manifest_row = manifest_by_id[annotation_id]
        manual_row = manual_by_id.get(annotation_id, {})
        route = manifest_row.get("route", "UNKNOWN")
        batch_id = f"batch_{position // 10 + 1:02d}"
        row_complete = True
        for slot in SLOTS:
            value = manual_row.get(slot, "").strip()
            if not value:
                row_complete = False
                missing_cells.append({"annotation_id": annotation_id, "slot": slot})
                slot_counts[slot]["missing"] += 1
                route_counts[route]["missing"] += 1
                batch_counts[batch_id]["missing"] += 1
                continue
            if value not in VOCAB[slot]:
                row_complete = False
                invalid_values.append({"annotation_id": annotation_id, "slot": slot, "value": value})
                slot_counts[slot]["invalid"] += 1
                route_counts[route]["invalid"] += 1
                batch_counts[batch_id]["invalid"] += 1
                continue
            completed_slots += 1
            slot_counts[slot]["complete"] += 1
            route_counts[route]["complete"] += 1
            batch_counts[batch_id]["complete"] += 1
            if value == "unknown":
                slot_counts[slot]["unknown"] += 1
                route_counts[route]["unknown"] += 1
                batch_counts[batch_id]["unknown"] += 1
        if row_complete:
            completed_rows += 1
            route_counts[route]["complete_rows"] += 1
            batch_counts[batch_id]["complete_rows"] += 1
        route_counts[route]["rows"] += 1
        batch_counts[batch_id]["rows"] += 1

    required_slots = len(manifest) * len(SLOTS)
    return {
        "manifest_rows": len(manifest),
        "manual_rows": len(manual),
        "completed_rows": completed_rows,
        "completed_slot_labels": completed_slots,
        "required_slot_labels": required_slots,
        "missing_ids": missing_ids,
        "extra_ids": extra_ids,
        "duplicate_ids": duplicate_ids,
        "missing_cells": missing_cells,
        "invalid_values": invalid_values,
        "status": completion_status(
            completed_slots,
            required_slots,
            len(invalid_values),
            len(missing_ids) + len(extra_ids),
            len(duplicate_ids),
        ),
        "slot_counts": slot_counts,
        "route_counts": route_counts,
        "batch_counts": batch_counts,
    }


def summary_rows(report: dict[str, Any], source_name: str) -> list[dict[str, Any]]:
    rows = [
        {
            "source": source_name,
            "scope": "overall",
            "key": "all",
            "rows": report["manifest_rows"],
            "completed_rows": report["completed_rows"],
            "required_slot_labels": report["required_slot_labels"],
            "completed_slot_labels": report["completed_slot_labels"],
            "missing_slot_labels": report["required_slot_labels"] - report["completed_slot_labels"],
            "unknown_labels": "",
            "invalid_values": len(report["invalid_values"]),
            "status": report["status"],
        }
    ]
    for slot in SLOTS:
        counts = report["slot_counts"][slot]
        required = report["manifest_rows"]
        rows.append(
            {
                "source": source_name,
                "scope": "slot",
                "key": slot,
                "rows": report["manifest_rows"],
                "completed_rows": "",
                "required_slot_labels": required,
                "completed_slot_labels": counts["complete"],
                "missing_slot_labels": required - counts["complete"],
                "unknown_labels": counts["unknown"],
                "invalid_values": counts["invalid"],
                "status": "PASS" if counts["complete"] == required and counts["invalid"] == 0 else "INCOMPLETE",
            }
        )
    for route, counts in sorted(report["route_counts"].items()):
        required = counts["rows"] * len(SLOTS)
        rows.append(
            {
                "source": source_name,
                "scope": "route",
                "key": route,
                "rows": counts["rows"],
                "completed_rows": counts["complete_rows"],
                "required_slot_labels": required,
                "completed_slot_labels": counts["complete"],
                "missing_slot_labels": required - counts["complete"],
                "unknown_labels": counts["unknown"],
                "invalid_values": counts["invalid"],
                "status": "PASS" if counts["complete"] == required and counts["invalid"] == 0 else "INCOMPLETE",
            }
        )
    for batch_id, counts in sorted(report["batch_counts"].items()):
        required = counts["rows"] * len(SLOTS)
        rows.append(
            {
                "source": source_name,
                "scope": "batch",
                "key": batch_id,
                "rows": counts["rows"],
                "completed_rows": counts["complete_rows"],
                "required_slot_labels": required,
                "completed_slot_labels": counts["complete"],
                "missing_slot_labels": required - counts["complete"],
                "unknown_labels": counts["unknown"],
                "invalid_values": counts["invalid"],
                "status": "PASS" if counts["complete"] == required and counts["invalid"] == 0 else "INCOMPLETE",
            }
        )
    return rows


def md_table(rows: list[dict[str, Any]], fields: list[str]) -> list[str]:
    if not rows:
        return ["(none)"]
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        safe = {field: str(row.get(field, "")).replace("|", "\\|").replace("\n", " ") for field in fields}
        lines.append("| " + " | ".join(safe[field] for field in fields) + " |")
    return lines


def inspect_batch_files(validation_dir: Path, manifest: list[dict[str, str]]) -> list[dict[str, Any]]:
    batch_root = validation_dir / "annotation_batches"
    rows = []
    for batch_csv in sorted(batch_root.glob("batch_*/batch_*_manual_annotations.csv")):
        batch_manifest_path = batch_csv.with_name(batch_csv.name.replace("_manual_annotations.csv", "_annotation_manifest.csv"))
        batch_manifest = read_csv(batch_manifest_path)
        if not batch_manifest:
            batch_manifest = manifest
        report = inspect_rows(batch_manifest, read_csv(batch_csv))
        rows.append(
            {
                "batch_id": batch_csv.parent.name,
                "manual_path": str(batch_csv),
                "rows": report["manual_rows"],
                "completed_rows": report["completed_rows"],
                "completed_slot_labels": report["completed_slot_labels"],
                "required_slot_labels": report["required_slot_labels"],
                "invalid_values": len(report["invalid_values"]),
                "status": report["status"],
            }
        )
    return rows


def write_report(out_dir: Path, validation_dir: Path, main_report: dict[str, Any], rows: list[dict[str, Any]], batch_rows: list[dict[str, Any]]) -> None:
    overall = rows[0]
    incomplete_slots = [row for row in rows if row["scope"] == "slot" and row["status"] != "PASS"]
    route_rows = [row for row in rows if row["scope"] == "route"]
    batch_summary_rows = [row for row in rows if row["scope"] == "batch"]
    lines = [
        "# Manual Annotation Progress Audit",
        "",
        f"Status: {main_report['status']}",
        "",
        "This report audits annotation completion only. It does not infer labels and does not replace manual validation.",
        "",
        "## Overall",
        "",
        *md_table([overall], ["source", "rows", "completed_rows", "required_slot_labels", "completed_slot_labels", "missing_slot_labels", "invalid_values", "status"]),
        "",
        "## Slots",
        "",
        *md_table([row for row in rows if row["scope"] == "slot"], ["key", "required_slot_labels", "completed_slot_labels", "missing_slot_labels", "unknown_labels", "invalid_values", "status"]),
        "",
        "## Routes",
        "",
        *md_table(route_rows, ["key", "rows", "completed_rows", "required_slot_labels", "completed_slot_labels", "missing_slot_labels", "unknown_labels", "invalid_values", "status"]),
        "",
        "## Batch Progress From Main CSV Order",
        "",
        *md_table(batch_summary_rows, ["key", "rows", "completed_rows", "required_slot_labels", "completed_slot_labels", "missing_slot_labels", "unknown_labels", "invalid_values", "status"]),
        "",
        "## Per-Batch CSV Files",
        "",
        *md_table(batch_rows, ["batch_id", "rows", "completed_rows", "required_slot_labels", "completed_slot_labels", "invalid_values", "status", "manual_path"]),
        "",
        "## Missing Or Invalid Examples",
        "",
        f"- Missing ids: `{len(main_report['missing_ids'])}`",
        f"- Extra ids: `{len(main_report['extra_ids'])}`",
        f"- Duplicate ids: `{len(main_report['duplicate_ids'])}`",
        f"- Missing cells: `{len(main_report['missing_cells'])}`",
        f"- Invalid values: `{len(main_report['invalid_values'])}`",
        "",
        "## Next Gate",
        "",
        f"Fill `{validation_dir / 'manual_annotations.csv'}` until this report reaches `READY_TO_VALIDATE`, then run `build_semantic_entropy_extractor_validation.py --mode validate-manual`.",
    ]
    if incomplete_slots:
        lines.extend(["", "## Incomplete Slots", "", *md_table(incomplete_slots, ["key", "missing_slot_labels", "invalid_values", "status"])])
    (out_dir / "manual_annotation_progress.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    validation_dir = Path(args.extractor_validation_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_csv(validation_dir / "annotation_manifest.csv")
    manual = read_csv(validation_dir / "manual_annotations.csv")
    report = inspect_rows(manifest, manual)
    rows = summary_rows(report, "manual_annotations.csv")
    batch_rows = inspect_batch_files(validation_dir, manifest)
    write_csv(out_dir / "manual_annotation_progress.csv", rows)
    write_csv(out_dir / "manual_annotation_batch_progress.csv", batch_rows)
    json_payload = dict(report)
    for key in ("slot_counts", "route_counts", "batch_counts"):
        json_payload[key] = {name: dict(counts) for name, counts in report[key].items()}
    (out_dir / "manual_annotation_progress.json").write_text(
        json.dumps(json_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(out_dir, validation_dir, report, rows, batch_rows)
    print(json.dumps({"status": report["status"], "completed_slot_labels": report["completed_slot_labels"], "required_slot_labels": report["required_slot_labels"]}, indent=2))


if __name__ == "__main__":
    main()
