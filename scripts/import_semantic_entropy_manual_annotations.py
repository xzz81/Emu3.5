#!/usr/bin/env python3
"""Safely import returned manual annotations after human labeling."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SLOTS = ["object_1", "color_1", "object_2", "color_2", "relation", "background"]
MANUAL_FIELDS = ["annotation_id", "annotator_id", *SLOTS, "notes"]
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
    parser.add_argument("--input", required=True, help="Returned manual_annotations.csv from annotator")
    parser.add_argument("--extractor-validation-dir", default="outputs/semantic_entropy_umm/extractor_validation")
    parser.add_argument("--out-dir", default="outputs/semantic_entropy_umm/manual_annotation_import")
    parser.add_argument("--backup-dir", default="outputs/semantic_entropy_umm/extractor_validation/manual_annotation_backups")
    parser.add_argument("--write", action="store_true", help="Write validated annotations into extractor-validation manual_annotations.csv")
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        return fields, list(reader)


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def analyze(manifest_rows: list[dict[str, str]], input_fields: list[str], input_rows: list[dict[str, str]]) -> dict[str, Any]:
    manifest_ids = [row["annotation_id"] for row in manifest_rows]
    manifest_id_set = set(manifest_ids)
    missing_fields = [field for field in MANUAL_FIELDS if field not in input_fields]
    extra_fields = [field for field in input_fields if field not in MANUAL_FIELDS]
    by_id: dict[str, dict[str, str]] = {}
    duplicate_ids = []
    blank_ids = []
    for row in input_rows:
        annotation_id = row.get("annotation_id", "").strip()
        if not annotation_id:
            blank_ids.append(row)
            continue
        if annotation_id in by_id:
            duplicate_ids.append(annotation_id)
        by_id[annotation_id] = row
    missing_ids = [annotation_id for annotation_id in manifest_ids if annotation_id not in by_id]
    extra_ids = sorted(set(by_id) - manifest_id_set)
    missing_cells = []
    invalid_values = []
    completed_slot_labels = 0
    completed_rows = 0
    for annotation_id in manifest_ids:
        row = by_id.get(annotation_id, {})
        row_complete = True
        for slot in SLOTS:
            value = row.get(slot, "").strip()
            if not value:
                row_complete = False
                missing_cells.append({"annotation_id": annotation_id, "slot": slot})
                continue
            if value not in VOCAB[slot]:
                row_complete = False
                invalid_values.append({"annotation_id": annotation_id, "slot": slot, "value": value})
                continue
            completed_slot_labels += 1
        if row_complete:
            completed_rows += 1
    required_slot_labels = len(manifest_rows) * len(SLOTS)
    ready = (
        not missing_fields
        and not duplicate_ids
        and not blank_ids
        and not missing_ids
        and not extra_ids
        and not missing_cells
        and not invalid_values
        and completed_slot_labels == required_slot_labels
    )
    if ready:
        status = "READY_TO_IMPORT"
    elif completed_slot_labels == 0 and not invalid_values:
        status = "EMPTY"
    else:
        status = "NOT_READY"
    return {
        "status": status,
        "ready_to_import": ready,
        "manifest_rows": len(manifest_rows),
        "input_rows": len(input_rows),
        "completed_rows": completed_rows,
        "completed_slot_labels": completed_slot_labels,
        "required_slot_labels": required_slot_labels,
        "missing_fields": missing_fields,
        "extra_fields": extra_fields,
        "missing_ids": missing_ids,
        "extra_ids": extra_ids,
        "duplicate_ids": duplicate_ids,
        "blank_id_rows": len(blank_ids),
        "missing_cells": missing_cells,
        "invalid_values": invalid_values,
    }


def ordered_rows(manifest_rows: list[dict[str, str]], input_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_id = {row["annotation_id"].strip(): row for row in input_rows}
    ordered = []
    for manifest_row in manifest_rows:
        row = by_id[manifest_row["annotation_id"]]
        ordered.append({field: row.get(field, "").strip() for field in MANUAL_FIELDS})
    return ordered


def write_report(out_dir: Path, input_path: Path, target_path: Path, report: dict[str, Any], wrote: bool, backup_path: Path | None) -> None:
    lines = [
        "# Manual Annotation Import Report",
        "",
        f"Status: {report['status']}",
        "",
        "This report validates a returned manual annotation CSV before it can replace the main extractor-validation manual annotations.",
        "",
        "## Summary",
        "",
        f"- Input: `{input_path}`",
        f"- Target: `{target_path}`",
        f"- Input rows: `{report['input_rows']}`",
        f"- Manifest rows: `{report['manifest_rows']}`",
        f"- Completed rows: `{report['completed_rows']}`",
        f"- Completed slot labels: `{report['completed_slot_labels']}` / `{report['required_slot_labels']}`",
        f"- Missing fields: `{len(report['missing_fields'])}`",
        f"- Missing ids: `{len(report['missing_ids'])}`",
        f"- Extra ids: `{len(report['extra_ids'])}`",
        f"- Duplicate ids: `{len(report['duplicate_ids'])}`",
        f"- Missing cells: `{len(report['missing_cells'])}`",
        f"- Invalid values: `{len(report['invalid_values'])}`",
        f"- Wrote target: `{wrote}`",
        f"- Backup path: `{backup_path if backup_path else ''}`",
        "",
        "## Next Step",
        "",
        "If status is `READY_TO_IMPORT`, rerun with `--write`. After import, run manual progress audit and `build_semantic_entropy_extractor_validation.py --mode validate-manual`.",
    ]
    (out_dir / "manual_annotation_import_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    validation_dir = Path(args.extractor_validation_dir)
    out_dir = Path(args.out_dir)
    backup_dir = Path(args.backup_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _, manifest_rows = read_csv(validation_dir / "annotation_manifest.csv")
    input_fields, input_rows = read_csv(input_path)
    report = analyze(manifest_rows, input_fields, input_rows)
    wrote = False
    backup_path = None
    target_path = validation_dir / "manual_annotations.csv"
    if args.write:
        if not report["ready_to_import"]:
            raise SystemExit(f"input is not ready to import: status={report['status']}")
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_path = backup_dir / f"manual_annotations.{stamp}.csv"
        if target_path.exists():
            shutil.copy2(target_path, backup_path)
        write_csv(target_path, ordered_rows(manifest_rows, input_rows), MANUAL_FIELDS)
        wrote = True
    report_payload = {
        **report,
        "input_path": str(input_path),
        "target_path": str(target_path),
        "wrote_target": wrote,
        "backup_path": str(backup_path) if backup_path else "",
    }
    (out_dir / "manual_annotation_import_report.json").write_text(
        json.dumps(report_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(out_dir, input_path, target_path, report, wrote, backup_path)
    print(
        json.dumps(
            {
                "status": report["status"],
                "ready_to_import": report["ready_to_import"],
                "completed_slot_labels": report["completed_slot_labels"],
                "required_slot_labels": report["required_slot_labels"],
                "wrote_target": wrote,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
