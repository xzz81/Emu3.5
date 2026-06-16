#!/usr/bin/env python3
"""Find and dry-run returned manual annotation CSV candidates."""

from __future__ import annotations

import argparse
import csv
import json
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
DEFAULT_EXCLUDED_PARTS = {
    ".git",
    "__pycache__",
    "manual_annotation_package",
    "manual_annotation_package_qa",
    "manual_annotation_progress",
    "manual_annotation_backups",
    "annotation_batches",
    "manual_annotation_import",
    "manual_annotation_return_scan",
    "returned_manual_gate",
    "returned_manual_gate_smoke",
}
NAME_HINTS = (
    "manual_annotations",
    "manual_annotation",
    "returned",
    "annotation_return",
    "annotations_return",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--extractor-validation-dir",
        default="outputs/semantic_entropy_umm/extractor_validation",
    )
    parser.add_argument(
        "--scan-root",
        action="append",
        default=[],
        help="Directory to scan. Can be passed more than once.",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Specific CSV candidate to analyze. Can be passed more than once.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/semantic_entropy_umm/manual_annotation_return_scan",
    )
    parser.add_argument("--include-known-templates", action="store_true")
    parser.add_argument("--max-files", type=int, default=5000)
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames or [], list(reader)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def should_exclude(path: Path, include_known_templates: bool) -> bool:
    if include_known_templates:
        return False
    parts = set(path.parts)
    if parts & DEFAULT_EXCLUDED_PARTS:
        return True
    if path.name == "manual_annotations.csv" and "extractor_validation" in parts:
        return True
    if path.name in {"manual_annotations_template.csv", "manual_annotation_progress.csv", "manual_annotation_batch_progress.csv"}:
        return True
    return False


def discover_csvs(roots: list[Path], candidates: list[Path], include_known_templates: bool, max_files: int) -> list[Path]:
    discovered: list[Path] = []
    for candidate in candidates:
        if candidate.exists() and candidate.suffix.lower() == ".csv":
            discovered.append(candidate)
    scanned = 0
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            if root.suffix.lower() == ".csv":
                discovered.append(root)
            continue
        for path in root.rglob("*.csv"):
            scanned += 1
            if scanned > max_files:
                break
            lower_name = path.name.lower()
            if not any(hint in lower_name for hint in NAME_HINTS):
                continue
            if should_exclude(path, include_known_templates):
                continue
            discovered.append(path)
    return sorted(set(path.resolve() for path in discovered))


def analyze_candidate(path: Path, manifest_rows: list[dict[str, str]]) -> dict[str, Any]:
    try:
        input_fields, input_rows = read_csv(path)
    except Exception as exc:  # noqa: BLE001 - report bad candidate without aborting scan.
        return {
            "path": str(path),
            "status": "UNREADABLE",
            "error": str(exc),
            "ready_to_import": False,
        }

    manifest_ids = [row["annotation_id"] for row in manifest_rows]
    manifest_id_set = set(manifest_ids)
    missing_fields = [field for field in MANUAL_FIELDS if field not in input_fields]
    if "annotation_id" not in input_fields:
        return {
            "path": str(path),
            "status": "NOT_MANUAL_ANNOTATIONS",
            "ready_to_import": False,
            "fields": input_fields,
            "input_rows": len(input_rows),
            "missing_fields": missing_fields,
        }
    by_id: dict[str, dict[str, str]] = {}
    duplicate_ids = []
    blank_id_rows = 0
    for row in input_rows:
        annotation_id = row.get("annotation_id", "").strip()
        if not annotation_id:
            blank_id_rows += 1
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
        and blank_id_rows == 0
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
    elif completed_slot_labels:
        status = "PARTIAL_OR_INVALID"
    else:
        status = "NOT_READY"
    return {
        "path": str(path),
        "status": status,
        "ready_to_import": ready,
        "fields": input_fields,
        "input_rows": len(input_rows),
        "manifest_rows": len(manifest_rows),
        "completed_rows": completed_rows,
        "completed_slot_labels": completed_slot_labels,
        "required_slot_labels": required_slot_labels,
        "missing_fields": missing_fields,
        "missing_ids_count": len(missing_ids),
        "extra_ids_count": len(extra_ids),
        "duplicate_ids_count": len(duplicate_ids),
        "blank_id_rows": blank_id_rows,
        "missing_cells_count": len(missing_cells),
        "invalid_values_count": len(invalid_values),
    }


def md_table(rows: list[dict[str, Any]]) -> list[str]:
    fields = [
        "path",
        "status",
        "input_rows",
        "completed_slot_labels",
        "required_slot_labels",
        "missing_cells_count",
        "invalid_values_count",
    ]
    if not rows:
        return ["(none)"]
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        safe = {field: str(row.get(field, "")).replace("|", "\\|").replace("\n", " ") for field in fields}
        lines.append("| " + " | ".join(safe[field] for field in fields) + " |")
    return lines


def write_report(out_dir: Path, payload: dict[str, Any]) -> None:
    rows = payload["candidates"]
    lines = [
        "# Manual Annotation Return Scan",
        "",
        f"Status: {payload['status']}",
        "",
        "This scan looks for returned human annotation CSVs and dry-runs import readiness. It never overwrites the main manual annotation table.",
        "",
        "## Summary",
        "",
        f"- Scan roots: `{payload['scan_roots']}`",
        f"- Explicit candidates: `{payload['explicit_candidates']}`",
        f"- Candidate CSVs analyzed: `{len(rows)}`",
        f"- Ready-to-import candidates: `{payload['ready_to_import_count']}`",
        "",
        "## Candidates",
        "",
        *md_table(rows),
        "",
        "## Next Step",
        "",
        "If a row is `READY_TO_IMPORT`, run `scripts/import_semantic_entropy_manual_annotations.py --input <path>` first, then rerun with `--write` only after reviewing the dry-run report.",
    ]
    (out_dir / "manual_annotation_return_scan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    validation_dir = Path(args.extractor_validation_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _, manifest_rows = read_csv(validation_dir / "annotation_manifest.csv")
    roots = [Path(item).expanduser() for item in args.scan_root]
    if not roots and not args.candidate:
        roots = [
            Path("outputs/semantic_entropy_umm/manual_annotation_returns"),
            Path.home() / "Downloads",
            Path.home() / "Desktop",
        ]
    candidates = [Path(item).expanduser() for item in args.candidate]
    discovered = discover_csvs(roots, candidates, args.include_known_templates, args.max_files)
    rows = [analyze_candidate(path, manifest_rows) for path in discovered]
    ready_count = sum(1 for row in rows if row.get("ready_to_import"))
    if ready_count:
        status = "READY_CANDIDATE_FOUND"
    elif rows:
        status = "NO_READY_CANDIDATE"
    else:
        status = "NO_CANDIDATES_FOUND"
    payload = {
        "status": status,
        "scan_roots": [str(root) for root in roots],
        "explicit_candidates": [str(candidate) for candidate in candidates],
        "candidate_count": len(rows),
        "ready_to_import_count": ready_count,
        "candidates": rows,
        "claim_boundary": "return scan only; not manual validation evidence",
    }
    (out_dir / "manual_annotation_return_scan.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(
        out_dir / "manual_annotation_return_scan.csv",
        rows,
        [
            "path",
            "status",
            "ready_to_import",
            "input_rows",
            "manifest_rows",
            "completed_rows",
            "completed_slot_labels",
            "required_slot_labels",
            "missing_fields",
            "missing_ids_count",
            "extra_ids_count",
            "duplicate_ids_count",
            "blank_id_rows",
            "missing_cells_count",
            "invalid_values_count",
        ],
    )
    write_report(out_dir, payload)
    print(json.dumps({"status": status, "candidate_count": len(rows), "ready_to_import_count": ready_count}, indent=2))


if __name__ == "__main__":
    main()
