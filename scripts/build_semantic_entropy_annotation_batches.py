#!/usr/bin/env python3
"""Prepare and optionally merge manual annotation batches."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


MANUAL_FIELDS = [
    "annotation_id",
    "annotator_id",
    "object_1",
    "color_1",
    "object_2",
    "color_2",
    "relation",
    "background",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prepare", "merge"), default="prepare")
    parser.add_argument("--extractor-validation-dir", default="outputs/semantic_entropy_umm/extractor_validation")
    parser.add_argument("--out-dir", default="outputs/semantic_entropy_umm/extractor_validation/annotation_batches")
    parser.add_argument("--rows-per-batch", type=int, default=10)
    parser.add_argument("--write-main", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0].keys()) if rows else [])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def manual_by_id(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {row["annotation_id"]: row for row in read_csv(path)}


def chunks(rows: list[dict], size: int) -> list[list[dict]]:
    return [rows[idx : idx + size] for idx in range(0, len(rows), size)]


def completed_slots(rows: list[dict]) -> int:
    slots = ["object_1", "color_1", "object_2", "color_2", "relation", "background"]
    return sum(bool(row.get(slot, "").strip()) for row in rows for slot in slots)


def prepare(args: argparse.Namespace) -> None:
    validation_dir = Path(args.extractor_validation_dir)
    out_dir = Path(args.out_dir)
    manifest_path = validation_dir / "annotation_manifest.csv"
    manual_path = validation_dir / "manual_annotations.csv"
    manifest = read_csv(manifest_path)
    manual = manual_by_id(manual_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    batch_rows = []
    for batch_idx, batch_manifest in enumerate(chunks(manifest, args.rows_per_batch), start=1):
        batch_id = f"batch_{batch_idx:02d}"
        batch_dir = out_dir / batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)
        manual_rows = []
        for row in batch_manifest:
            existing = manual.get(row["annotation_id"], {})
            manual_row = {"annotation_id": row["annotation_id"]}
            for field in MANUAL_FIELDS[1:]:
                manual_row[field] = existing.get(field, "")
            manual_rows.append(manual_row)
        manifest_out = batch_dir / f"{batch_id}_annotation_manifest.csv"
        manual_out = batch_dir / f"{batch_id}_manual_annotations.csv"
        write_csv(manifest_out, batch_manifest)
        write_csv(manual_out, manual_rows, MANUAL_FIELDS)
        (batch_dir / f"{batch_id}_annotation_ids.txt").write_text(
            "\n".join(row["annotation_id"] for row in batch_manifest) + "\n",
            encoding="utf-8",
        )
        sheet_name = f"contact_sheet_{batch_idx:02d}.png"
        contact_sheet = validation_dir / "contact_sheets" / sheet_name
        readme = [
            f"# Annotation {batch_id}",
            "",
            f"- Rows: {len(batch_manifest)}",
            f"- Manual CSV: `{manual_out}`",
            f"- Manifest CSV: `{manifest_out}`",
            f"- Contact sheet: `{contact_sheet}`" if contact_sheet.exists() else "- Contact sheet: not found",
            "",
            "Fill all six slot columns using the fixed vocabulary, then merge batches with:",
            "",
            "```bash",
            "python scripts/build_semantic_entropy_annotation_batches.py --mode merge",
            "```",
            "",
            "The merge command writes `merged_manual_annotations.csv` and does not overwrite the main manual file unless `--write-main` is set.",
        ]
        (batch_dir / f"{batch_id}_README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
        batch_rows.append(
            {
                "batch_id": batch_id,
                "rows": len(batch_manifest),
                "completed_slot_labels": completed_slots(manual_rows),
                "manifest_path": str(manifest_out),
                "manual_path": str(manual_out),
                "contact_sheet": str(contact_sheet) if contact_sheet.exists() else "",
            }
        )
    write_csv(out_dir / "annotation_batches_manifest.csv", batch_rows)
    report_lines = [
        "# Annotation Batches",
        "",
        "Status: READY_FOR_BATCH_LABELING",
        "",
        "This package splits the fixed extractor-validation manifest into small manual-labeling batches. It does not alter the main `manual_annotations.csv`.",
        "",
        "## Summary",
        "",
        f"- Source manifest rows: {len(manifest)}",
        f"- Rows per batch: {args.rows_per_batch}",
        f"- Batches: {len(batch_rows)}",
        f"- Current completed slot labels copied into batches: {sum(row['completed_slot_labels'] for row in batch_rows)}",
        "",
        "## Merge",
        "",
        "```bash",
        "python scripts/build_semantic_entropy_annotation_batches.py --mode merge",
        "```",
        "",
        "Use `--write-main` only after checking `merged_manual_annotations.csv`.",
        "",
        "## Batches",
        "",
        "| Batch | Rows | Completed Slot Labels | Manual CSV | Contact Sheet |",
        "|---|---:|---:|---|---|",
    ]
    for row in batch_rows:
        report_lines.append(
            f"| {row['batch_id']} | {row['rows']} | {row['completed_slot_labels']} | `{row['manual_path']}` | `{row['contact_sheet']}` |"
        )
    (out_dir / "annotation_batches_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "READY_FOR_BATCH_LABELING", "batches": len(batch_rows), "rows": len(manifest)}, indent=2))


def merge(args: argparse.Namespace) -> None:
    validation_dir = Path(args.extractor_validation_dir)
    out_dir = Path(args.out_dir)
    manifest = read_csv(validation_dir / "annotation_manifest.csv")
    expected_ids = [row["annotation_id"] for row in manifest]
    merged_by_id: dict[str, dict] = {}
    for path in sorted(out_dir.glob("batch_*/batch_*_manual_annotations.csv")):
        for row in read_csv(path):
            annotation_id = row["annotation_id"]
            if annotation_id in merged_by_id:
                raise ValueError(f"duplicate annotation_id across batches: {annotation_id}")
            merged_by_id[annotation_id] = row
    missing = [annotation_id for annotation_id in expected_ids if annotation_id not in merged_by_id]
    extra = sorted(set(merged_by_id) - set(expected_ids))
    if missing or extra:
        raise ValueError(f"batch merge id mismatch: missing={missing[:10]} extra={extra[:10]}")
    merged = [merged_by_id[annotation_id] for annotation_id in expected_ids]
    merged_path = out_dir / "merged_manual_annotations.csv"
    write_csv(merged_path, merged, MANUAL_FIELDS)
    if args.write_main:
        write_csv(validation_dir / "manual_annotations.csv", merged, MANUAL_FIELDS)
    print(
        json.dumps(
            {
                "status": "MERGED",
                "rows": len(merged),
                "completed_slot_labels": completed_slots(merged),
                "merged_path": str(merged_path),
                "wrote_main": bool(args.write_main),
            },
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    if args.mode == "prepare":
        prepare(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()
