#!/usr/bin/env python3
"""Smoke-test post-manual import/validate/score mechanics in an isolated copy."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


SLOTS = ["object_1", "color_1", "object_2", "color_2", "relation", "background"]
MANUAL_FIELDS = ["annotation_id", "annotator_id", *SLOTS, "notes"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extractor-validation-dir", default="outputs/semantic_entropy_umm/extractor_validation")
    parser.add_argument("--strict-dir", default="outputs/semantic_entropy_umm/strict_compare")
    parser.add_argument("--out-dir", default="outputs/semantic_entropy_umm/post_manual_smoke")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_command(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def synthetic_manual_rows(manifest: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    for item in manifest:
        row = {
            "annotation_id": item["annotation_id"],
            "annotator_id": "synthetic_smoke_gold_copy",
            "notes": "SMOKE_ONLY_NOT_EVIDENCE",
        }
        for slot in SLOTS:
            row[slot] = item[f"gold_{slot}"]
        rows.append(row)
    return rows


def md_table(rows: list[dict[str, Any]], fields: list[str]) -> list[str]:
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        safe = {field: str(row.get(field, "")).replace("|", "\\|").replace("\n", " ") for field in fields}
        lines.append("| " + " | ".join(safe[field] for field in fields) + " |")
    return lines


def write_report(out_dir: Path, payload: dict[str, Any]) -> None:
    command_rows = [
        {
            "step": idx + 1,
            "command": " ".join(item["command"]),
            "returncode": item["returncode"],
        }
        for idx, item in enumerate(payload["commands"])
    ]
    lines = [
        "# Post-Manual Pipeline Smoke Report",
        "",
        f"Status: {payload['status']}",
        "",
        "This smoke test uses synthetic labels copied from the fixed manifest only to test import/validate/score mechanics in an isolated temporary extractor-validation directory. It is not human validation evidence and must not be cited as extractor accuracy.",
        "",
        "## Summary",
        "",
        f"- Manifest rows: `{payload['manifest_rows']}`",
        f"- Synthetic completed slot labels: `{payload['synthetic_completed_slot_labels']}`",
        f"- Import status: `{payload['import_status']}`",
        f"- Validate status: `{payload['validate_status']}`",
        f"- Joined annotation rows: `{payload['joined_annotation_rows']}`",
        f"- Metric rows: `{payload['metric_rows']}`",
        f"- Wrote real manual annotations: `False`",
        "",
        "## Commands",
        "",
        *md_table(command_rows, ["step", "returncode", "command"]),
        "",
        "## Claim Boundary",
        "",
        "Allowed: toolchain mechanics are smoke-tested in isolation.",
        "",
        "Not allowed: human extractor validation, extractor accuracy, or final paper-readiness claims.",
    ]
    (out_dir / "post_manual_smoke_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_validation_dir = Path(args.extractor_validation_dir)
    strict_dir = Path(args.strict_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_csv(source_validation_dir / "annotation_manifest.csv")
    temp_root_obj = tempfile.TemporaryDirectory(prefix="semantic_entropy_post_manual_smoke_")
    temp_root = Path(temp_root_obj.name)
    temp_validation_dir = temp_root / "extractor_validation"
    shutil.copytree(source_validation_dir, temp_validation_dir)
    returned_csv = temp_root / "synthetic_returned_manual_annotations.csv"
    synthetic_rows = synthetic_manual_rows(manifest)
    write_csv(returned_csv, synthetic_rows, MANUAL_FIELDS)

    command_results = []
    import_dir = temp_root / "manual_annotation_import"
    backup_dir = temp_root / "manual_annotation_backups"
    command_results.append(
        run_command(
            [
                args.python,
                "scripts/import_semantic_entropy_manual_annotations.py",
                "--input",
                str(returned_csv),
                "--extractor-validation-dir",
                str(temp_validation_dir),
                "--out-dir",
                str(import_dir),
            ]
        )
    )
    command_results.append(
        run_command(
            [
                args.python,
                "scripts/import_semantic_entropy_manual_annotations.py",
                "--input",
                str(returned_csv),
                "--extractor-validation-dir",
                str(temp_validation_dir),
                "--out-dir",
                str(import_dir),
                "--backup-dir",
                str(backup_dir),
                "--write",
            ]
        )
    )
    command_results.append(
        run_command(
            [
                args.python,
                "scripts/build_semantic_entropy_extractor_validation.py",
                "--mode",
                "validate-manual",
                "--out-dir",
                str(temp_validation_dir),
            ]
        )
    )
    command_results.append(
        run_command(
            [
                args.python,
                "scripts/build_semantic_entropy_extractor_validation.py",
                "--mode",
                "score",
                "--strict-dir",
                str(strict_dir),
                "--out-dir",
                str(temp_validation_dir),
            ]
        )
    )

    import_report = json.loads((import_dir / "manual_annotation_import_report.json").read_text(encoding="utf-8"))
    validate_report = json.loads((temp_validation_dir / "manual_annotation_validation_report.json").read_text(encoding="utf-8"))
    joined_rows = read_csv(temp_validation_dir / "extractor_joined_annotations.csv")
    metric_rows = read_csv(temp_validation_dir / "extractor_validation_metrics.csv")
    status = (
        "PASS_SMOKE_ONLY"
        if all(item["returncode"] == 0 for item in command_results)
        and import_report.get("status") == "READY_TO_IMPORT"
        and validate_report.get("status") == "PASS"
        and len(joined_rows) == len(manifest)
        and len(metric_rows) > 0
        else "FAIL"
    )
    payload = {
        "status": status,
        "claim_boundary": "SMOKE_ONLY_NOT_EVIDENCE",
        "manifest_rows": len(manifest),
        "synthetic_completed_slot_labels": len(manifest) * len(SLOTS),
        "import_status": import_report.get("status"),
        "validate_status": validate_report.get("status"),
        "joined_annotation_rows": len(joined_rows),
        "metric_rows": len(metric_rows),
        "temp_root": str(temp_root) if args.keep_temp else "",
        "commands": command_results,
    }
    (out_dir / "post_manual_smoke_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(out_dir, payload)
    print(
        json.dumps(
            {
                "status": status,
                "import_status": payload["import_status"],
                "validate_status": payload["validate_status"],
                "joined_annotation_rows": payload["joined_annotation_rows"],
                "metric_rows": payload["metric_rows"],
            },
            indent=2,
        )
    )
    if args.keep_temp:
        temp_root_obj._finalizer.detach()
    if status != "PASS_SMOKE_ONLY":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
