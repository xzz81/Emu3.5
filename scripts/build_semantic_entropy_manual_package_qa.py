#!/usr/bin/env python3
"""QA the portable manual annotation package independently of its builder."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any


SLOTS = ["object_1", "color_1", "object_2", "color_2", "relation", "background"]
BLIND_FIELDS = ["annotation_id", "route", "sample_id", "image_file"]
PROMPT_CONTEXT_FIELDS = ["annotation_id", "route", "sample_id", "canonical_prompt", "image_file"]
MANUAL_FIELDS = ["annotation_id", "annotator_id", *SLOTS, "notes"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", default="outputs/semantic_entropy_umm/manual_annotation_package")
    parser.add_argument("--zip-path", default="outputs/semantic_entropy_umm/manual_annotation_package.zip")
    parser.add_argument("--out-dir", default="outputs/semantic_entropy_umm/manual_annotation_package_qa")
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        return [], []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames or [], list(reader)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def add_check(rows: list[dict[str, Any]], check: str, status: str, observed: str, expected: str) -> None:
    rows.append({"check": check, "status": status, "observed": observed, "expected": expected})


def md_table(rows: list[dict[str, Any]], fields: list[str]) -> list[str]:
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        safe = {field: str(row.get(field, "")).replace("|", "\\|").replace("\n", " ") for field in fields}
        lines.append("| " + " | ".join(safe[field] for field in fields) + " |")
    return lines


def zip_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "entry_count": 0, "testzip": "missing"}
    with zipfile.ZipFile(path) as handle:
        bad = handle.testzip()
        names = handle.namelist()
    return {"exists": True, "entry_count": len(names), "testzip": bad or "", "names": names}


def main() -> None:
    args = parse_args()
    package_dir = Path(args.package_dir)
    zip_path = Path(args.zip_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    package_manifest = read_json(package_dir / "package_manifest.json")
    quickstart_zh = package_dir / "ANNOTATOR_QUICKSTART_zh.md"
    quickstart_text = quickstart_zh.read_text(encoding="utf-8") if quickstart_zh.exists() else ""
    allowed_values = read_json(package_dir / "allowed_values.json")
    validator_path = package_dir / "validate_returned_annotations.py"
    batch_merger_path = package_dir / "merge_batch_annotations.py"
    blind_fields, blind_rows = read_csv(package_dir / "blind_manifest.csv")
    prompt_fields, prompt_rows = read_csv(package_dir / "prompt_context_manifest.csv")
    manual_fields, manual_rows = read_csv(package_dir / "manual_annotations_template.csv")
    image_paths = sorted((package_dir / "images").glob("*"))
    contact_sheets = sorted((package_dir / "contact_sheets").glob("*.png"))
    batch_paths = sorted((package_dir / "batch_manual_templates").glob("batch_*/batch_*_manual_annotations.csv"))
    checks: list[dict[str, Any]] = []

    add_check(
        checks,
        "package manifest READY_FOR_ANNOTATOR",
        "PASS" if package_manifest.get("status") == "READY_FOR_ANNOTATOR" else "FAIL",
        str(package_manifest.get("status", "NA")),
        "READY_FOR_ANNOTATOR",
    )
    add_check(
        checks,
        "blind manifest fields",
        "PASS" if blind_fields == BLIND_FIELDS else "FAIL",
        str(blind_fields),
        str(BLIND_FIELDS),
    )
    forbidden_blind_fields = [field for field in blind_fields if field.startswith("gold_") or field == "concept_id" or field == "canonical_prompt"]
    add_check(
        checks,
        "blind manifest no gold/concept/prompt fields",
        "PASS" if not forbidden_blind_fields else "FAIL",
        str(forbidden_blind_fields),
        "[]",
    )
    forbidden_prompt_fields = [field for field in prompt_fields if field.startswith("gold_") or field == "concept_id"]
    add_check(
        checks,
        "prompt-context manifest no gold/concept fields",
        "PASS" if prompt_fields == PROMPT_CONTEXT_FIELDS and not forbidden_prompt_fields else "FAIL",
        f"fields={prompt_fields}; forbidden={forbidden_prompt_fields}",
        str(PROMPT_CONTEXT_FIELDS),
    )
    add_check(
        checks,
        "manual template fields",
        "PASS" if manual_fields == MANUAL_FIELDS else "FAIL",
        str(manual_fields),
        str(MANUAL_FIELDS),
    )
    manual_slots_blank = all(not row.get(slot, "").strip() for row in manual_rows for slot in SLOTS)
    add_check(
        checks,
        "manual template slot labels blank",
        "PASS" if manual_slots_blank else "FAIL",
        f"rows={len(manual_rows)}; all_slots_blank={manual_slots_blank}",
        "60 rows with blank slot labels",
    )
    add_check(checks, "blind manifest rows", "PASS" if len(blind_rows) == 60 else "FAIL", str(len(blind_rows)), "60")
    add_check(checks, "prompt manifest rows", "PASS" if len(prompt_rows) == 60 else "FAIL", str(len(prompt_rows)), "60")
    add_check(checks, "manual template rows", "PASS" if len(manual_rows) == 60 else "FAIL", str(len(manual_rows)), "60")
    blind_ids = [row.get("annotation_id", "") for row in blind_rows]
    prompt_ids = [row.get("annotation_id", "") for row in prompt_rows]
    manual_ids = [row.get("annotation_id", "") for row in manual_rows]
    duplicate_ids = [item for item, count in Counter(blind_ids).items() if count > 1]
    add_check(
        checks,
        "annotation ids aligned and unique",
        "PASS" if blind_ids == prompt_ids == manual_ids and not duplicate_ids else "FAIL",
        f"aligned={blind_ids == prompt_ids == manual_ids}; duplicates={duplicate_ids}",
        "blind/prompt/manual ids identical and unique",
    )
    missing_image_refs = [
        row.get("image_file", "")
        for row in blind_rows
        if not (package_dir / row.get("image_file", "")).exists()
    ]
    add_check(
        checks,
        "blind manifest image refs exist",
        "PASS" if not missing_image_refs else "FAIL",
        f"missing={len(missing_image_refs)}",
        "0 missing image refs",
    )
    add_check(checks, "copied image files", "PASS" if len(image_paths) == 60 else "FAIL", str(len(image_paths)), "60")
    add_check(checks, "contact sheets", "PASS" if len(contact_sheets) == 6 else "FAIL", str(len(contact_sheets)), "6")
    batch_ids: list[str] = []
    for path in batch_paths:
        _, rows = read_csv(path)
        batch_ids.extend(row.get("annotation_id", "") for row in rows)
    add_check(checks, "batch template files", "PASS" if len(batch_paths) == 6 else "FAIL", str(len(batch_paths)), "6")
    add_check(
        checks,
        "batch templates cover all ids once",
        "PASS" if sorted(batch_ids) == sorted(blind_ids) and len(batch_ids) == len(set(batch_ids)) == 60 else "FAIL",
        f"batch_ids={len(batch_ids)}; unique={len(set(batch_ids))}; covers={sorted(batch_ids) == sorted(blind_ids)}",
        "60 unique ids matching blind manifest",
    )
    zip_info = zip_summary(zip_path)
    add_check(
        checks,
        "zip integrity",
        "PASS" if zip_info.get("exists") and zip_info.get("testzip") == "" else "FAIL",
        f"exists={zip_info.get('exists')}; testzip={zip_info.get('testzip')}; entries={zip_info.get('entry_count')}",
        "zip exists; testzip empty",
    )
    actual_zip_sha = sha256_file(zip_path) if zip_path.exists() else ""
    add_check(
        checks,
        "zip sha256 matches manifest",
        "PASS" if actual_zip_sha and actual_zip_sha == package_manifest.get("zip_sha256") else "FAIL",
        f"actual={actual_zip_sha}; manifest={package_manifest.get('zip_sha256', '')}",
        "actual sha256 equals package_manifest.zip_sha256",
    )
    zip_names = set(zip_info.get("names", []))
    expected_zip_refs = {f"manual_annotation_package/{row['image_file']}" for row in blind_rows}
    add_check(
        checks,
        "zip contains all blind image refs",
        "PASS" if expected_zip_refs.issubset(zip_names) else "FAIL",
        f"missing={len(expected_zip_refs - zip_names)}",
        "0 missing image refs in zip",
    )
    quickstart_required_terms = ["固定词表", "unknown", "manual_annotations.csv", "READY_TO_IMPORT"]
    missing_quickstart_terms = [term for term in quickstart_required_terms if term not in quickstart_text]
    add_check(
        checks,
        "Chinese annotator quickstart exists",
        "PASS" if quickstart_zh.exists() and not missing_quickstart_terms else "FAIL",
        f"exists={quickstart_zh.exists()}; missing_terms={missing_quickstart_terms}",
        "exists with fixed vocabulary, unknown, return CSV, and importer readiness terms",
    )
    add_check(
        checks,
        "zip contains Chinese annotator quickstart",
        "PASS" if "manual_annotation_package/ANNOTATOR_QUICKSTART_zh.md" in zip_names else "FAIL",
        str("manual_annotation_package/ANNOTATOR_QUICKSTART_zh.md" in zip_names),
        "True",
    )
    add_check(
        checks,
        "allowed-values metadata exists",
        "PASS" if allowed_values and all(slot in allowed_values for slot in SLOTS) else "FAIL",
        f"slots={sorted(allowed_values)}",
        str(SLOTS),
    )
    add_check(
        checks,
        "package-local returned CSV validator exists",
        "PASS" if validator_path.exists() else "FAIL",
        str(validator_path.exists()),
        "True",
    )
    add_check(
        checks,
        "package-local batch merger exists",
        "PASS" if batch_merger_path.exists() else "FAIL",
        str(batch_merger_path.exists()),
        "True",
    )
    add_check(
        checks,
        "zip contains validator assets",
        "PASS"
        if {
            "manual_annotation_package/allowed_values.json",
            "manual_annotation_package/validate_returned_annotations.py",
            "manual_annotation_package/merge_batch_annotations.py",
        }.issubset(zip_names)
        else "FAIL",
        str(
            {
                "manual_annotation_package/allowed_values.json",
                "manual_annotation_package/validate_returned_annotations.py",
                "manual_annotation_package/merge_batch_annotations.py",
            }.issubset(zip_names)
        ),
        "True",
    )
    validator_smoke = subprocess.run(
        [sys.executable, str(validator_path), str(package_dir / "manual_annotations_template.csv"), "--json"],
        text=True,
        capture_output=True,
        check=False,
    )
    try:
        validator_payload = json.loads(validator_smoke.stdout)
    except json.JSONDecodeError:
        validator_payload = {}
    add_check(
        checks,
        "validator rejects blank template",
        "PASS"
        if validator_smoke.returncode != 0
        and validator_payload.get("status") == "FAIL"
        and validator_payload.get("completed_slot_labels") == 0
        and validator_payload.get("required_slot_labels") == 360
        else "FAIL",
        f"returncode={validator_smoke.returncode}; status={validator_payload.get('status')}; completed={validator_payload.get('completed_slot_labels')}/{validator_payload.get('required_slot_labels')}",
        "nonzero returncode, FAIL, completed=0/360",
    )

    completed_rows_by_id = {
        row["annotation_id"]: {
            "annotation_id": row["annotation_id"],
            "annotator_id": "qa_smoke",
            "object_1": "cube",
            "color_1": "red",
            "object_2": "sphere",
            "color_2": "blue",
            "relation": "object_1_left_of_object_2",
            "background": "white",
            "notes": "",
        }
        for row in manual_rows
    }
    smoke_batch_dir = out_dir / "merge_batch_smoke" / "batch_manual_templates"
    if smoke_batch_dir.exists():
        for old_path in sorted(smoke_batch_dir.rglob("*"), reverse=True):
            if old_path.is_file():
                old_path.unlink()
            elif old_path.is_dir():
                old_path.rmdir()
    smoke_batch_dir.mkdir(parents=True, exist_ok=True)
    for path in batch_paths:
        _, rows = read_csv(path)
        smoke_rows = [completed_rows_by_id[row["annotation_id"]] for row in rows]
        write_csv(
            smoke_batch_dir / path.parent.name / path.name,
            smoke_rows,
            MANUAL_FIELDS,
        )
    merge_smoke_output = out_dir / "merge_batch_smoke" / "manual_annotations_merged.csv"
    merge_smoke = subprocess.run(
        [
            sys.executable,
            str(batch_merger_path),
            "--batch-dir",
            str(smoke_batch_dir.resolve()),
            "--output",
            str(merge_smoke_output.resolve()),
            "--json",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    try:
        merge_payload = json.loads(merge_smoke.stdout)
    except json.JSONDecodeError:
        merge_payload = {}
    add_check(
        checks,
        "batch merger accepts completed batch templates",
        "PASS"
        if merge_smoke.returncode == 0
        and merge_payload.get("status") == "PASS"
        and merge_payload.get("rows_written") == 60
        and merge_payload.get("validator_status") == "PASS"
        and merge_payload.get("validator_completed_slot_labels") == 360
        else "FAIL",
        f"returncode={merge_smoke.returncode}; status={merge_payload.get('status')}; rows={merge_payload.get('rows_written')}; validator={merge_payload.get('validator_status')} {merge_payload.get('validator_completed_slot_labels')}/{merge_payload.get('validator_required_slot_labels')}",
        "returncode=0, PASS, rows=60, validator PASS 360/360",
    )

    status = "PASS" if all(row["status"] == "PASS" for row in checks) else "FAIL"
    payload = {
        "status": status,
        "package_dir": str(package_dir),
        "zip_path": str(zip_path),
        "checks": checks,
        "claim_boundary": "package QA only; not manual validation evidence",
    }
    (out_dir / "manual_annotation_package_qa.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(out_dir / "manual_annotation_package_qa.csv", checks, ["check", "status", "observed", "expected"])
    lines = [
        "# Manual Annotation Package QA",
        "",
        f"Status: {status}",
        "",
        "This QA checks the portable annotation package for file completeness, manifest alignment, zip integrity, and blind-manifest leakage risk. It does not validate human labels.",
        "",
        *md_table(checks, ["check", "status", "observed", "expected"]),
    ]
    (out_dir / "manual_annotation_package_qa.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "checks": len(checks)}, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
