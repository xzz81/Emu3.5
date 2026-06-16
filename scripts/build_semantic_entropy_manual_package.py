#!/usr/bin/env python3
"""Build a portable human-annotation package for extractor validation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any


SLOTS = ["object_1", "color_1", "object_2", "color_2", "relation", "background"]
MANUAL_FIELDS = ["annotation_id", "annotator_id", *SLOTS, "notes"]
BLIND_FIELDS = ["annotation_id", "route", "sample_id", "image_file"]
PROMPT_CONTEXT_FIELDS = ["annotation_id", "route", "sample_id", "canonical_prompt", "image_file"]
VOCAB = {
    "object_1": ["cube", "sphere", "cone", "unknown"],
    "object_2": ["cube", "sphere", "cone", "unknown"],
    "color_1": ["red", "blue", "green", "yellow", "unknown"],
    "color_2": ["red", "blue", "green", "yellow", "unknown"],
    "relation": [
        "object_1_left_of_object_2",
        "object_1_right_of_object_2",
        "object_1_above_object_2",
        "object_1_below_object_2",
        "unknown",
    ],
    "background": ["white", "other", "unknown"],
}
PACKAGE_VALIDATOR = r'''#!/usr/bin/env python3
"""Validate a returned semantic-entropy manual-annotation CSV.

This script is intentionally standalone: it uses only Python's standard
library and the package-local manual_annotations_template.csv /
allowed_values.json files. It does not need the Emu3.5 repository.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


SLOTS = ["object_1", "color_1", "object_2", "color_2", "relation", "background"]
REQUIRED_FIELDS = ["annotation_id", "annotator_id", *SLOTS, "notes"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Returned manual annotation CSV to validate")
    parser.add_argument("--template", default="manual_annotations_template.csv")
    parser.add_argument("--allowed-values", default="allowed_values.json")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only")
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames or [], list(reader)


def main() -> None:
    args = parse_args()
    package_dir = Path(__file__).resolve().parent
    csv_path = Path(args.csv_path)
    if not csv_path.is_absolute():
        csv_path = Path.cwd() / csv_path
    template_path = Path(args.template)
    if not template_path.is_absolute():
        template_path = package_dir / template_path
    allowed_path = Path(args.allowed_values)
    if not allowed_path.is_absolute():
        allowed_path = package_dir / allowed_path

    fields, rows = read_csv(csv_path)
    _, template_rows = read_csv(template_path)
    allowed = json.loads(allowed_path.read_text(encoding="utf-8"))
    expected_ids = [row["annotation_id"] for row in template_rows]
    observed_ids = [row.get("annotation_id", "") for row in rows]
    duplicate_ids = sorted([item for item, count in Counter(observed_ids).items() if count > 1])
    missing_ids = sorted(set(expected_ids) - set(observed_ids))
    unknown_ids = sorted(set(observed_ids) - set(expected_ids))
    missing_fields = [field for field in REQUIRED_FIELDS if field not in fields]
    missing_cells = []
    illegal_values = []
    leakage_notes = []
    for row_idx, row in enumerate(rows, start=2):
        annotation_id = row.get("annotation_id", "")
        notes = row.get("notes", "")
        if any(token in notes.lower() for token in ["gold_", "concept_id", "canonical_prompt"]):
            leakage_notes.append({"row": row_idx, "annotation_id": annotation_id})
        for slot in SLOTS:
            value = (row.get(slot) or "").strip()
            if not value:
                missing_cells.append({"row": row_idx, "annotation_id": annotation_id, "slot": slot})
            elif value not in allowed[slot]:
                illegal_values.append(
                    {
                        "row": row_idx,
                        "annotation_id": annotation_id,
                        "slot": slot,
                        "value": value,
                        "allowed": allowed[slot],
                    }
                )
    completed_slot_labels = sum(1 for row in rows for slot in SLOTS if (row.get(slot) or "").strip())
    required_slot_labels = len(expected_ids) * len(SLOTS)
    status = (
        "PASS"
        if not missing_fields
        and len(rows) == len(expected_ids)
        and observed_ids == expected_ids
        and not duplicate_ids
        and not missing_ids
        and not unknown_ids
        and not missing_cells
        and not illegal_values
        and not leakage_notes
        else "FAIL"
    )
    payload = {
        "status": status,
        "csv_path": str(csv_path),
        "template": str(template_path),
        "rows": len(rows),
        "expected_rows": len(expected_ids),
        "completed_slot_labels": completed_slot_labels,
        "required_slot_labels": required_slot_labels,
        "missing_fields": missing_fields,
        "duplicate_annotation_ids": duplicate_ids,
        "missing_annotation_ids": missing_ids,
        "unknown_annotation_ids": unknown_ids,
        "missing_cells": missing_cells[:50],
        "missing_cells_count": len(missing_cells),
        "illegal_values": illegal_values[:50],
        "illegal_values_count": len(illegal_values),
        "leakage_notes": leakage_notes[:50],
        "leakage_notes_count": len(leakage_notes),
    }
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        print(f"Status: {status}")
        print(f"Rows: {len(rows)} / {len(expected_ids)}")
        print(f"Completed slot labels: {completed_slot_labels} / {required_slot_labels}")
        print(f"Missing fields: {len(missing_fields)}")
        print(f"Duplicate annotation_ids: {len(duplicate_ids)}")
        print(f"Missing annotation_ids: {len(missing_ids)}")
        print(f"Unknown annotation_ids: {len(unknown_ids)}")
        print(f"Missing cells: {len(missing_cells)}")
        print(f"Illegal values: {len(illegal_values)}")
        print(f"Leakage notes: {len(leakage_notes)}")
        if status != "PASS":
            print("\nRun with --json for details.")
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
'''
BATCH_MERGER = r'''#!/usr/bin/env python3
"""Merge package-local batch annotation CSVs and validate the result.

This standalone helper is for annotators who fill the six CSV files under
batch_manual_templates/. It preserves the template annotation_id order and then
runs validate_returned_annotations.py on the merged CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


SLOTS = ["object_1", "color_1", "object_2", "color_2", "relation", "background"]
REQUIRED_FIELDS = ["annotation_id", "annotator_id", *SLOTS, "notes"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", default="batch_manual_templates")
    parser.add_argument("--template", default="manual_annotations_template.csv")
    parser.add_argument("--output", default="manual_annotations_merged.csv")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only")
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames or [], list(reader)


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    package_dir = Path(__file__).resolve().parent
    batch_dir = Path(args.batch_dir)
    if not batch_dir.is_absolute():
        batch_dir = package_dir / batch_dir
    template_path = Path(args.template)
    if not template_path.is_absolute():
        template_path = package_dir / template_path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = package_dir / output_path

    _, template_rows = read_csv(template_path)
    expected_ids = [row["annotation_id"] for row in template_rows]
    batch_paths = sorted(batch_dir.glob("batch_*/batch_*_manual_annotations.csv"))
    merged_by_id: dict[str, dict[str, str]] = {}
    duplicate_ids = []
    missing_fields_by_file = {}
    for path in batch_paths:
        fields, rows = read_csv(path)
        missing_fields = [field for field in REQUIRED_FIELDS if field not in fields]
        if missing_fields:
            missing_fields_by_file[str(path)] = missing_fields
        for row in rows:
            annotation_id = row.get("annotation_id", "")
            if annotation_id in merged_by_id:
                duplicate_ids.append(annotation_id)
            merged_by_id[annotation_id] = {field: row.get(field, "") for field in REQUIRED_FIELDS}
    missing_ids = sorted(set(expected_ids) - set(merged_by_id))
    unknown_ids = sorted(set(merged_by_id) - set(expected_ids))
    duplicate_ids = sorted([item for item, count in Counter(duplicate_ids).items() if count >= 1])
    ordered_rows = [merged_by_id[annotation_id] for annotation_id in expected_ids if annotation_id in merged_by_id]
    write_csv(output_path, ordered_rows)

    validator = package_dir / "validate_returned_annotations.py"
    completed = subprocess.run(
        [sys.executable, str(validator), str(output_path), "--json"],
        text=True,
        capture_output=True,
        check=False,
    )
    try:
        validator_payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        validator_payload = {"status": "FAIL", "parse_error": completed.stdout}
    status = (
        "PASS"
        if completed.returncode == 0
        and validator_payload.get("status") == "PASS"
        and len(batch_paths) == 6
        and not duplicate_ids
        and not missing_ids
        and not unknown_ids
        and not missing_fields_by_file
        else "FAIL"
    )
    payload = {
        "status": status,
        "batch_dir": str(batch_dir),
        "batch_files": [str(path) for path in batch_paths],
        "batch_file_count": len(batch_paths),
        "output": str(output_path),
        "rows_written": len(ordered_rows),
        "expected_rows": len(expected_ids),
        "duplicate_annotation_ids": duplicate_ids,
        "missing_annotation_ids": missing_ids,
        "unknown_annotation_ids": unknown_ids,
        "missing_fields_by_file": missing_fields_by_file,
        "validator_returncode": completed.returncode,
        "validator_status": validator_payload.get("status"),
        "validator_completed_slot_labels": validator_payload.get("completed_slot_labels"),
        "validator_required_slot_labels": validator_payload.get("required_slot_labels"),
    }
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        print(f"Status: {status}")
        print(f"Batch files: {len(batch_paths)} / 6")
        print(f"Rows written: {len(ordered_rows)} / {len(expected_ids)}")
        print(f"Validator: {validator_payload.get('status')} ({validator_payload.get('completed_slot_labels')}/{validator_payload.get('required_slot_labels')})")
        if status != "PASS":
            print("\nRun with --json for details.")
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extractor-validation-dir", default="outputs/semantic_entropy_umm/extractor_validation")
    parser.add_argument("--manual-progress-dir", default="outputs/semantic_entropy_umm/manual_annotation_progress")
    parser.add_argument("--out-dir", default="outputs/semantic_entropy_umm/manual_annotation_package")
    parser.add_argument("--zip-name", default="manual_annotation_package.zip")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def build_package(args: argparse.Namespace) -> dict[str, Any]:
    validation_dir = Path(args.extractor_validation_dir)
    progress_dir = Path(args.manual_progress_dir)
    out_dir = Path(args.out_dir)
    reset_dir(out_dir)

    manifest_path = validation_dir / "annotation_manifest.csv"
    manual_path = validation_dir / "manual_annotations.csv"
    manifest = read_csv(manifest_path)
    manual = read_csv(manual_path)
    if not manifest:
        raise FileNotFoundError(manifest_path)
    if len(manual) != len(manifest):
        raise ValueError(f"manual row count mismatch: manifest={len(manifest)} manual={len(manual)}")

    images_dir = out_dir / "images"
    contact_sheets_dir = out_dir / "contact_sheets"
    batch_dir = out_dir / "batch_manual_templates"
    forms_dir = out_dir / "forms"
    copied_images = []
    missing_images = []
    blind_rows = []
    prompt_rows = []
    image_name_counts: dict[str, int] = {}

    for row in manifest:
        annotation_id = row["annotation_id"]
        src = Path(row["image_path"])
        suffix = src.suffix or ".png"
        image_name = f"{annotation_id}{suffix}"
        image_name_counts[image_name] = image_name_counts.get(image_name, 0) + 1
        if image_name_counts[image_name] > 1:
            image_name = f"{annotation_id}_{image_name_counts[image_name]}{suffix}"
        dst = images_dir / image_name
        rel_image = f"images/{image_name}"
        if src.exists():
            copy_if_exists(src, dst)
            copied_images.append({"annotation_id": annotation_id, "source": str(src), "package_path": rel_image})
        else:
            missing_images.append({"annotation_id": annotation_id, "source": str(src)})
        blind_rows.append(
            {
                "annotation_id": annotation_id,
                "route": row["route"],
                "sample_id": row["sample_id"],
                "image_file": rel_image,
            }
        )
        prompt_rows.append(
            {
                "annotation_id": annotation_id,
                "route": row["route"],
                "sample_id": row["sample_id"],
                "canonical_prompt": row["canonical_prompt"],
                "image_file": rel_image,
            }
        )

    write_csv(out_dir / "blind_manifest.csv", blind_rows, BLIND_FIELDS)
    write_csv(out_dir / "prompt_context_manifest.csv", prompt_rows, PROMPT_CONTEXT_FIELDS)
    write_csv(out_dir / "manual_annotations_template.csv", manual, MANUAL_FIELDS)
    (out_dir / "allowed_values.json").write_text(
        json.dumps(VOCAB, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validator_path = out_dir / "validate_returned_annotations.py"
    validator_path.write_text(PACKAGE_VALIDATOR + "\n", encoding="utf-8")
    validator_path.chmod(0o755)
    batch_merger_path = out_dir / "merge_batch_annotations.py"
    batch_merger_path.write_text(BATCH_MERGER + "\n", encoding="utf-8")
    batch_merger_path.chmod(0o755)

    for src in sorted((validation_dir / "contact_sheets").glob("*.png")):
        copy_if_exists(src, contact_sheets_dir / src.name)
    for src in sorted((validation_dir / "annotation_batches").glob("batch_*/batch_*_manual_annotations.csv")):
        copy_if_exists(src, batch_dir / src.parent.name / src.name)
    for src in [
        validation_dir / "annotation_form.html",
        validation_dir / "annotation_form_prompt_context.html",
        validation_dir / "manual_annotation_handoff.md",
        validation_dir / "manual_annotation_validation_report.md",
        progress_dir / "manual_annotation_progress.md",
        progress_dir / "manual_annotation_progress.csv",
    ]:
        copy_if_exists(src, forms_dir / src.name)

    leak_check = {
        "blind_manifest_fields": BLIND_FIELDS,
        "blind_manifest_excludes_gold_fields": True,
        "blind_manifest_excludes_concept_id": True,
        "prompt_context_manifest_excludes_gold_fields": True,
        "prompt_context_manifest_includes_prompt": True,
        "gold_manifest_included": False,
    }
    status = "READY_FOR_ANNOTATOR" if len(copied_images) == len(manifest) and not missing_images else "MISSING_IMAGES"
    package_manifest = {
        "status": status,
        "manifest_rows": len(manifest),
        "manual_rows": len(manual),
        "copied_images": len(copied_images),
        "missing_images": missing_images,
        "contact_sheets": len(list(contact_sheets_dir.glob("*.png"))),
        "batch_templates": len(list(batch_dir.glob("batch_*/batch_*_manual_annotations.csv"))),
        "helper_scripts": {
            "validate_returned_annotations": str(validator_path.relative_to(out_dir)),
            "merge_batch_annotations": str(batch_merger_path.relative_to(out_dir)),
        },
        "leak_check": leak_check,
        "source_files": {
            "annotation_manifest": str(manifest_path),
            "manual_annotations": str(manual_path),
            "manual_progress": str(progress_dir / "manual_annotation_progress.md"),
        },
    }
    (out_dir / "package_manifest.json").write_text(
        json.dumps(package_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    readme = [
        "# Semantic Entropy Manual Annotation Package",
        "",
        f"Status: {status}",
        "",
        "This package is for human extractor validation. It contains fixed sampled images and empty/manual annotation templates only. It does not contain gold slot labels.",
        "",
        "## Recommended Files",
        "",
        "- `forms/annotation_form.html`: blind image-only form with embedded thumbnails.",
        "- `forms/annotation_form_prompt_context.html`: prompt-context form for target binding when object_1/object_2 is ambiguous.",
        "- `manual_annotations_template.csv`: CSV with the required output fields.",
        "- `validate_returned_annotations.py`: standalone local checker for the completed CSV.",
        "- `merge_batch_annotations.py`: standalone merger/checker for the six batch CSV files.",
        "- `allowed_values.json`: machine-readable fixed vocabulary used by the checker.",
        "- `blind_manifest.csv`: no gold labels, no concept_id, no prompt.",
        "- `prompt_context_manifest.csv`: includes prompt context but no gold labels.",
        "- `contact_sheets/`: six image contact sheets.",
        "- `batch_manual_templates/`: six 10-row CSV templates.",
        "",
        "## Fixed Vocabulary",
        "",
        *[f"- `{slot}`: {', '.join(values)}" for slot, values in VOCAB.items()],
        "",
        "## Rules",
        "",
        "- Fill all six slot columns for each annotation row.",
        "- Use `unknown` when the image is unclear; do not guess.",
        "- Relation is from object_1 to object_2 under the target binding.",
        "- Return the completed CSV as `manual_annotations.csv`.",
        "",
        "## Package Checks",
        "",
        f"- Manifest rows: `{len(manifest)}`",
        f"- Copied images: `{len(copied_images)}`",
        f"- Missing images: `{len(missing_images)}`",
        f"- Contact sheets: `{package_manifest['contact_sheets']}`",
        f"- Batch templates: `{package_manifest['batch_templates']}`",
        "- Blind manifest excludes `concept_id` and all `gold_*` fields.",
        "- Prompt-context manifest excludes all `gold_*` fields.",
        "",
        "## After Annotation",
        "",
        "First dry-run the returned CSV through the safe importer:",
        "",
        "Annotators can first run this package-local check without the Emu3.5 repository:",
        "",
        "```bash",
        "python3 validate_returned_annotations.py manual_annotations.csv",
        "```",
        "",
        "If using the six batch templates, merge and validate them first:",
        "",
        "```bash",
        "python3 merge_batch_annotations.py",
        "```",
        "",
        "Then dry-run the returned CSV through the repository safe importer:",
        "",
        "```bash",
        "PY=./.venv-transformers/bin/python",
        "$PY scripts/import_semantic_entropy_manual_annotations.py --input data/manual_annotation_returns/returned_manual_annotations.csv",
        "```",
        "",
        "If the import report says `READY_TO_IMPORT`, write it into the main validation directory with an automatic backup:",
        "",
        "```bash",
        "$PY scripts/import_semantic_entropy_manual_annotations.py --input data/manual_annotation_returns/returned_manual_annotations.csv --write",
        "```",
        "",
        "Then run:",
        "",
        "```bash",
        "$PY scripts/build_semantic_entropy_manual_progress.py",
        "$PY scripts/build_semantic_entropy_extractor_validation.py --mode validate-manual",
        "```",
    ]
    (out_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    quickstart_zh = [
        "# 语义熵人工标注快速说明",
        "",
        f"状态：{status}",
        "",
        "这个包用于完成 extractor validation 的人工标注门禁。包内没有 gold 标签；请只根据图片和允许的词表填写标注。",
        "",
        "## 推荐工作流",
        "",
        "1. 打开 `forms/annotation_form.html` 进行盲标。",
        "2. 如果 object_1 / object_2 绑定不清楚，再打开 `forms/annotation_form_prompt_context.html` 查看 prompt context。",
        "3. 填写 `manual_annotations_template.csv`，或填写 `batch_manual_templates/` 下的 6 个 10 行 batch CSV。",
        "4. 如果使用 batch CSV，先运行 `python3 merge_batch_annotations.py` 合并并自检。",
        "5. 如果直接填写整表，回传前运行 `python3 validate_returned_annotations.py manual_annotations.csv` 做本地自检。",
        "6. 返回一个完成的 CSV，文件名建议为 `manual_annotations.csv`。",
        "",
        "## 必须填写的列",
        "",
        "`annotation_id, annotator_id, object_1, color_1, object_2, color_2, relation, background, notes`",
        "",
        "每一行都必须填写 6 个 slot：`object_1, color_1, object_2, color_2, relation, background`。",
        "",
        "## 固定词表",
        "",
        *[f"- `{slot}`: {', '.join(values)}" for slot, values in VOCAB.items()],
        "",
        "## 标注规则",
        "",
        "- 只能使用固定词表中的值；不要写同义词、中文、缩写或额外解释到 slot 列。",
        "- 看不清、无法确定、图像缺失或绑定不明确时，slot 填 `unknown`，不要猜。",
        "- `relation` 必须表示 object_1 到 object_2 的关系，不是任意两个物体的关系。",
        "- 如果颜色和物体绑定不确定，在 `notes` 中说明；slot 仍按固定词表填写。",
        "- 不要把 prompt、concept_id、gold label 或推测过程复制到 `notes`。",
        "",
        "## 交付前自检",
        "",
        f"- 行数应为 `{len(manifest)}`。",
        "- 每行 6 个 slot 都应非空。",
        f"- 总 slot 标签数应为 `{len(manifest) * len(SLOTS)}`。",
        "- `annotation_id` 不要改动、删除或重复。",
        "- 如果分 batch 标注，6 个 batch 合并后应覆盖全部 60 个 annotation_id。",
        "- 如果分 batch 标注，运行 `python3 merge_batch_annotations.py` 应显示 `Status: PASS`。",
        "- 在 package 根目录运行 `python3 validate_returned_annotations.py manual_annotations.csv` 应显示 `Status: PASS`。",
        "",
        "## 回收后的机器校验",
        "",
        "先 dry-run，不要直接覆盖主表：",
        "",
        "```bash",
        "PY=./.venv-transformers/bin/python",
        "$PY scripts/import_semantic_entropy_manual_annotations.py --input data/manual_annotation_returns/returned_manual_annotations.csv",
        "```",
        "",
        "只有报告显示 `READY_TO_IMPORT` 后，才写入主目录：",
        "",
        "```bash",
        "$PY scripts/import_semantic_entropy_manual_annotations.py --input data/manual_annotation_returns/returned_manual_annotations.csv --write",
        "$PY scripts/build_semantic_entropy_extractor_validation.py --mode validate-manual",
        "```",
        "",
        "校验通过前，不允许运行 extractor score 或 full robustness。",
    ]
    (out_dir / "ANNOTATOR_QUICKSTART_zh.md").write_text(
        "\n".join(quickstart_zh) + "\n",
        encoding="utf-8",
    )

    zip_path = out_dir.parent / args.zip_name
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_handle:
        for path in sorted(out_dir.rglob("*")):
            if path.is_file():
                zip_handle.write(path, path.relative_to(out_dir.parent))
    package_manifest["zip_path"] = str(zip_path)
    package_manifest["zip_sha256"] = sha256_file(zip_path)
    (out_dir / "package_manifest.json").write_text(
        json.dumps(package_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return package_manifest


def main() -> None:
    args = parse_args()
    manifest = build_package(args)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "manifest_rows": manifest["manifest_rows"],
                "copied_images": manifest["copied_images"],
                "zip_path": manifest["zip_path"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
