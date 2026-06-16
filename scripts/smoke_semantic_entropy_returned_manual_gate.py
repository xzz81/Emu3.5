#!/usr/bin/env python3
"""Smoke-test returned manual gate write behavior in a temporary validation dir."""

from __future__ import annotations

import argparse
import csv
import json
import os
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
    parser.add_argument("--py", default=sys.executable)
    parser.add_argument("--extractor-validation-dir", default="outputs/semantic_entropy_umm/extractor_validation")
    parser.add_argument("--out-dir", default="outputs/semantic_entropy_umm/returned_manual_gate_smoke")
    parser.add_argument(
        "--run-post-manual-pipeline",
        action="store_true",
        help="Also smoke the explicit post-manual pipeline branch with RUN_FULL_ROBUSTNESS=none.",
    )
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_command(cmd: list[str], env: dict[str, str] | None = None) -> dict[str, Any]:
    completed = subprocess.run(cmd, text=True, capture_output=True, check=False, env=env)
    return {
        "cmd": cmd,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_report(out_dir: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Returned Manual Gate Smoke",
        "",
        f"Status: {payload['status']}",
        "",
        "This smoke test verifies the returned-manual gate can write a ready candidate into a temporary extractor-validation directory. It does not modify the real manual annotation table and does not run full robustness.",
        "",
        "## Summary",
        "",
        f"- Temporary directory: `{payload.get('temp_dir', '')}`",
        f"- Gate status: `{payload.get('gate_status')}`",
        f"- Wrote target: `{payload.get('wrote_target')}`",
        f"- Post-manual pipeline ran: `{payload.get('post_manual_pipeline_ran')}`",
        f"- Post-manual pipeline requested: `{payload.get('run_post_manual_pipeline')}`",
        f"- Target completed labels: `{payload.get('target_completed_slot_labels')}` / `{payload.get('target_required_slot_labels')}`",
        f"- Backup files: `{payload.get('backup_files')}`",
        "",
        "## Command",
        "",
    ]
    command = payload.get("command", {})
    if command:
        lines.extend(["```bash", " ".join(command.get("cmd", [])), "```", ""])
        lines.extend([f"- Return code: `{command.get('returncode')}`", ""])
        if command.get("stdout_tail"):
            lines.extend(["stdout tail:", "", "```text", command["stdout_tail"], "```", ""])
        if command.get("stderr_tail"):
            lines.extend(["stderr tail:", "", "```text", command["stderr_tail"], "```", ""])
    (out_dir / "returned_manual_gate_smoke_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def completed_rows(manifest_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    for manifest_row in manifest_rows:
        rows.append(
            {
                "annotation_id": manifest_row["annotation_id"],
                "annotator_id": "gate_smoke",
                "object_1": "cube",
                "color_1": "red",
                "object_2": "sphere",
                "color_2": "blue",
                "relation": "object_1_left_of_object_2",
                "background": "white",
                "notes": "",
            }
        )
    return rows


def count_completed(path: Path) -> tuple[int, int]:
    rows = read_csv(path)
    completed = sum(1 for row in rows for slot in SLOTS if (row.get(slot) or "").strip())
    return completed, len(rows) * len(SLOTS)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_validation_dir = Path(args.extractor_validation_dir)
    temp_root = Path(tempfile.mkdtemp(prefix="semantic_entropy_gate_smoke_"))
    temp_validation_dir = temp_root / "extractor_validation"
    temp_scan_dir = temp_root / "return_scan"
    temp_import_dir = temp_root / "manual_import"
    temp_gate_dir = temp_root / "gate"
    temp_return_dir = temp_root / "returns"
    temp_outputs_dir = temp_root / "post_manual_outputs"
    command_result: dict[str, Any] = {}
    payload: dict[str, Any]
    try:
        shutil.copytree(source_validation_dir, temp_validation_dir)
        manifest_rows = read_csv(temp_validation_dir / "annotation_manifest.csv")
        candidate_path = temp_return_dir / "manual_annotations.csv"
        write_csv(candidate_path, completed_rows(manifest_rows), MANUAL_FIELDS)
        gate_cmd = [
            args.py,
            "scripts/run_semantic_entropy_returned_manual_gate.py",
            "--py",
            args.py,
            "--candidate",
            str(candidate_path),
            "--extractor-validation-dir",
            str(temp_validation_dir),
            "--return-scan-dir",
            str(temp_scan_dir),
            "--import-dir",
            str(temp_import_dir),
            "--out-dir",
            str(temp_gate_dir),
            "--write",
        ]
        if args.run_post_manual_pipeline:
            gate_cmd.extend(["--run-post-manual-pipeline", "--run-full-robustness", "none"])
        env = None
        if args.run_post_manual_pipeline:
            env = {
                **os.environ,
                "STRICT_DIR": "outputs/semantic_entropy_umm/strict_compare",
                "EXTRACTOR_VALIDATION_DIR": str(temp_validation_dir),
                "MANUAL_PROGRESS_DIR": str(temp_outputs_dir / "manual_annotation_progress"),
                "MANUAL_PACKAGE_DIR": "outputs/semantic_entropy_umm/manual_annotation_package",
                "MANUAL_PACKAGE_QA_DIR": "outputs/semantic_entropy_umm/manual_annotation_package_qa",
                "MANUAL_IMPORT_DIR": str(temp_import_dir),
                "POST_MANUAL_SMOKE_DIR": "outputs/semantic_entropy_umm/post_manual_smoke",
                "PILOT_DIR": "outputs/semantic_entropy_umm/pilot",
                "QUADRANT_DIR": "outputs/semantic_entropy_umm/quadrant_analysis",
                "BOOTSTRAP_DIR": "outputs/semantic_entropy_umm/bootstrap_stability",
                "SAMPLE_SIZE_DIR": "outputs/semantic_entropy_umm/sample_size_sensitivity",
                "OPTION_ORDER_SMOKE_DIR": "outputs/semantic_entropy_umm/robustness_option_order_smoke",
                "OPTION_ORDER_FULL_DIR": str(temp_outputs_dir / "robustness_option_order"),
                "PROMPT_TEMPLATE_SMOKE_DIR": "outputs/semantic_entropy_umm/robustness_prompt_template_smoke",
                "PROMPT_TEMPLATE_FULL_DIR": str(temp_outputs_dir / "robustness_prompt_template"),
                "ROBUSTNESS_PREFLIGHT_DIR": str(temp_outputs_dir / "robustness_preflight"),
                "ALL_GPU_RUNBOOK_DIR": str(temp_outputs_dir / "all_gpu_execution_runbook"),
                "OBJECT_BINDING_DIR": "outputs/semantic_entropy_umm/object_binding_sanity",
                "IMAGE_DUPLICATE_DIR": "outputs/semantic_entropy_umm/image_duplicate_mode_collapse",
                "CONCEPT_DIFFICULTY_DIR": "outputs/semantic_entropy_umm/concept_difficulty_split",
                "RANDOM_CONTROL_DIR": "outputs/semantic_entropy_umm/random_concept_control",
                "SLOT_ABLATION_DIR": "outputs/semantic_entropy_umm/slot_ablation",
                "CLAIM_BOUNDARY_DIR": str(temp_outputs_dir / "claim_boundary_audit"),
                "STAGE_REPORT_AUDIT_DIR": "outputs/semantic_entropy_umm/stage_report_audit",
                "REQUIREMENT_AUDIT_DIR": str(temp_outputs_dir / "requirement_audit"),
                "EXECUTION_STATUS_DIR": str(temp_outputs_dir / "execution_status"),
                "PAPER_DIR": str(temp_outputs_dir / "paper_tables_preliminary"),
            }
        command_result = run_command(gate_cmd, env=env)
        gate_report = read_json(temp_gate_dir / "returned_manual_gate_report.json")
        import_report = read_json(temp_import_dir / "manual_annotation_import_report.json")
        target_completed, target_required = count_completed(temp_validation_dir / "manual_annotations.csv")
        backup_files = sorted(str(path) for path in (temp_validation_dir / "manual_annotation_backups").glob("*.csv"))
        metrics_exists = (temp_validation_dir / "extractor_validation_metrics.csv").exists()
        joined_exists = (temp_validation_dir / "extractor_joined_annotations.csv").exists()
        temp_execution_exists = (temp_outputs_dir / "execution_status" / "execution_status_report.md").exists()
        temp_paper_exists = (temp_outputs_dir / "paper_tables_preliminary" / "paper_tables_preliminary.md").exists()
        status = (
            "PASS"
            if command_result.get("returncode") == 0
            and gate_report.get("status")
            == (
                "POST_MANUAL_PIPELINE_PASS"
                if args.run_post_manual_pipeline
                else "WRITTEN_READY_FOR_POST_MANUAL_PIPELINE"
            )
            and import_report.get("wrote_target") is True
            and gate_report.get("post_manual_pipeline_ran") is bool(args.run_post_manual_pipeline)
            and gate_report.get("run_full_robustness") == "none"
            and target_completed == target_required == 360
            and backup_files
            and (not args.run_post_manual_pipeline or (metrics_exists and joined_exists and temp_execution_exists and temp_paper_exists))
            else "FAIL"
        )
        payload = {
            "status": status,
            "temp_dir": str(temp_root) if args.keep_temp else "",
            "gate_status": gate_report.get("status"),
            "wrote_target": import_report.get("wrote_target"),
            "post_manual_pipeline_ran": gate_report.get("post_manual_pipeline_ran"),
            "run_post_manual_pipeline": bool(args.run_post_manual_pipeline),
            "run_full_robustness": gate_report.get("run_full_robustness"),
            "target_completed_slot_labels": target_completed,
            "target_required_slot_labels": target_required,
            "metrics_exists": metrics_exists,
            "joined_annotations_exists": joined_exists,
            "temp_execution_status_exists": temp_execution_exists,
            "temp_paper_tables_exists": temp_paper_exists,
            "backup_files": backup_files,
            "command": command_result,
            "claim_boundary": "write-branch smoke only; not human validation or full robustness evidence",
        }
    finally:
        if not args.keep_temp:
            shutil.rmtree(temp_root, ignore_errors=True)
    (out_dir / "returned_manual_gate_smoke_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(out_dir, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "gate_status": payload.get("gate_status"),
                "wrote_target": payload.get("wrote_target"),
                "post_manual_pipeline_ran": payload.get("post_manual_pipeline_ran"),
            },
            indent=2,
        )
    )
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
