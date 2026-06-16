#!/usr/bin/env python3
"""Smoke-test positive full-robustness path without launching workers."""

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
    parser.add_argument("--out-dir", default="outputs/semantic_entropy_umm/full_robustness_dry_run_smoke")
    parser.add_argument("--run-full-robustness", choices=["option-order", "prompt-template"], default="option-order")
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


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def run_command(cmd: list[str], env: dict[str, str]) -> dict[str, Any]:
    completed = subprocess.run(cmd, text=True, capture_output=True, check=False, env=env)
    return {
        "cmd": cmd,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-5000:],
        "stderr_tail": completed.stderr[-5000:],
    }


def completed_rows(manifest_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "annotation_id": row["annotation_id"],
            "annotator_id": "full_dry_run_smoke",
            "object_1": "cube",
            "color_1": "red",
            "object_2": "sphere",
            "color_2": "blue",
            "relation": "object_1_left_of_object_2",
            "background": "white",
            "notes": "",
        }
        for row in manifest_rows
    ]


def write_fake_nvidia_smi(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "$*" == *"--query-gpu=index,name,memory.total,memory.used,utilization.gpu"* ]]; then
  for i in 0 1 2 3 4 5 6 7; do
    printf "%s, NVIDIA A800 80GB PCIe, 81920, 0, 0\\n" "$i"
  done
  exit 0
fi
if [[ "$*" == *"--query-gpu=index"* ]]; then
  printf "0\\n1\\n2\\n3\\n4\\n5\\n6\\n7\\n"
  exit 0
fi
printf "fake nvidia-smi for semantic entropy full dry-run smoke\\n"
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def write_report(out_dir: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Full Robustness Dry-Run Smoke",
        "",
        f"Status: {payload['status']}",
        "",
        "This smoke test simulates completed manual annotations and available 8-GPU inventory. It verifies the pipeline reaches the selected full-robustness launcher in dry-run mode without starting model workers.",
        "",
        "## Summary",
        "",
        f"- Requested full robustness: `{payload.get('run_full_robustness')}`",
        f"- Gate status: `{payload.get('gate_status')}`",
        f"- Preflight GPU availability OK: `{payload.get('gpu_availability_ok_for_launch')}`",
        f"- Dry-run status: `{payload.get('dry_run_status')}`",
        f"- Dry-run family: `{payload.get('dry_run_family')}`",
        f"- Worker outputs present: `{payload.get('worker_outputs_present')}`",
        "",
        "## Command",
        "",
    ]
    command = payload.get("command", {})
    if command:
        lines.extend(["```bash", " ".join(command.get("cmd", [])), "```", ""])
        lines.append(f"- Return code: `{command.get('returncode')}`")
        lines.append("")
        if command.get("stdout_tail"):
            lines.extend(["stdout tail:", "", "```text", command["stdout_tail"], "```", ""])
        if command.get("stderr_tail"):
            lines.extend(["stderr tail:", "", "```text", command["stderr_tail"], "```", ""])
    (out_dir / "full_robustness_dry_run_smoke_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix="semantic_entropy_full_robustness_dry_run_"))
    payload: dict[str, Any]
    try:
        temp_validation_dir = temp_root / "extractor_validation"
        temp_return_dir = temp_root / "returns"
        temp_scan_dir = temp_root / "return_scan"
        temp_import_dir = temp_root / "manual_import"
        temp_gate_dir = temp_root / "gate"
        temp_outputs_dir = temp_root / "post_manual_outputs"
        fake_bin = temp_root / "fake_bin"
        shutil.copytree(Path(args.extractor_validation_dir), temp_validation_dir)
        manifest_rows = read_csv(temp_validation_dir / "annotation_manifest.csv")
        candidate_path = temp_return_dir / "manual_annotations.csv"
        write_csv(candidate_path, completed_rows(manifest_rows), MANUAL_FIELDS)
        write_fake_nvidia_smi(fake_bin / "nvidia-smi")
        env = {
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ.get('PATH', '')}",
            "NVIDIA_SMI": str(fake_bin / "nvidia-smi"),
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
            "FULL_ROBUSTNESS_DRY_RUN_DIR": str(temp_outputs_dir / "full_robustness_dry_run"),
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
        cmd = [
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
            "--run-post-manual-pipeline",
            "--run-full-robustness",
            args.run_full_robustness,
            "--dry-run-full-robustness",
            "--wait-for-gpus-seconds",
            "0",
        ]
        command = run_command(cmd, env=env)
        gate_report = read_json(temp_gate_dir / "returned_manual_gate_report.json")
        preflight_report = read_json(temp_outputs_dir / "robustness_preflight" / "robustness_preflight_report.json")
        dry_report = read_json(temp_outputs_dir / "full_robustness_dry_run" / "full_robustness_dry_run_report.json")
        worker_outputs_present = any(
            path.name.endswith("_audit.json")
            for root in [
                temp_outputs_dir / "robustness_option_order",
                temp_outputs_dir / "robustness_prompt_template",
            ]
            for path in root.rglob("*")
            if path.is_file()
        )
        expected_family = args.run_full_robustness
        status = (
            "PASS"
            if command["returncode"] == 0
            and gate_report.get("status") == "POST_MANUAL_PIPELINE_PASS"
            and gate_report.get("dry_run_full_robustness") is True
            and preflight_report.get("manual_validation_status") == "PASS"
            and preflight_report.get("launcher_gpu_coverage_ok") is True
            and preflight_report.get("gpu_availability_ok_for_launch") is True
            and preflight_report.get("busy_gpu_indices") == []
            and dry_report.get("status") == "DRY_RUN_READY_TO_LAUNCH"
            and dry_report.get("family") == expected_family
            and not worker_outputs_present
            else "FAIL"
        )
        payload = {
            "status": status,
            "temp_dir": str(temp_root) if args.keep_temp else "",
            "run_full_robustness": args.run_full_robustness,
            "gate_status": gate_report.get("status"),
            "dry_run_full_robustness": gate_report.get("dry_run_full_robustness"),
            "manual_validation_status": preflight_report.get("manual_validation_status"),
            "launcher_gpu_coverage_ok": preflight_report.get("launcher_gpu_coverage_ok"),
            "gpu_availability_ok_for_launch": preflight_report.get("gpu_availability_ok_for_launch"),
            "busy_gpu_indices": preflight_report.get("busy_gpu_indices"),
            "dry_run_status": dry_report.get("status"),
            "dry_run_family": dry_report.get("family"),
            "dry_run_launcher": dry_report.get("launcher"),
            "worker_outputs_present": worker_outputs_present,
            "command": command,
            "claim_boundary": "dry-run positive path only; no model worker launched",
        }
    finally:
        if not args.keep_temp:
            shutil.rmtree(temp_root, ignore_errors=True)
    (out_dir / "full_robustness_dry_run_smoke_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(out_dir, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "gate_status": payload.get("gate_status"),
                "dry_run_status": payload.get("dry_run_status"),
                "worker_outputs_present": payload.get("worker_outputs_present"),
            },
            indent=2,
        )
    )
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
