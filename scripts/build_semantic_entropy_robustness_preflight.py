#!/usr/bin/env python3
"""Audit full-robustness launchers without starting model inference."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-report", default="outputs/semantic_entropy_umm/extractor_validation/manual_annotation_validation_report.json")
    parser.add_argument("--option-launcher-dir", default="outputs/semantic_entropy_umm/robustness_option_order/launchers")
    parser.add_argument("--prompt-launcher-dir", default="outputs/semantic_entropy_umm/robustness_prompt_template/launchers")
    parser.add_argument("--out-dir", default="outputs/semantic_entropy_umm/robustness_preflight")
    parser.add_argument("--run-guard-smoke", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=20)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def detect_gpu_indices() -> list[int]:
    nvidia_smi = os.environ.get("NVIDIA_SMI", "nvidia-smi")
    try:
        completed = subprocess.run(
            [nvidia_smi, "--query-gpu=index", "--format=csv,noheader"],
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        completed = None
    if not completed or completed.returncode != 0:
        return []
    indices = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if line:
            indices.append(int(line.split(",")[0].strip()))
    return sorted(indices)


def gpu_inventory(memory_busy_threshold_mib: int = 4096) -> list[dict[str, Any]]:
    nvidia_smi = os.environ.get("NVIDIA_SMI", "nvidia-smi")
    try:
        completed = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        completed = None
    if not completed or completed.returncode != 0:
        return []
    rows = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        index = int(parts[0])
        used_mib = int(float(parts[3]))
        rows.append(
            {
                "index": index,
                "name": parts[1],
                "memory_total_mib": int(float(parts[2])),
                "memory_used_mib": used_mib,
                "utilization_gpu_percent": int(float(parts[4])),
                "busy": used_mib > memory_busy_threshold_mib,
            }
        )
    return rows


def covered_gpu_indices(launchers: list[dict[str, Any]]) -> list[int]:
    covered: set[int] = set()
    for launcher in launchers:
        for pair in launcher.get("gpu_pairs", []):
            for item in pair.split(","):
                item = item.strip()
                if item:
                    covered.add(int(item))
    return sorted(covered)


def launcher_audit(path: Path) -> dict[str, Any]:
    text = read_text(path)
    gpu_pairs = sorted(set(re.findall(r"CUDA_VISIBLE_DEVICES=([0-9,]+)", text)))
    worker_modes = len(re.findall(r"--mode\s+worker", text))
    return {
        "path": str(path),
        "exists": path.exists(),
        "has_manual_validation_guard": "manual_annotation_validation_report.json" in text and "status') != 'PASS'" in text,
        "has_force_bypass_warning": "FORCE_ROBUSTNESS=1 bypassed manual validation guard" in text,
        "worker_command_count": worker_modes,
        "gpu_pairs": gpu_pairs,
    }


def run_guard_smoke(path: Path, timeout_seconds: int) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            ["bash", str(path)],
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
            env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"},
        )
        stdout = completed.stdout
        stderr = completed.stderr
        return {
            "path": str(path),
            "returncode": completed.returncode,
            "stdout_tail": stdout[-1000:],
            "stderr_tail": stderr[-1000:],
            "blocked_by_manual_guard": completed.returncode != 0
            and "manual annotation validation is not PASS" in (stdout + stderr),
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "path": str(path),
            "returncode": None,
            "stdout_tail": (exc.stdout or "")[-1000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-1000:] if isinstance(exc.stderr, str) else "",
            "blocked_by_manual_guard": False,
            "timed_out": True,
        }


def md_table(rows: list[dict[str, Any]], fields: list[str]) -> list[str]:
    if not rows:
        return ["(none)"]
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        safe = {field: str(row.get(field, "")).replace("|", "\\|").replace("\n", " ") for field in fields}
        lines.append("| " + " | ".join(safe[field] for field in fields) + " |")
    return lines


def write_report(out_dir: Path, payload: dict[str, Any]) -> None:
    launcher_rows = []
    for family in ("option_order", "prompt_template"):
        for item in payload[f"{family}_launchers"]:
            launcher_rows.append(
                {
                    "family": family,
                    "path": item["path"],
                    "guard": item["has_manual_validation_guard"],
                    "bypass_warning": item["has_force_bypass_warning"],
                    "worker_commands": item["worker_command_count"],
                    "gpu_pairs": item["gpu_pairs"],
                }
            )
    smoke_rows = payload.get("guard_smoke", [])
    lines = [
        "# Robustness Preflight Audit",
        "",
        f"Status: {payload['status']}",
        "",
        "This audit checks full-robustness launcher readiness and manual-validation guards without starting model inference.",
        "",
        "## Summary",
        "",
        f"- Manual validation status: `{payload['manual_validation_status']}`",
        f"- Option-order launcher status: `{payload['option_manifest_status']}`",
        f"- Prompt-template launcher status: `{payload['prompt_manifest_status']}`",
        f"- Guard smoke executed: `{bool(smoke_rows)}`",
        f"- Detected GPU indices: `{payload['detected_gpu_indices']}`",
        f"- Launcher GPU coverage OK: `{payload['launcher_gpu_coverage_ok']}`",
        f"- GPU availability OK for launch: `{payload['gpu_availability_ok_for_launch']}`",
        f"- Busy GPU indices: `{payload['busy_gpu_indices']}`",
        f"- Claim boundary: `{payload['claim_boundary']}`",
        "",
        "## GPU Availability",
        "",
        *md_table(
            payload.get("gpu_inventory", []),
            ["index", "name", "memory_total_mib", "memory_used_mib", "utilization_gpu_percent", "busy"],
        ),
        "",
        "## Launcher Checks",
        "",
        *md_table(launcher_rows, ["family", "path", "guard", "bypass_warning", "worker_commands", "gpu_pairs"]),
        "",
        "## Guard Smoke",
        "",
        *md_table(smoke_rows, ["path", "returncode", "blocked_by_manual_guard", "timed_out"]),
        "",
        "## Next Gate",
        "",
        "Do not run full robustness until manual annotation validation is PASS. After PASS, run either the option-order or prompt-template full launcher.",
    ]
    (out_dir / "robustness_preflight_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    validation_report = read_json(Path(args.validation_report))
    option_dir = Path(args.option_launcher_dir)
    prompt_dir = Path(args.prompt_launcher_dir)
    option_manifest = read_json(option_dir / "option_order_full_launcher_manifest.json")
    prompt_manifest = read_json(prompt_dir / "prompt_template_full_launcher_manifest.json")
    option_launchers = [launcher_audit(path) for path in sorted(option_dir.glob("launch_option_order_seed_*.sh"))]
    prompt_launchers = [launcher_audit(path) for path in sorted(prompt_dir.glob("launch_prompt_template_*.sh"))]
    all_launchers = option_launchers + prompt_launchers
    inventory = gpu_inventory()
    detected_gpus = detect_gpu_indices()
    if not detected_gpus and inventory:
        detected_gpus = sorted(row["index"] for row in inventory)
    busy_gpus = sorted(row["index"] for row in inventory if row.get("busy"))
    gpu_availability_ok = bool(detected_gpus) and not busy_gpus
    option_covered = covered_gpu_indices(option_launchers)
    prompt_covered = covered_gpu_indices(prompt_launchers)
    expected_covered = detected_gpus or sorted(set(option_covered + prompt_covered))
    gpu_coverage_ok = (
        (not detected_gpus or option_covered == detected_gpus)
        and (not detected_gpus or prompt_covered == detected_gpus)
    )
    guard_smoke = []
    manual_status = validation_report.get("status", "NA")
    if args.run_guard_smoke and manual_status != "PASS":
        guard_smoke = [
            run_guard_smoke(option_dir / "launch_all_option_order_seeds.sh", args.timeout_seconds),
            run_guard_smoke(prompt_dir / "launch_all_prompt_templates.sh", args.timeout_seconds),
        ]
    ready = (
        option_manifest.get("status") == "READY_BUT_GUARDED"
        and prompt_manifest.get("status") == "READY_BUT_GUARDED"
        and len(option_launchers) == 3
        and len(prompt_launchers) == 2
        and all(item["exists"] and item["has_manual_validation_guard"] and item["has_force_bypass_warning"] for item in all_launchers)
        and gpu_coverage_ok
    )
    smoke_ok = not guard_smoke or all(item["blocked_by_manual_guard"] and not item["timed_out"] for item in guard_smoke)
    if ready and manual_status != "PASS" and smoke_ok:
        status = "PASS_GUARDED"
    elif ready and manual_status == "PASS":
        status = "READY_TO_RUN_AFTER_MANUAL_PASS"
    else:
        status = "FAIL"
    payload = {
        "status": status,
        "manual_validation_status": manual_status,
        "manual_completed_slot_labels": validation_report.get("completed_slot_labels"),
        "manual_required_slot_labels": validation_report.get("required_slot_labels"),
        "option_manifest_status": option_manifest.get("status", "NA"),
        "prompt_manifest_status": prompt_manifest.get("status", "NA"),
        "option_gpu_pairs": option_manifest.get("gpu_pairs", []),
        "prompt_gpu_pairs": prompt_manifest.get("gpu_pairs", []),
        "detected_gpu_indices": detected_gpus,
        "gpu_inventory": inventory,
        "busy_gpu_indices": busy_gpus,
        "gpu_availability_ok_for_launch": gpu_availability_ok,
        "expected_gpu_coverage": expected_covered,
        "option_covered_gpu_indices": option_covered,
        "prompt_covered_gpu_indices": prompt_covered,
        "launcher_gpu_coverage_ok": gpu_coverage_ok,
        "option_order_launchers": option_launchers,
        "prompt_template_launchers": prompt_launchers,
        "guard_smoke": guard_smoke,
        "claim_boundary": "guard readiness only; not full robustness evidence",
    }
    (out_dir / "robustness_preflight_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(out_dir, payload)
    print(json.dumps({"status": status, "manual_validation_status": manual_status, "guard_smoke": bool(guard_smoke)}, indent=2))
    if status == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
