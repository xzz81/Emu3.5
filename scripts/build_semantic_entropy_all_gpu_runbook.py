#!/usr/bin/env python3
"""Build the lb-gzs all-GPU execution runbook from current guarded evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validation-report",
        default="outputs/semantic_entropy_umm/extractor_validation/manual_annotation_validation_report.json",
    )
    parser.add_argument(
        "--robustness-preflight-report",
        default="outputs/semantic_entropy_umm/robustness_preflight/robustness_preflight_report.json",
    )
    parser.add_argument(
        "--option-launcher-dir",
        default="outputs/semantic_entropy_umm/robustness_option_order/launchers",
    )
    parser.add_argument(
        "--prompt-launcher-dir",
        default="outputs/semantic_entropy_umm/robustness_prompt_template/launchers",
    )
    parser.add_argument(
        "--return-scan-report",
        default="outputs/semantic_entropy_umm/manual_annotation_return_scan/manual_annotation_return_scan.json",
    )
    parser.add_argument(
        "--returned-gate-report",
        default="outputs/semantic_entropy_umm/returned_manual_gate/returned_manual_gate_report.json",
    )
    parser.add_argument(
        "--returned-watch-dir",
        default="outputs/semantic_entropy_umm/returned_manual_watch",
    )
    parser.add_argument(
        "--returned-gate-smoke-report",
        default="outputs/semantic_entropy_umm/returned_manual_gate_smoke/returned_manual_gate_smoke_report.json",
    )
    parser.add_argument(
        "--gpu-guard-smoke-report",
        default="outputs/semantic_entropy_umm/full_robustness_gpu_guard_smoke/full_robustness_gpu_guard_smoke_report.json",
    )
    parser.add_argument(
        "--full-robustness-dry-run-smoke-report",
        default="outputs/semantic_entropy_umm/full_robustness_dry_run_smoke/full_robustness_dry_run_smoke_report.json",
    )
    parser.add_argument("--out-dir", default="outputs/semantic_entropy_umm/all_gpu_execution_runbook")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def exists(path: Path) -> bool:
    return path.exists()


def list_launcher_paths(path: Path, pattern: str) -> list[str]:
    return [str(item) for item in sorted(path.glob(pattern))]


def fenced(lines: list[str]) -> list[str]:
    return ["```bash", *lines, "```"]


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    validation_report_path = Path(args.validation_report)
    preflight_report_path = Path(args.robustness_preflight_report)
    return_scan_report_path = Path(args.return_scan_report)
    returned_gate_report_path = Path(args.returned_gate_report)
    returned_watch_dir = Path(args.returned_watch_dir)
    returned_watch_report_path = returned_watch_dir / "returned_manual_watch_report.json"
    returned_watch_pid_path = returned_watch_dir / "watch_background.pid"
    returned_gate_smoke_report_path = Path(args.returned_gate_smoke_report)
    gpu_guard_smoke_report_path = Path(args.gpu_guard_smoke_report)
    dry_run_smoke_report_path = Path(args.full_robustness_dry_run_smoke_report)
    option_launcher_dir = Path(args.option_launcher_dir)
    prompt_launcher_dir = Path(args.prompt_launcher_dir)
    validation = read_json(validation_report_path)
    preflight = read_json(preflight_report_path)
    return_scan = read_json(return_scan_report_path)
    returned_gate = read_json(returned_gate_report_path)
    returned_watch = read_json(returned_watch_report_path)
    returned_gate_smoke = read_json(returned_gate_smoke_report_path)
    gpu_guard_smoke = read_json(gpu_guard_smoke_report_path)
    dry_run_smoke = read_json(dry_run_smoke_report_path)

    manual_status = validation.get("status", preflight.get("manual_validation_status", "NA"))
    completed_slots = validation.get("completed_slot_labels", preflight.get("manual_completed_slot_labels"))
    required_slots = validation.get("required_slot_labels", preflight.get("manual_required_slot_labels"))
    coverage_ok = preflight.get("launcher_gpu_coverage_ok") is True
    gpu_available = preflight.get("gpu_availability_ok_for_launch") is True
    detected_gpus = preflight.get("detected_gpu_indices", [])
    option_covered = preflight.get("option_covered_gpu_indices", [])
    prompt_covered = preflight.get("prompt_covered_gpu_indices", [])
    busy_gpus = preflight.get("busy_gpu_indices", [])

    option_launchers = list_launcher_paths(option_launcher_dir, "launch_option_order_seed_*.sh")
    prompt_launchers = list_launcher_paths(prompt_launcher_dir, "launch_prompt_template_*.sh")
    aggregate_launchers = {
        "option_order": str(option_launcher_dir / "launch_all_option_order_seeds.sh"),
        "prompt_template": str(prompt_launcher_dir / "launch_all_prompt_templates.sh"),
    }
    launchers_exist = all(exists(Path(path)) for path in aggregate_launchers.values()) and bool(option_launchers) and bool(
        prompt_launchers
    )

    launch_permitted = manual_status == "PASS" and coverage_ok and launchers_exist
    immediate_launch_ready = launch_permitted and gpu_available
    wait_launch_ready = launch_permitted and not gpu_available
    if manual_status != "PASS":
        status = "BLOCKED_MANUAL_VALIDATION"
    elif not launchers_exist:
        status = "BLOCKED_LAUNCHERS_MISSING"
    elif not coverage_ok:
        status = "BLOCKED_GPU_COVERAGE"
    elif gpu_available:
        status = "READY_TO_LAUNCH_NOW"
    else:
        status = "READY_TO_WAIT_FOR_GPUS"

    return {
        "status": status,
        "host": "lb-gzs",
        "remote_workspace": ".",
        "python": "./.venv-transformers/bin/python",
        "environment": {
            "HF_HOME": "$PWD/.cache/huggingface",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "PYTHONPATH": "$PWD",
        },
        "manual_validation_status": manual_status,
        "manual_completed_slot_labels": completed_slots,
        "manual_required_slot_labels": required_slots,
        "validation_report": str(validation_report_path),
        "preflight_report": str(preflight_report_path),
        "return_scan_report": str(return_scan_report_path),
        "returned_gate_report": str(returned_gate_report_path),
        "returned_watch_report": str(returned_watch_report_path),
        "returned_watch_pid_file": str(returned_watch_pid_path),
        "returned_gate_smoke_report": str(returned_gate_smoke_report_path),
        "gpu_guard_smoke_report": str(gpu_guard_smoke_report_path),
        "full_robustness_dry_run_smoke_report": str(dry_run_smoke_report_path),
        "preflight_status": preflight.get("status", "NA"),
        "return_scan_status": return_scan.get("status", "NA"),
        "return_scan_candidate_count": return_scan.get("candidate_count"),
        "return_scan_ready_to_import_count": return_scan.get("ready_to_import_count"),
        "returned_gate_status": returned_gate.get("status", "NA"),
        "returned_gate_wrote_target": returned_gate.get("wrote_target"),
        "returned_gate_post_manual_pipeline_ran": returned_gate.get("post_manual_pipeline_ran"),
        "returned_watch_status": returned_watch.get("status", "NA"),
        "returned_watch_gate_ran": returned_watch.get("gate_ran", "NA"),
        "returned_watch_write_requested": returned_watch.get("write_requested", "NA"),
        "returned_watch_run_full_robustness": returned_watch.get("run_full_robustness", "NA"),
        "returned_watch_attempt_count": len(returned_watch.get("attempts", [])),
        "returned_watch_started_at_utc": returned_watch.get("started_at_utc", ""),
        "returned_watch_deadline_utc": returned_watch.get("deadline_utc", ""),
        "returned_watch_poll_interval_seconds": returned_watch.get("poll_interval_seconds", ""),
        "returned_watch_pid": returned_watch_pid_path.read_text(encoding="utf-8").strip()
        if returned_watch_pid_path.exists()
        else "",
        "returned_gate_smoke_status": returned_gate_smoke.get("status", "NA"),
        "returned_gate_smoke_gate_status": returned_gate_smoke.get("gate_status", "NA"),
        "gpu_guard_smoke_status": gpu_guard_smoke.get("status", "NA"),
        "gpu_guard_smoke_gate_status": gpu_guard_smoke.get("gate_status", "NA"),
        "gpu_guard_smoke_worker_outputs_present": gpu_guard_smoke.get("worker_outputs_present"),
        "full_robustness_dry_run_smoke_status": dry_run_smoke.get("status", "NA"),
        "full_robustness_dry_run_smoke_gate_status": dry_run_smoke.get("gate_status", "NA"),
        "full_robustness_dry_run_status": dry_run_smoke.get("dry_run_status", "NA"),
        "full_robustness_dry_run_worker_outputs_present": dry_run_smoke.get("worker_outputs_present"),
        "detected_gpu_indices": detected_gpus,
        "option_covered_gpu_indices": option_covered,
        "prompt_covered_gpu_indices": prompt_covered,
        "launcher_gpu_coverage_ok": coverage_ok,
        "gpu_availability_ok_for_launch": gpu_available,
        "busy_gpu_indices": busy_gpus,
        "option_launchers": option_launchers,
        "prompt_launchers": prompt_launchers,
        "aggregate_launchers": aggregate_launchers,
        "launchers_exist": launchers_exist,
        "launch_permitted": launch_permitted,
        "immediate_launch_ready": immediate_launch_ready,
        "wait_launch_ready": wait_launch_ready,
        "claim_boundary": "runbook only; not full robustness evidence",
    }


def write_report(out_dir: Path, payload: dict[str, Any]) -> None:
    env_lines = [
        'cd "$(git rev-parse --show-toplevel)"',
        'export HF_HOME="$PWD/.cache/huggingface"',
        "export TRANSFORMERS_OFFLINE=1",
        "export HF_HUB_OFFLINE=1",
        'export PYTHONPATH="$PWD"',
    ]
    preflight_lines = [
        *env_lines,
        "./.venv-transformers/bin/python scripts/build_semantic_entropy_robustness_preflight.py "
        "--validation-report outputs/semantic_entropy_umm/extractor_validation/manual_annotation_validation_report.json "
        "--option-launcher-dir outputs/semantic_entropy_umm/robustness_option_order/launchers "
        "--prompt-launcher-dir outputs/semantic_entropy_umm/robustness_prompt_template/launchers "
        "--out-dir outputs/semantic_entropy_umm/robustness_preflight",
    ]
    scan_return_lines = [
        *env_lines,
        "./.venv-transformers/bin/python scripts/run_semantic_entropy_returned_manual_gate.py "
        "--py ./.venv-transformers/bin/python "
        "--scan-root outputs/semantic_entropy_umm "
        "--scan-root data/manual_annotation_returns",
    ]
    dry_run_candidate_lines = [
        *env_lines,
        "./.venv-transformers/bin/python scripts/run_semantic_entropy_returned_manual_gate.py "
        "--py ./.venv-transformers/bin/python "
        "--candidate data/manual_annotation_returns/returned_manual_annotations.csv",
    ]
    write_and_score_lines = [
        *env_lines,
        "./.venv-transformers/bin/python scripts/run_semantic_entropy_returned_manual_gate.py "
        "--py ./.venv-transformers/bin/python "
        "--candidate data/manual_annotation_returns/returned_manual_annotations.csv "
        "--write --run-post-manual-pipeline --run-full-robustness none",
    ]
    option_lines = [
        *env_lines,
        "./.venv-transformers/bin/python scripts/run_semantic_entropy_returned_manual_gate.py "
        "--py ./.venv-transformers/bin/python "
        "--candidate data/manual_annotation_returns/returned_manual_annotations.csv "
        "--write --run-post-manual-pipeline --run-full-robustness option-order "
        "--wait-for-gpus-seconds 3600 --wait-for-gpus-interval-seconds 60",
    ]
    prompt_lines = [
        *env_lines,
        "./.venv-transformers/bin/python scripts/run_semantic_entropy_returned_manual_gate.py "
        "--py ./.venv-transformers/bin/python "
        "--candidate data/manual_annotation_returns/returned_manual_annotations.csv "
        "--write --run-post-manual-pipeline --run-full-robustness prompt-template "
        "--wait-for-gpus-seconds 3600 --wait-for-gpus-interval-seconds 60",
    ]
    both_lines = [
        *env_lines,
        "./.venv-transformers/bin/python scripts/run_semantic_entropy_returned_manual_gate.py "
        "--py ./.venv-transformers/bin/python "
        "--candidate data/manual_annotation_returns/returned_manual_annotations.csv "
        "--write --run-post-manual-pipeline --run-full-robustness both "
        "--wait-for-gpus-seconds 3600 --wait-for-gpus-interval-seconds 60",
    ]
    watch_and_launch_lines = [
        *env_lines,
        "./.venv-transformers/bin/python scripts/watch_semantic_entropy_returned_manual_and_launch.py "
        "--py ./.venv-transformers/bin/python "
        "--scan-root outputs/semantic_entropy_umm/manual_annotation_returns "
        "--scan-root data/manual_annotation_returns "
        "--max-wait-seconds 21600 --poll-interval-seconds 300 "
        "--write --run-post-manual-pipeline --run-full-robustness both "
        "--wait-for-gpus-seconds 3600 --wait-for-gpus-interval-seconds 60",
    ]

    blockers: list[str] = []
    if payload["manual_validation_status"] != "PASS":
        blockers.append(
            "Manual validation is not PASS "
            f"({payload['manual_completed_slot_labels']}/{payload['manual_required_slot_labels']} slot labels complete)."
        )
    if not payload["launchers_exist"]:
        blockers.append("Full robustness aggregate or per-family launchers are missing.")
    if not payload["launcher_gpu_coverage_ok"]:
        blockers.append(
            "Launchers do not cover all detected GPUs "
            f"(detected={payload['detected_gpu_indices']}; "
            f"option={payload['option_covered_gpu_indices']}; "
            f"prompt={payload['prompt_covered_gpu_indices']})."
        )
    if payload["launch_permitted"] and not payload["gpu_availability_ok_for_launch"]:
        blockers.append(f"Immediate launch is waiting on busy GPUs: {payload['busy_gpu_indices']}.")

    lines = [
        "# lb-gzs All-GPU Semantic Entropy Runbook",
        "",
        f"Status: {payload['status']}",
        "",
        "This runbook records the exact guarded path for running full robustness on all detected `lb-gzs` GPUs. It does not start model inference and is not evidence that full robustness has completed.",
        "",
        "## Current Gate Snapshot",
        "",
        f"- Manual validation status: `{payload['manual_validation_status']}`",
        f"- Manual slot labels: `{payload['manual_completed_slot_labels']}/{payload['manual_required_slot_labels']}`",
        f"- Preflight status: `{payload['preflight_status']}`",
        f"- Detected GPUs: `{payload['detected_gpu_indices']}`",
        f"- Option-order covered GPUs: `{payload['option_covered_gpu_indices']}`",
        f"- Prompt-template covered GPUs: `{payload['prompt_covered_gpu_indices']}`",
        f"- Launcher GPU coverage OK: `{payload['launcher_gpu_coverage_ok']}`",
        f"- GPU availability OK now: `{payload['gpu_availability_ok_for_launch']}`",
        f"- Busy GPUs: `{payload['busy_gpu_indices']}`",
        f"- Launch permitted: `{payload['launch_permitted']}`",
        f"- Immediate launch ready: `{payload['immediate_launch_ready']}`",
        f"- Wait-loop launch ready: `{payload['wait_launch_ready']}`",
        f"- Returned annotation scan status: `{payload['return_scan_status']}`",
        f"- Ready returned annotation candidates: `{payload['return_scan_ready_to_import_count']}`",
        f"- Returned gate status: `{payload['returned_gate_status']}`",
        f"- Returned watch status: `{payload['returned_watch_status']}`",
        f"- Returned watch PID: `{payload['returned_watch_pid']}`",
        f"- Returned watch started/deadline UTC: `{payload['returned_watch_started_at_utc']}` / `{payload['returned_watch_deadline_utc']}`",
        f"- Returned watch poll interval seconds: `{payload['returned_watch_poll_interval_seconds']}`",
        f"- Returned watch attempts: `{payload['returned_watch_attempt_count']}`",
        f"- Returned watch write/full request: `{payload['returned_watch_write_requested']}` / `{payload['returned_watch_run_full_robustness']}`",
        f"- Returned watch gate ran: `{payload['returned_watch_gate_ran']}`",
        f"- Returned gate write/post-manual smoke: `{payload['returned_gate_smoke_status']}` / `{payload['returned_gate_smoke_gate_status']}`",
        f"- Full robustness GPU guard smoke: `{payload['gpu_guard_smoke_status']}` / `{payload['gpu_guard_smoke_gate_status']}`",
        f"- GPU guard smoke worker outputs present: `{payload['gpu_guard_smoke_worker_outputs_present']}`",
        f"- Full robustness positive dry-run smoke: `{payload['full_robustness_dry_run_smoke_status']}` / `{payload['full_robustness_dry_run_smoke_gate_status']}` / `{payload['full_robustness_dry_run_status']}`",
        f"- Full robustness dry-run worker outputs present: `{payload['full_robustness_dry_run_worker_outputs_present']}`",
        "",
        "## Active Blockers",
        "",
        *(f"- {item}" for item in blockers),
    ]
    if not blockers:
        lines.extend(["- None."])
    lines.extend(
        [
            "",
            "## Required Preflight",
            "",
            *fenced(preflight_lines),
            "",
            "Required PASS conditions before any full worker starts:",
            "",
            "- `manual_validation_status == PASS`",
            "- `launcher_gpu_coverage_ok == true`",
            "- `gpu_availability_ok_for_launch == true` for immediate launch, or use the wait-loop command below after manual PASS.",
            "",
            "## Returned Annotation Gate",
            "",
            "Scan common return locations without writing:",
            "",
            *fenced(scan_return_lines),
            "",
            "Dry-run a specific returned CSV:",
            "",
            *fenced(dry_run_candidate_lines),
            "",
            "Import a reviewed `READY_TO_WRITE` candidate and run validate/score/reports without full robustness:",
            "",
            *fenced(write_and_score_lines),
            "",
            "Watch return locations, then launch both full robustness families after a single ready CSV is found:",
            "",
            *fenced(watch_and_launch_lines),
            "",
            "## Full Robustness Commands",
            "",
            "These commands are only for a returned CSV that already dry-runs as `READY_TO_WRITE`. They still rerun the guarded preflight before any worker starts.",
            "",
            "Option-order only:",
            "",
            *fenced(option_lines),
            "",
            "Prompt-template only:",
            "",
            *fenced(prompt_lines),
            "",
            "Both robustness families:",
            "",
            *fenced(both_lines),
            "",
            "## Launcher Inventory",
            "",
            f"- Aggregate option-order launcher: `{payload['aggregate_launchers']['option_order']}`",
            f"- Aggregate prompt-template launcher: `{payload['aggregate_launchers']['prompt_template']}`",
            f"- Option-order seed launchers: `{payload['option_launchers']}`",
            f"- Prompt-template launchers: `{payload['prompt_launchers']}`",
            "",
            "## Claim Boundary",
            "",
            "`all_gpu_execution_runbook.md` is execution readiness documentation only. The robustness requirement is still incomplete until at least one full robustness family writes its completed audit outputs.",
        ]
    )
    (out_dir / "all_gpu_execution_runbook.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = build_payload(args)
    (out_dir / "all_gpu_execution_runbook.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(out_dir, payload)
    print(json.dumps({"status": payload["status"], "launch_permitted": payload["launch_permitted"]}, indent=2))


if __name__ == "__main__":
    main()
