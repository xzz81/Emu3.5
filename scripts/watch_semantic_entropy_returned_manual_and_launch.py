#!/usr/bin/env python3
"""Poll for returned manual annotations, then hand off to the guarded launcher."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--py", default=sys.executable)
    parser.add_argument("--scan-root", action="append", default=[])
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--extractor-validation-dir", default="outputs/semantic_entropy_umm/extractor_validation")
    parser.add_argument("--return-scan-dir", default="outputs/semantic_entropy_umm/manual_annotation_return_scan")
    parser.add_argument("--gate-out-dir", default="outputs/semantic_entropy_umm/returned_manual_gate")
    parser.add_argument("--out-dir", default="outputs/semantic_entropy_umm/returned_manual_watch")
    parser.add_argument("--max-wait-seconds", type=int, default=0)
    parser.add_argument("--poll-interval-seconds", type=int, default=300)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--run-post-manual-pipeline", action="store_true")
    parser.add_argument(
        "--run-full-robustness",
        choices=["none", "option-order", "prompt-template", "both"],
        default="none",
    )
    parser.add_argument("--wait-for-gpus-seconds", type=int, default=3600)
    parser.add_argument("--wait-for-gpus-interval-seconds", type=int, default=60)
    parser.add_argument("--dry-run-full-robustness", action="store_true")
    return parser.parse_args()


def run_command(cmd: list[str]) -> dict[str, Any]:
    completed = subprocess.run(cmd, text=True, capture_output=True, check=False)
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
        "# Returned Manual Watch And Launch",
        "",
        f"Status: {payload['status']}",
        "",
        "This watcher polls for a single ready returned manual annotation CSV. It only writes annotations or launches robustness when the corresponding explicit flags are present.",
        "",
        "## Summary",
        "",
        f"- Attempts: `{len(payload.get('attempts', []))}`",
        f"- PID: `{payload.get('pid', '')}`",
        f"- Started at UTC: `{payload.get('started_at_utc', '')}`",
        f"- Deadline UTC: `{payload.get('deadline_utc', '')}`",
        f"- Poll interval seconds: `{payload.get('poll_interval_seconds', '')}`",
        f"- Selected candidate: `{payload.get('selected_candidate', '')}`",
        f"- Write requested: `{payload.get('write_requested')}`",
        f"- Post-manual pipeline requested: `{payload.get('run_post_manual_pipeline')}`",
        f"- Requested full robustness: `{payload.get('run_full_robustness')}`",
        f"- Gate ran: `{payload.get('gate_ran')}`",
        f"- Gate status: `{payload.get('gate_status', '')}`",
        "",
        "## Attempts",
        "",
    ]
    for attempt in payload.get("attempts", []):
        scan = attempt.get("scan", {})
        lines.extend(
            [
                f"- attempt `{attempt.get('attempt')}`: status `{scan.get('status')}`, candidates `{scan.get('candidate_count')}`, ready `{scan.get('ready_to_import_count')}`",
            ]
        )
    if payload.get("gate_command"):
        lines.extend(
            [
                "",
                "## Gate Command",
                "",
                "```bash",
                " ".join(payload["gate_command"]),
                "```",
            ]
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "`returned_manual_watch` is an execution handoff helper. It is not manual validation evidence and is not robustness evidence unless the guarded downstream reports complete successfully.",
        ]
    )
    (out_dir / "returned_manual_watch_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def persist_report(out_dir: Path, payload: dict[str, Any]) -> None:
    (out_dir / "returned_manual_watch_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(out_dir, payload)


def scan_once(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    scan_cmd = [
        args.py,
        "scripts/find_semantic_entropy_returned_annotations.py",
        "--extractor-validation-dir",
        args.extractor_validation_dir,
        "--out-dir",
        args.return_scan_dir,
    ]
    for root in args.scan_root:
        scan_cmd.extend(["--scan-root", root])
    for candidate in args.candidate:
        scan_cmd.extend(["--candidate", candidate])
    result = run_command(scan_cmd)
    return result, read_json(Path(args.return_scan_dir) / "manual_annotation_return_scan.json")


def main() -> None:
    args = parse_args()
    if args.run_full_robustness != "none" and not args.run_post_manual_pipeline:
        raise SystemExit("--run-full-robustness requires --run-post-manual-pipeline")
    if args.run_post_manual_pipeline and not args.write:
        raise SystemExit("--run-post-manual-pipeline requires --write")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc)
    deadline = time.monotonic() + max(0, args.max_wait_seconds)
    deadline_utc = (
        datetime.fromtimestamp(started_at.timestamp() + max(0, args.max_wait_seconds), tz=timezone.utc)
        if args.max_wait_seconds > 0
        else None
    )
    attempts: list[dict[str, Any]] = []
    status = "WAITING_FOR_READY_CANDIDATE"
    selected_candidate = ""
    gate_result: dict[str, Any] | None = None
    gate_payload: dict[str, Any] = {}
    gate_cmd: list[str] = []
    attempt_idx = 0

    while True:
        attempt_idx += 1
        scan_result, scan_payload = scan_once(args)
        attempts.append({"attempt": attempt_idx, "command": scan_result, "scan": scan_payload})
        if scan_result["returncode"] != 0:
            status = "SCAN_FAILED"
            break

        ready_candidates = [row for row in scan_payload.get("candidates", []) if row.get("ready_to_import")]
        if len(ready_candidates) > 1:
            status = "MULTIPLE_READY_CANDIDATES"
            break
        if len(ready_candidates) == 1:
            selected_candidate = ready_candidates[0]["path"]
            if not args.write:
                status = "READY_CANDIDATE_FOUND_NO_WRITE"
                break
            gate_cmd = [
                args.py,
                "scripts/run_semantic_entropy_returned_manual_gate.py",
                "--py",
                args.py,
                "--candidate",
                selected_candidate,
                "--extractor-validation-dir",
                args.extractor_validation_dir,
                "--return-scan-dir",
                args.return_scan_dir,
                "--out-dir",
                args.gate_out_dir,
                "--write",
            ]
            if args.run_post_manual_pipeline:
                gate_cmd.append("--run-post-manual-pipeline")
                gate_cmd.extend(["--run-full-robustness", args.run_full_robustness])
                gate_cmd.extend(["--wait-for-gpus-seconds", str(args.wait_for_gpus_seconds)])
                gate_cmd.extend(["--wait-for-gpus-interval-seconds", str(args.wait_for_gpus_interval_seconds)])
                if args.dry_run_full_robustness:
                    gate_cmd.append("--dry-run-full-robustness")
            gate_result = run_command(gate_cmd)
            gate_payload = read_json(Path(args.gate_out_dir) / "returned_manual_gate_report.json")
            status = gate_payload.get("status", "GATE_REPORT_MISSING")
            if gate_result["returncode"] != 0 and status == "GATE_REPORT_MISSING":
                status = "GATE_FAILED"
            break

        if args.max_wait_seconds <= 0 or time.monotonic() >= deadline:
            status = "WAITING_FOR_READY_CANDIDATE"
            break
        persist_report(
            out_dir,
            {
                "status": "WAITING_FOR_READY_CANDIDATE",
                "pid": os.getpid(),
                "started_at_utc": started_at.isoformat(),
                "deadline_utc": "" if deadline_utc is None else deadline_utc.isoformat(),
                "max_wait_seconds": args.max_wait_seconds,
                "poll_interval_seconds": args.poll_interval_seconds,
                "selected_candidate": selected_candidate,
                "write_requested": bool(args.write),
                "run_post_manual_pipeline": bool(args.run_post_manual_pipeline),
                "run_full_robustness": args.run_full_robustness,
                "dry_run_full_robustness": bool(args.dry_run_full_robustness),
                "wait_for_gpus_seconds": args.wait_for_gpus_seconds,
                "gate_ran": False,
                "gate_status": "",
                "gate_returncode": None,
                "gate_command": [],
                "attempts": attempts,
                "claim_boundary": "guarded poll-and-handoff only; not manual validation or robustness evidence by itself",
            },
        )
        time.sleep(max(1, args.poll_interval_seconds))

    payload = {
        "status": status,
        "pid": os.getpid(),
        "started_at_utc": started_at.isoformat(),
        "deadline_utc": "" if deadline_utc is None else deadline_utc.isoformat(),
        "max_wait_seconds": args.max_wait_seconds,
        "poll_interval_seconds": args.poll_interval_seconds,
        "selected_candidate": selected_candidate,
        "write_requested": bool(args.write),
        "run_post_manual_pipeline": bool(args.run_post_manual_pipeline),
        "run_full_robustness": args.run_full_robustness,
        "dry_run_full_robustness": bool(args.dry_run_full_robustness),
        "wait_for_gpus_seconds": args.wait_for_gpus_seconds,
        "gate_ran": gate_result is not None,
        "gate_status": gate_payload.get("status", ""),
        "gate_returncode": None if gate_result is None else gate_result["returncode"],
        "gate_command": gate_cmd,
        "attempts": attempts,
        "claim_boundary": "guarded poll-and-handoff only; not manual validation or robustness evidence by itself",
    }
    persist_report(out_dir, payload)
    print(
        json.dumps(
            {
                "status": status,
                "selected_candidate": selected_candidate,
                "gate_ran": gate_result is not None,
                "gate_status": gate_payload.get("status", ""),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
