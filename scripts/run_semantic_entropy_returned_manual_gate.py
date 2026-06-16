#!/usr/bin/env python3
"""Guarded handoff from returned manual annotations to post-manual execution."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--py", default=sys.executable)
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--scan-root", action="append", default=[])
    parser.add_argument("--extractor-validation-dir", default="outputs/semantic_entropy_umm/extractor_validation")
    parser.add_argument(
        "--backup-dir",
        default="",
        help="Manual annotation backup directory. Defaults to <extractor-validation-dir>/manual_annotation_backups.",
    )
    parser.add_argument("--return-scan-dir", default="outputs/semantic_entropy_umm/manual_annotation_return_scan")
    parser.add_argument("--import-dir", default="outputs/semantic_entropy_umm/manual_annotation_import")
    parser.add_argument("--out-dir", default="outputs/semantic_entropy_umm/returned_manual_gate")
    parser.add_argument("--write", action="store_true", help="Import the single READY_TO_IMPORT candidate into manual_annotations.csv")
    parser.add_argument(
        "--run-post-manual-pipeline",
        action="store_true",
        help="After --write, run the post-manual pipeline. Full robustness remains controlled by --run-full-robustness.",
    )
    parser.add_argument(
        "--run-full-robustness",
        choices=["none", "option-order", "prompt-template", "both"],
        default="none",
        help="Forwarded to run_semantic_entropy_post_manual_pipeline.sh after --write.",
    )
    parser.add_argument("--wait-for-gpus-seconds", type=int, default=0)
    parser.add_argument("--wait-for-gpus-interval-seconds", type=int, default=60)
    parser.add_argument(
        "--dry-run-full-robustness",
        action="store_true",
        help="When running the post-manual pipeline with full robustness requested, stop after guarded preflight and write a dry-run launcher report.",
    )
    return parser.parse_args()


def run_command(cmd: list[str], env: dict[str, str] | None = None) -> dict[str, Any]:
    completed = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
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
        "# Returned Manual Annotation Gate",
        "",
        f"Status: {payload['status']}",
        "",
        "This gate scans returned manual annotation CSVs, dry-runs import readiness, and optionally writes/imports one ready candidate. It does not run the post-manual pipeline unless `--run-post-manual-pipeline` is explicitly set, and it never launches full robustness unless that flag plus `--run-full-robustness` are both explicitly set.",
        "",
        "## Summary",
        "",
        f"- Candidate count: `{payload.get('candidate_count')}`",
        f"- Ready-to-import count: `{payload.get('ready_to_import_count')}`",
        f"- Selected candidate: `{payload.get('selected_candidate', '')}`",
        f"- Write requested: `{payload.get('write_requested')}`",
        f"- Wrote target: `{payload.get('wrote_target')}`",
        f"- Post-manual pipeline requested: `{payload.get('run_post_manual_pipeline')}`",
        f"- Requested full robustness: `{payload.get('run_full_robustness')}`",
        f"- Post-manual pipeline ran: `{payload.get('post_manual_pipeline_ran')}`",
        "",
        "## Command Results",
        "",
    ]
    for item in payload.get("commands", []):
        lines.extend(
            [
                f"### returncode `{item['returncode']}`",
                "",
                "```bash",
                " ".join(item["cmd"]),
                "```",
                "",
            ]
        )
        if item.get("stdout_tail"):
            lines.extend(["stdout tail:", "", "```text", item["stdout_tail"], "```", ""])
        if item.get("stderr_tail"):
            lines.extend(["stderr tail:", "", "```text", item["stderr_tail"], "```", ""])
    lines.extend(
        [
            "## Next Step",
            "",
            "If status is `READY_TO_WRITE`, rerun this gate with `--write`. Add `--run-post-manual-pipeline` only when you want the guarded validate/score/report pipeline to run; use `--run-full-robustness` only after manual validation PASS and all-GPU preflight readiness are acceptable.",
        ]
    )
    (out_dir / "returned_manual_gate_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.run_full_robustness != "none" and not args.run_post_manual_pipeline:
        raise SystemExit("--run-full-robustness requires --run-post-manual-pipeline")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    backup_dir = args.backup_dir or str(Path(args.extractor_validation_dir) / "manual_annotation_backups")
    commands: list[dict[str, Any]] = []

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
    scan_result = run_command(scan_cmd)
    commands.append(scan_result)
    scan_payload = read_json(Path(args.return_scan_dir) / "manual_annotation_return_scan.json")
    ready_candidates = [row for row in scan_payload.get("candidates", []) if row.get("ready_to_import")]

    selected_candidate = ""
    import_payload: dict[str, Any] = {}
    wrote_target = False
    post_manual_result: dict[str, Any] | None = None

    if scan_result["returncode"] != 0:
        status = "SCAN_FAILED"
    elif len(ready_candidates) == 0:
        status = "WAITING_FOR_READY_CANDIDATE"
    elif len(ready_candidates) > 1:
        status = "MULTIPLE_READY_CANDIDATES"
    else:
        selected_candidate = ready_candidates[0]["path"]
        import_cmd = [
            args.py,
            "scripts/import_semantic_entropy_manual_annotations.py",
            "--input",
            selected_candidate,
            "--extractor-validation-dir",
            args.extractor_validation_dir,
            "--out-dir",
            args.import_dir,
            "--backup-dir",
            backup_dir,
        ]
        if args.write:
            import_cmd.append("--write")
        import_result = run_command(import_cmd)
        commands.append(import_result)
        import_payload = read_json(Path(args.import_dir) / "manual_annotation_import_report.json")
        wrote_target = bool(import_payload.get("wrote_target"))
        if import_result["returncode"] != 0:
            status = "IMPORT_FAILED"
        elif not args.write:
            status = "READY_TO_WRITE"
        elif not args.run_post_manual_pipeline:
            status = "WRITTEN_READY_FOR_POST_MANUAL_PIPELINE"
        else:
            env = os.environ.copy()
            env.setdefault("HF_HOME", str(Path.cwd() / ".cache/huggingface"))
            env.setdefault("TRANSFORMERS_OFFLINE", "1")
            env.setdefault("HF_HUB_OFFLINE", "1")
            env.setdefault("PYTHONPATH", str(Path.cwd()))
            env["EXTRACTOR_VALIDATION_DIR"] = args.extractor_validation_dir
            env["RUN_FULL_ROBUSTNESS"] = args.run_full_robustness
            env["WAIT_FOR_GPUS_SECONDS"] = str(args.wait_for_gpus_seconds)
            env["WAIT_FOR_GPUS_INTERVAL_SECONDS"] = str(args.wait_for_gpus_interval_seconds)
            env["DRY_RUN_FULL_ROBUSTNESS"] = "1" if args.dry_run_full_robustness else "0"
            env["PY"] = args.py
            post_manual_result = run_command(["bash", "scripts/run_semantic_entropy_post_manual_pipeline.sh"], env=env)
            commands.append(post_manual_result)
            status = "POST_MANUAL_PIPELINE_PASS" if post_manual_result["returncode"] == 0 else "POST_MANUAL_PIPELINE_FAILED"

    payload = {
        "status": status,
        "candidate_count": scan_payload.get("candidate_count", 0),
        "ready_to_import_count": scan_payload.get("ready_to_import_count", 0),
        "selected_candidate": selected_candidate,
        "write_requested": bool(args.write),
        "wrote_target": wrote_target,
        "run_post_manual_pipeline": bool(args.run_post_manual_pipeline),
        "run_full_robustness": args.run_full_robustness,
        "dry_run_full_robustness": bool(args.dry_run_full_robustness),
        "post_manual_pipeline_ran": post_manual_result is not None,
        "scan_report": str(Path(args.return_scan_dir) / "manual_annotation_return_scan.json"),
        "import_report": str(Path(args.import_dir) / "manual_annotation_import_report.json"),
        "commands": commands,
        "claim_boundary": "guarded execution handoff only; not manual validation or robustness evidence by itself",
    }
    (out_dir / "returned_manual_gate_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(out_dir, payload)
    print(
        json.dumps(
            {
                "status": status,
                "candidate_count": payload["candidate_count"],
                "ready_to_import_count": payload["ready_to_import_count"],
                "write_requested": payload["write_requested"],
                "post_manual_pipeline_ran": payload["post_manual_pipeline_ran"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
