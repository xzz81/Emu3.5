#!/usr/bin/env python3
"""Check that the post-manual pipeline is gated by manual validation.

When manual annotations are incomplete, this script runs the real
``run_semantic_entropy_post_manual_pipeline.sh`` entrypoint and verifies that it
stops at manual validation before scoring or full robustness. When manual
validation is already PASS, it skips execution to avoid mutating score outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any


SLOTS = ["object_1", "color_1", "object_2", "color_2", "relation", "background"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", default="scripts/run_semantic_entropy_post_manual_pipeline.sh")
    parser.add_argument("--extractor-validation-dir", default="outputs/semantic_entropy_umm/extractor_validation")
    parser.add_argument("--out-dir", default="outputs/semantic_entropy_umm/post_manual_pipeline_guard")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return max(0, len([line for line in text if line.strip()]) - 1)


def run_pipeline(pipeline: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.setdefault("HF_HOME", str(Path.cwd() / ".cache/huggingface"))
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("PYTHONPATH", str(Path.cwd()))
    env["RUN_FULL_ROBUSTNESS"] = "0"
    return subprocess.run(
        ["bash", str(pipeline)],
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )


def write_report(out_dir: Path, payload: dict[str, Any]) -> None:
    stdout_tail = "\n".join(payload.get("stdout", "").splitlines()[-40:])
    stderr_tail = "\n".join(payload.get("stderr", "").splitlines()[-40:])
    lines = [
        "# Post-Manual Pipeline Guard",
        "",
        f"Status: {payload['status']}",
        "",
        "This report checks the real post-manual shell entrypoint. It is a guard audit only; it does not provide human validation evidence.",
        "",
        "## Input Files",
        "",
        f"- `{payload['pipeline']}`",
        f"- `{payload['extractor_validation_dir']}/manual_annotations.csv`",
        f"- `{payload['extractor_validation_dir']}/manual_annotation_validation_report.json`",
        "",
        "## Commands",
        "",
        "```bash",
        "RUN_FULL_ROBUSTNESS=0 bash " + payload["pipeline"],
        "```",
        "",
        "## Output Files",
        "",
        f"- `{out_dir / 'post_manual_pipeline_guard.json'}`",
        f"- `{out_dir / 'post_manual_pipeline_guard.md'}`",
        f"- `{out_dir / 'post_manual_pipeline_guard_stdout.txt'}`",
        f"- `{out_dir / 'post_manual_pipeline_guard_stderr.txt'}`",
        "",
        "## Sample Counts",
        "",
        f"- Manifest rows: `{payload['manifest_rows']}`",
        f"- Manual rows: `{payload['manual_rows']}`",
        f"- Completed slot labels before run: `{payload['completed_slot_labels_before']}` / `{payload['required_slot_labels_before']}`",
        f"- Completed slot labels after run: `{payload['completed_slot_labels_after']}` / `{payload['required_slot_labels_after']}`",
        "",
        "## Pass/Fail Checks",
        "",
        f"- Manual validation was incomplete before run: `{payload['manual_incomplete_before']}`",
        f"- Pipeline return code was nonzero: `{payload['returncode_nonzero']}`",
        f"- Stopped before scoring: `{payload['stopped_before_scoring']}`",
        f"- Stopped before full robustness: `{payload['stopped_before_full_robustness']}`",
        f"- Blocked by manual validation: `{payload['blocked_by_manual_validation']}`",
        "",
        "## Stdout Tail",
        "",
        "```text",
        stdout_tail,
        "```",
        "",
        "## Stderr Tail",
        "",
        "```text",
        stderr_tail,
        "```",
        "",
        "## Claim Allowed After This Step",
        "",
        "Allowed: the current post-manual entrypoint is guarded by manual validation while labels are incomplete.",
        "",
        "## Claim Still Not Allowed",
        "",
        "Do not claim extractor validation, scoring completion, full robustness, or final paper-readiness from this guard audit.",
    ]
    (out_dir / "post_manual_pipeline_guard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline = Path(args.pipeline)
    validation_dir = Path(args.extractor_validation_dir)
    before = read_json(validation_dir / "manual_annotation_validation_report.json")
    manual_incomplete_before = before.get("status") != "PASS"
    manifest_rows = count_csv_rows(validation_dir / "annotation_manifest.csv")
    manual_rows = count_csv_rows(validation_dir / "manual_annotations.csv")

    if manual_incomplete_before:
        completed = run_pipeline(pipeline)
        stdout = completed.stdout
        stderr = completed.stderr
        returncode = completed.returncode
    else:
        stdout = "[INFO] manual validation already PASS; guard run skipped to avoid scoring side effects\n"
        stderr = ""
        returncode = 0

    after = read_json(validation_dir / "manual_annotation_validation_report.json")
    scoring_marker = "[INFO] scoring extractor validation"
    robustness_marker = "running full"
    stopped_before_scoring = scoring_marker not in stdout
    stopped_before_full_robustness = robustness_marker not in stdout
    blocked = (
        manual_incomplete_before
        and returncode != 0
        and after.get("status") != "PASS"
        and stopped_before_scoring
        and stopped_before_full_robustness
    )
    status = "PASS_BLOCKED_BY_MANUAL_VALIDATION" if blocked else "SKIPPED_MANUAL_ALREADY_PASS" if not manual_incomplete_before else "FAIL"

    payload = {
        "status": status,
        "pipeline": str(pipeline),
        "extractor_validation_dir": str(validation_dir),
        "manifest_rows": manifest_rows,
        "manual_rows": manual_rows,
        "manual_status_before": before.get("status", "MISSING"),
        "manual_status_after": after.get("status", "MISSING"),
        "completed_slot_labels_before": before.get("completed_slot_labels", 0),
        "required_slot_labels_before": before.get("required_slot_labels", manifest_rows * len(SLOTS)),
        "completed_slot_labels_after": after.get("completed_slot_labels", 0),
        "required_slot_labels_after": after.get("required_slot_labels", manifest_rows * len(SLOTS)),
        "manual_incomplete_before": manual_incomplete_before,
        "returncode": returncode,
        "returncode_nonzero": returncode != 0,
        "stopped_before_scoring": stopped_before_scoring,
        "stopped_before_full_robustness": stopped_before_full_robustness,
        "blocked_by_manual_validation": blocked,
        "claim_boundary": "guard audit only; not human validation evidence",
        "stdout": stdout,
        "stderr": stderr,
    }
    (out_dir / "post_manual_pipeline_guard_stdout.txt").write_text(stdout, encoding="utf-8")
    (out_dir / "post_manual_pipeline_guard_stderr.txt").write_text(stderr, encoding="utf-8")
    (out_dir / "post_manual_pipeline_guard.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(out_dir, payload)
    print(
        json.dumps(
            {
                "status": status,
                "returncode": returncode,
                "manual_status_before": payload["manual_status_before"],
                "manual_status_after": payload["manual_status_after"],
                "blocked_by_manual_validation": blocked,
            },
            indent=2,
        )
    )
    if status == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
