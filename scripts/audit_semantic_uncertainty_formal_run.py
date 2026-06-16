#!/usr/bin/env python3
"""Audit completion of the semantic-uncertainty-style formal I2T/T2I run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_CONCEPTS = 30
EXPECTED_SAMPLES = 10
EXPECTED_T2I_IMAGES = EXPECTED_CONCEPTS * EXPECTED_SAMPLES
EXPECTED_STATES = EXPECTED_CONCEPTS * EXPECTED_SAMPLES * 2
EXPECTED_SLOT_ANSWERS = EXPECTED_STATES * 6
EXPECTED_ROUTE_ENTROPY_LINES = EXPECTED_CONCEPTS * 2 * 7 + 1
EXPECTED_ROUTE_ERROR_LINES = EXPECTED_CONCEPTS * 2 * 6 + 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        default="outputs/semantic_entropy_umm/semantic_uncertainty_formal_emu35_20260606_095911",
        help="Formal run directory to audit.",
    )
    parser.add_argument("--repo-root", default=".", help="Repository root containing third_party/semantic_uncertainty.")
    parser.add_argument("--write", action="store_true", help="Write formal_completion_audit.{json,md} into run-dir.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def line_count(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def jsonl_count(path: Path) -> int | None:
    return line_count(path)


def png_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.glob("*/*.png"))


def check(name: str, ok: bool, evidence: str, expected: Any = None, actual: Any = None, next_step: str = "") -> dict[str, Any]:
    return {
        "name": name,
        "status": "PASS" if ok else "FAIL",
        "expected": expected,
        "actual": actual,
        "evidence": evidence,
        "next_step": next_step,
    }


def build_audit(run_dir: Path, repo_root: Path) -> dict[str, Any]:
    pilot_dir = run_dir / "pilot"
    strict_dir = run_dir / "strict_compare"
    qa_dir = run_dir / "semantic_uncertainty_qa"
    driver_log = run_dir / "logs" / "driver.log"
    strict_audit = read_json(strict_dir / "comparability_audit.json")
    qa_audit = read_json(qa_dir / "qa_comparability_audit.json")
    driver_text = driver_log.read_text(encoding="utf-8", errors="replace") if driver_log.exists() else ""

    t2i_worker_rows = {
        path.name: jsonl_count(path)
        for path in sorted((pilot_dir / "workers").glob("t2i_samples_worker*.jsonl"))
    }

    checks = [
        check(
            "semantic_uncertainty repository cloned",
            (repo_root / "third_party" / "semantic_uncertainty" / "semantic_uncertainty").exists(),
            str(repo_root / "third_party" / "semantic_uncertainty"),
            expected="jlko/semantic_uncertainty checkout",
            actual="present" if (repo_root / "third_party" / "semantic_uncertainty").exists() else "missing",
            next_step="clone https://github.com/jlko/semantic_uncertainty under third_party",
        ),
        check(
            "formal protocol captured",
            (run_dir / "protocol.md").exists(),
            str(run_dir / "protocol.md"),
            expected="protocol.md in run dir",
            actual="present" if (run_dir / "protocol.md").exists() else "missing",
            next_step="copy docs/semantic_uncertainty_formal_emu35_protocol.md into the run directory",
        ),
        check(
            "concept set size",
            line_count(pilot_dir / "concepts.jsonl") == EXPECTED_CONCEPTS,
            str(pilot_dir / "concepts.jsonl"),
            expected=EXPECTED_CONCEPTS,
            actual=line_count(pilot_dir / "concepts.jsonl"),
            next_step="rerun prepare with NUM_CONCEPTS=30 if this fails",
        ),
        check(
            "T2I generated image count",
            png_count(pilot_dir / "generated_images") == EXPECTED_T2I_IMAGES,
            str(pilot_dir / "generated_images"),
            expected=EXPECTED_T2I_IMAGES,
            actual=png_count(pilot_dir / "generated_images"),
            next_step="wait for T2I generation or resume workers with --skip-existing",
        ),
        check(
            "T2I sample rows aggregated",
            jsonl_count(pilot_dir / "t2i_samples.jsonl") == EXPECTED_T2I_IMAGES,
            str(pilot_dir / "t2i_samples.jsonl"),
            expected=EXPECTED_T2I_IMAGES,
            actual=jsonl_count(pilot_dir / "t2i_samples.jsonl"),
            next_step="run pilot aggregate after T2I workers finish",
        ),
        check(
            "strict audit verdict",
            strict_audit.get("verdict") == "PASS",
            str(strict_dir / "comparability_audit.json"),
            expected="PASS",
            actual=strict_audit.get("verdict", "missing"),
            next_step="run strict aggregate/audit and inspect failures",
        ),
        check(
            "strict semantic state rows",
            jsonl_count(strict_dir / "strict_semantic_states.jsonl") == EXPECTED_STATES,
            str(strict_dir / "strict_semantic_states.jsonl"),
            expected=EXPECTED_STATES,
            actual=jsonl_count(strict_dir / "strict_semantic_states.jsonl"),
            next_step="finish strict workers and aggregate",
        ),
        check(
            "strict slot answer rows",
            jsonl_count(strict_dir / "strict_slot_answers.jsonl") == EXPECTED_SLOT_ANSWERS,
            str(strict_dir / "strict_slot_answers.jsonl"),
            expected=EXPECTED_SLOT_ANSWERS,
            actual=jsonl_count(strict_dir / "strict_slot_answers.jsonl"),
            next_step="finish strict workers and aggregate",
        ),
        check(
            "strict route entropy CSV rows",
            line_count(strict_dir / "strict_route_entropy.csv") == EXPECTED_ROUTE_ENTROPY_LINES,
            str(strict_dir / "strict_route_entropy.csv"),
            expected=EXPECTED_ROUTE_ENTROPY_LINES,
            actual=line_count(strict_dir / "strict_route_entropy.csv"),
            next_step="rerun strict aggregate",
        ),
        check(
            "strict route error CSV rows",
            line_count(strict_dir / "strict_route_error.csv") == EXPECTED_ROUTE_ERROR_LINES,
            str(strict_dir / "strict_route_error.csv"),
            expected=EXPECTED_ROUTE_ERROR_LINES,
            actual=line_count(strict_dir / "strict_route_error.csv"),
            next_step="rerun strict aggregate",
        ),
        check(
            "semantic-uncertainty QA audit verdict",
            qa_audit.get("verdict") == "PASS",
            str(qa_dir / "qa_comparability_audit.json"),
            expected="PASS",
            actual=qa_audit.get("verdict", "missing"),
            next_step="run QA aggregate/audit and inspect failures",
        ),
        check(
            "QA semantic state rows",
            jsonl_count(qa_dir / "qa_semantic_states.jsonl") == EXPECTED_STATES,
            str(qa_dir / "qa_semantic_states.jsonl"),
            expected=EXPECTED_STATES,
            actual=jsonl_count(qa_dir / "qa_semantic_states.jsonl"),
            next_step="finish QA workers and aggregate",
        ),
        check(
            "QA slot answer rows",
            jsonl_count(qa_dir / "qa_slot_answers.jsonl") == EXPECTED_SLOT_ANSWERS,
            str(qa_dir / "qa_slot_answers.jsonl"),
            expected=EXPECTED_SLOT_ANSWERS,
            actual=jsonl_count(qa_dir / "qa_slot_answers.jsonl"),
            next_step="finish QA workers and aggregate",
        ),
        check(
            "QA route entropy CSV rows",
            line_count(qa_dir / "qa_route_entropy.csv") == EXPECTED_ROUTE_ENTROPY_LINES,
            str(qa_dir / "qa_route_entropy.csv"),
            expected=EXPECTED_ROUTE_ENTROPY_LINES,
            actual=line_count(qa_dir / "qa_route_entropy.csv"),
            next_step="rerun QA aggregate",
        ),
        check(
            "QA route error CSV rows",
            line_count(qa_dir / "qa_route_error.csv") == EXPECTED_ROUTE_ERROR_LINES,
            str(qa_dir / "qa_route_error.csv"),
            expected=EXPECTED_ROUTE_ERROR_LINES,
            actual=line_count(qa_dir / "qa_route_error.csv"),
            next_step="rerun QA aggregate",
        ),
        check(
            "driver completed",
            "[DONE]" in driver_text,
            str(driver_log),
            expected="[DONE] line",
            actual="[DONE] present" if "[DONE]" in driver_text else "not yet",
            next_step="wait for launcher to finish or inspect worker failures",
        ),
    ]
    status = "PASS" if all(row["status"] == "PASS" for row in checks) else "INCOMPLETE"
    return {
        "status": status,
        "run_dir": str(run_dir),
        "expected": {
            "concepts": EXPECTED_CONCEPTS,
            "samples_per_concept": EXPECTED_SAMPLES,
            "t2i_images": EXPECTED_T2I_IMAGES,
            "states_per_compare": EXPECTED_STATES,
            "slot_answers_per_compare": EXPECTED_SLOT_ANSWERS,
            "route_entropy_lines": EXPECTED_ROUTE_ENTROPY_LINES,
            "route_error_lines": EXPECTED_ROUTE_ERROR_LINES,
        },
        "current": {
            "t2i_worker_rows": t2i_worker_rows,
            "png_count": png_count(pilot_dir / "generated_images"),
            "t2i_aggregate_rows": jsonl_count(pilot_dir / "t2i_samples.jsonl"),
            "strict_verdict": strict_audit.get("verdict"),
            "qa_verdict": qa_audit.get("verdict"),
            "driver_done": "[DONE]" in driver_text,
        },
        "checks": checks,
    }


def write_markdown(path: Path, audit: dict[str, Any]) -> None:
    lines = [
        "# Semantic-Uncertainty Formal Run Completion Audit",
        "",
        f"Status: **{audit['status']}**",
        "",
        f"Run dir: `{audit['run_dir']}`",
        "",
        "## Current Counts",
        "",
        "```json",
        json.dumps(audit["current"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Checks",
        "",
        "| Check | Status | Expected | Actual | Evidence | Next step |",
        "|---|---|---:|---:|---|---|",
    ]
    for row in audit["checks"]:
        lines.append(
            "| {name} | {status} | `{expected}` | `{actual}` | `{evidence}` | {next_step} |".format(
                name=row["name"],
                status=row["status"],
                expected=row["expected"],
                actual=row["actual"],
                evidence=row["evidence"],
                next_step=row["next_step"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    repo_root = Path(args.repo_root)
    audit = build_audit(run_dir, repo_root)
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    if args.write:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "formal_completion_audit.json").write_text(
            json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        write_markdown(run_dir / "formal_completion_audit.md", audit)
    if audit["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
