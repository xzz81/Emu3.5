#!/usr/bin/env python3
"""Audit semantic-entropy stage reports against the execution-plan report contract."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


REQUIRED_SECTIONS = [
    "input files",
    "commands",
    "output files",
    "sample counts",
    "pass/fail checks",
    "claim allowed after this step",
    "claim still not allowed",
]

REPORTS = [
    ("strict_compare", "outputs/semantic_entropy_umm/strict_compare/comparability_audit.md", False),
    ("extractor_validation", "outputs/semantic_entropy_umm/extractor_validation/extractor_validation_report.md", True),
    ("quadrant_analysis", "outputs/semantic_entropy_umm/quadrant_analysis/quadrant_report.md", True),
    ("bootstrap_stability", "outputs/semantic_entropy_umm/bootstrap_stability/bootstrap_report.md", True),
    ("sample_size_sensitivity", "outputs/semantic_entropy_umm/sample_size_sensitivity/sample_size_sensitivity_report.md", False),
    ("option_order_smoke", "outputs/semantic_entropy_umm/robustness_option_order_smoke/option_order_audit.md", False),
    ("prompt_template_smoke", "outputs/semantic_entropy_umm/robustness_prompt_template_smoke/prompt_template_audit.md", False),
    ("post_manual_pipeline_guard", "outputs/semantic_entropy_umm/post_manual_pipeline_guard/post_manual_pipeline_guard.md", False),
    ("object_binding_sanity", "outputs/semantic_entropy_umm/object_binding_sanity/object_binding_sanity_report.md", False),
    ("image_duplicate_mode_collapse", "outputs/semantic_entropy_umm/image_duplicate_mode_collapse/image_duplicate_mode_collapse_report.md", False),
    ("image_embedding_duplicate_check", "outputs/semantic_entropy_umm/image_embedding_duplicate_check/image_embedding_duplicate_report.md", False),
    ("alternative_extractor_feasibility", "outputs/semantic_entropy_umm/alternative_extractor_feasibility/alternative_extractor_runbook.md", False),
    ("claim_boundary_audit", "outputs/semantic_entropy_umm/claim_boundary_audit/claim_boundary_audit_report.md", False),
    ("paper_tables_preliminary", "outputs/semantic_entropy_umm/paper_tables_preliminary/paper_tables_preliminary.md", False),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="outputs/semantic_entropy_umm/stage_report_audit")
    return parser.parse_args()


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def headings(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        match = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
        if match:
            heading = match.group(1).strip().strip("*").lower()
            out.append(heading)
    return out


def has_section(report_headings: list[str], required: str) -> bool:
    return any(required in heading for heading in report_headings)


def audit_report(stage: str, path: Path, dod_critical: bool) -> dict[str, Any]:
    text = read_text(path)
    report_headings = headings(text)
    section_status = {section: has_section(report_headings, section) for section in REQUIRED_SECTIONS}
    missing = [section for section, present in section_status.items() if not present]
    exists = path.exists()
    status = "PASS" if exists and not missing else "MISSING_SECTIONS" if exists else "MISSING_FILE"
    return {
        "stage": stage,
        "path": str(path),
        "dod_critical": str(bool(dod_critical)).lower(),
        "status": status,
        "missing_sections": ";".join(missing),
        **{section.replace("/", "_").replace(" ", "_"): str(section_status[section]).lower() for section in REQUIRED_SECTIONS},
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "stage",
        "path",
        "dod_critical",
        "status",
        "missing_sections",
        *[section.replace("/", "_").replace(" ", "_") for section in REQUIRED_SECTIONS],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def md_table(rows: list[dict[str, Any]], fields: list[str]) -> list[str]:
    if not rows:
        return ["(none)"]
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        safe = {field: str(row.get(field, "")).replace("|", "\\|").replace("\n", " ") for field in fields}
        lines.append("| " + " | ".join(safe[field] for field in fields) + " |")
    return lines


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    missing_rows = [row for row in rows if row["status"] != "PASS"]
    dod_missing = [row for row in missing_rows if row["dod_critical"] == "true"]
    status = "PASS" if not missing_rows else "ACTION_REQUIRED"
    lines = [
        "# Stage Report Completeness Audit",
        "",
        f"Status: {status}",
        "",
        "This audit checks whether stage reports satisfy the execution-plan reporting contract: input files, commands, output files, sample counts, pass/fail checks, and claim boundaries. It audits report structure only; it does not validate scientific results.",
        "",
        "## Summary",
        "",
        f"- Reports checked: `{len(rows)}`",
        f"- Reports with missing sections: `{len(missing_rows)}`",
        f"- DoD-critical reports with missing sections: `{len(dod_missing)}`",
        "",
        "## DoD-Critical Gaps",
        "",
        *md_table(dod_missing, ["stage", "status", "missing_sections", "path"]),
        "",
        "## All Missing Sections",
        "",
        *md_table(missing_rows, ["stage", "dod_critical", "status", "missing_sections", "path"]),
        "",
        "## Full Matrix",
        "",
        *md_table(rows, ["stage", "dod_critical", "status", "missing_sections", "path"]),
        "",
        "## Claim Boundary",
        "",
        "Allowed: report-completeness tracking.",
        "",
        "Not allowed: treating a structurally complete report as proof of manual validation, full robustness, or final paper readiness.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [audit_report(stage, Path(path), dod_critical) for stage, path, dod_critical in REPORTS]
    status = "PASS" if all(row["status"] == "PASS" for row in rows) else "ACTION_REQUIRED"
    payload = {
        "status": status,
        "required_sections": REQUIRED_SECTIONS,
        "reports_checked": len(rows),
        "reports_with_missing_sections": sum(row["status"] != "PASS" for row in rows),
        "dod_critical_missing_sections": sum(row["status"] != "PASS" and row["dod_critical"] == "true" for row in rows),
        "rows": rows,
        "claim_boundary": "report structure audit only; not scientific evidence",
    }
    (out_dir / "stage_report_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(out_dir / "stage_report_audit.csv", rows)
    write_report(out_dir / "stage_report_audit.md", rows)
    print(
        json.dumps(
            {
                "status": status,
                "reports_checked": payload["reports_checked"],
                "reports_with_missing_sections": payload["reports_with_missing_sections"],
                "dod_critical_missing_sections": payload["dod_critical_missing_sections"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
