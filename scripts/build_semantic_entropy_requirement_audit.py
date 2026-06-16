#!/usr/bin/env python3
"""Build a requirement and Definition-of-Done audit for semantic entropy work."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = [
    "category",
    "requirement",
    "status",
    "evidence",
    "observed",
    "next_step",
    "blocks_goal",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-dir", default="outputs/semantic_entropy_umm/strict_compare")
    parser.add_argument("--extractor-validation-dir", default="outputs/semantic_entropy_umm/extractor_validation")
    parser.add_argument("--manual-progress-dir", default="outputs/semantic_entropy_umm/manual_annotation_progress")
    parser.add_argument("--manual-package-dir", default="outputs/semantic_entropy_umm/manual_annotation_package")
    parser.add_argument("--manual-package-qa-dir", default="outputs/semantic_entropy_umm/manual_annotation_package_qa")
    parser.add_argument("--manual-return-scan-dir", default="outputs/semantic_entropy_umm/manual_annotation_return_scan")
    parser.add_argument("--manual-import-dir", default="outputs/semantic_entropy_umm/manual_annotation_import")
    parser.add_argument("--returned-manual-gate-dir", default="outputs/semantic_entropy_umm/returned_manual_gate")
    parser.add_argument("--returned-manual-watch-dir", default="outputs/semantic_entropy_umm/returned_manual_watch")
    parser.add_argument("--returned-manual-gate-smoke-dir", default="outputs/semantic_entropy_umm/returned_manual_gate_smoke")
    parser.add_argument("--full-robustness-gpu-guard-smoke-dir", default="outputs/semantic_entropy_umm/full_robustness_gpu_guard_smoke")
    parser.add_argument("--full-robustness-dry-run-smoke-dir", default="outputs/semantic_entropy_umm/full_robustness_dry_run_smoke")
    parser.add_argument("--post-manual-smoke-dir", default="outputs/semantic_entropy_umm/post_manual_smoke")
    parser.add_argument("--post-manual-guard-dir", default="outputs/semantic_entropy_umm/post_manual_pipeline_guard")
    parser.add_argument("--quadrant-dir", default="outputs/semantic_entropy_umm/quadrant_analysis")
    parser.add_argument("--bootstrap-dir", default="outputs/semantic_entropy_umm/bootstrap_stability")
    parser.add_argument("--sample-size-dir", default="outputs/semantic_entropy_umm/sample_size_sensitivity")
    parser.add_argument("--option-order-full-dir", default="outputs/semantic_entropy_umm/robustness_option_order")
    parser.add_argument("--option-order-full-fallback-dir", default="outputs/semantic_entropy_umm/robustness_option_order_idle_gpus")
    parser.add_argument("--prompt-template-full-dir", default="outputs/semantic_entropy_umm/robustness_prompt_template")
    parser.add_argument("--prompt-template-full-fallback-dir", default="outputs/semantic_entropy_umm/robustness_prompt_template_idle_gpus")
    parser.add_argument("--robustness-preflight-dir", default="outputs/semantic_entropy_umm/robustness_preflight")
    parser.add_argument("--all-gpu-runbook-dir", default="outputs/semantic_entropy_umm/all_gpu_execution_runbook")
    parser.add_argument("--object-binding-dir", default="outputs/semantic_entropy_umm/object_binding_sanity")
    parser.add_argument("--image-duplicate-dir", default="outputs/semantic_entropy_umm/image_duplicate_mode_collapse")
    parser.add_argument("--image-embedding-dir", default="outputs/semantic_entropy_umm/image_embedding_duplicate_check")
    parser.add_argument("--alternative-extractor-dir", default="outputs/semantic_entropy_umm/alternative_extractor_feasibility")
    parser.add_argument("--claim-boundary-dir", default="outputs/semantic_entropy_umm/claim_boundary_audit")
    parser.add_argument("--stage-report-audit-dir", default="outputs/semantic_entropy_umm/stage_report_audit")
    parser.add_argument("--paper-dir", default="outputs/semantic_entropy_umm/paper_tables_preliminary")
    parser.add_argument("--out-dir", default="outputs/semantic_entropy_umm/requirement_audit")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def exists(path: Path) -> bool:
    return path.exists()


def line_count(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8", errors="replace") as handle:
        return sum(1 for _ in handle)


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def candidate_dirs(*paths: str) -> list[Path]:
    out: list[Path] = []
    for path in paths:
        candidate = Path(path)
        if candidate not in out:
            out.append(candidate)
    return out


def robustness_audit_pass(audit: dict[str, Any]) -> bool:
    return (
        audit.get("status") == "PASS"
        and audit.get("state_rows") == 600
        and audit.get("slot_answer_rows") == 3600
        and audit.get("schema_ok") is True
        and audit.get("vocab_ok") is True
    )


def full_option_order_status(dirs: list[Path]) -> tuple[bool, str, str]:
    seeds = ["seed_20270605", "seed_20270606", "seed_20270607"]
    observations = []
    for directory in dirs:
        audits = [read_json(directory / seed / "option_order_audit.json") for seed in seeds]
        seed_ok = [robustness_audit_pass(audit) for audit in audits]
        changed = [audit.get("changed_slot_values_vs_baseline_subset", "NA") for audit in audits]
        observations.append(
            f"{directory}: seed_ok={seed_ok}; changed_slot_values={changed}"
        )
        if all(seed_ok):
            return True, str(directory), observations[-1]
    return False, "; ".join(str(directory) for directory in dirs), " | ".join(observations)


def full_prompt_template_status(dirs: list[Path]) -> tuple[bool, str, str]:
    templates = ["template_v2_short_direct", "template_v3_question_first"]
    observations = []
    for directory in dirs:
        audits = [read_json(directory / template / "prompt_template_audit.json") for template in templates]
        template_ok = [robustness_audit_pass(audit) for audit in audits]
        changed = [audit.get("changed_slot_values_vs_baseline_subset", "NA") for audit in audits]
        observations.append(
            f"{directory}: template_ok={template_ok}; changed_slot_values={changed}"
        )
        if all(template_ok):
            return True, str(directory), observations[-1]
    return False, "; ".join(str(directory) for directory in dirs), " | ".join(observations)


def row(
    category: str,
    requirement: str,
    status: str,
    evidence: Path | str,
    observed: str,
    next_step: str,
    blocks_goal: bool,
) -> dict[str, str]:
    return {
        "category": category,
        "requirement": requirement,
        "status": status,
        "evidence": str(evidence),
        "observed": observed,
        "next_step": next_step,
        "blocks_goal": bool_text(blocks_goal),
    }


def status_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in rows:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    return counts


def build_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    strict_dir = Path(args.strict_dir)
    extractor_dir = Path(args.extractor_validation_dir)
    manual_progress_dir = Path(args.manual_progress_dir)
    manual_package_dir = Path(args.manual_package_dir)
    manual_package_qa_dir = Path(args.manual_package_qa_dir)
    manual_return_scan_dir = Path(args.manual_return_scan_dir)
    manual_import_dir = Path(args.manual_import_dir)
    returned_manual_gate_dir = Path(args.returned_manual_gate_dir)
    returned_manual_watch_dir = Path(args.returned_manual_watch_dir)
    returned_manual_gate_smoke_dir = Path(args.returned_manual_gate_smoke_dir)
    full_robustness_gpu_guard_smoke_dir = Path(args.full_robustness_gpu_guard_smoke_dir)
    full_robustness_dry_run_smoke_dir = Path(args.full_robustness_dry_run_smoke_dir)
    post_manual_smoke_dir = Path(args.post_manual_smoke_dir)
    post_manual_guard_dir = Path(args.post_manual_guard_dir)
    quadrant_dir = Path(args.quadrant_dir)
    bootstrap_dir = Path(args.bootstrap_dir)
    sample_size_dir = Path(args.sample_size_dir)
    option_full_dir = Path(args.option_order_full_dir)
    option_full_dirs = candidate_dirs(args.option_order_full_dir, args.option_order_full_fallback_dir)
    prompt_full_dir = Path(args.prompt_template_full_dir)
    prompt_full_dirs = candidate_dirs(args.prompt_template_full_dir, args.prompt_template_full_fallback_dir)
    robustness_preflight_dir = Path(args.robustness_preflight_dir)
    all_gpu_runbook_dir = Path(args.all_gpu_runbook_dir)
    object_binding_dir = Path(args.object_binding_dir)
    image_duplicate_dir = Path(args.image_duplicate_dir)
    image_embedding_dir = Path(args.image_embedding_dir)
    alternative_dir = Path(args.alternative_extractor_dir)
    claim_dir = Path(args.claim_boundary_dir)
    stage_report_audit_dir = Path(args.stage_report_audit_dir)
    paper_dir = Path(args.paper_dir)

    strict_audit = read_json(strict_dir / "comparability_audit.json")
    manual_report = read_json(extractor_dir / "manual_annotation_validation_report.json")
    manual_return_scan = read_json(manual_return_scan_dir / "manual_annotation_return_scan.json")
    returned_manual_gate = read_json(returned_manual_gate_dir / "returned_manual_gate_report.json")
    returned_manual_watch = read_json(returned_manual_watch_dir / "returned_manual_watch_report.json")
    returned_manual_gate_smoke = read_json(returned_manual_gate_smoke_dir / "returned_manual_gate_smoke_report.json")
    full_robustness_gpu_guard_smoke = read_json(
        full_robustness_gpu_guard_smoke_dir / "full_robustness_gpu_guard_smoke_report.json"
    )
    full_robustness_dry_run_smoke = read_json(
        full_robustness_dry_run_smoke_dir / "full_robustness_dry_run_smoke_report.json"
    )
    targeted_binding_score = read_json(
        object_binding_dir / "targeted_questions" / "targeted_binding_score_report.json"
    )
    alt_score = read_json(alternative_dir / "alternative_extractor_score_report.json")
    alt_feasibility = read_json(alternative_dir / "alternative_extractor_feasibility.json")
    embedding_preflight = read_json(image_embedding_dir / "embedding_preflight.json")
    robustness_preflight = read_json(robustness_preflight_dir / "robustness_preflight_report.json")
    all_gpu_runbook = read_json(all_gpu_runbook_dir / "all_gpu_execution_runbook.json")

    strict_counts = {
        "states": line_count(strict_dir / "strict_semantic_states.jsonl"),
        "slot_answers": line_count(strict_dir / "strict_slot_answers.jsonl"),
        "route_entropy_rows": line_count(strict_dir / "strict_route_entropy.csv"),
        "route_error_rows": line_count(strict_dir / "strict_route_error.csv"),
    }
    strict_ok = (
        strict_audit.get("verdict") == "PASS"
        and strict_counts == {
            "states": 600,
            "slot_answers": 3600,
            "route_entropy_rows": 421,
            "route_error_rows": 361,
        }
    )
    schema_vocab_ok = bool(strict_audit.get("schema_ok")) and bool(
        strict_audit.get("vocab_ok", strict_audit.get("vocabulary_ok"))
    )

    manual_pass = manual_report.get("status") == "PASS"
    score_exists = exists(extractor_dir / "extractor_validation_metrics.csv") and exists(
        extractor_dir / "extractor_joined_annotations.csv"
    )
    extractor_report_exists = exists(extractor_dir / "extractor_validation_report.md")
    annotation_counts_ok = line_count(extractor_dir / "annotation_manifest.csv") == 61 and line_count(
        extractor_dir / "manual_annotations.csv"
    ) == 61
    batch_counts_ok = line_count(extractor_dir / "annotation_batches" / "annotation_batches_manifest.csv") == 7

    option_full_complete, option_full_evidence, option_full_observed = full_option_order_status(option_full_dirs)
    prompt_full_complete, prompt_full_evidence, prompt_full_observed = full_prompt_template_status(prompt_full_dirs)
    any_full_robustness = option_full_complete or prompt_full_complete
    option_full_ready = any(exists(directory / "launchers" / "launch_all_option_order_seeds.sh") for directory in option_full_dirs)
    prompt_full_ready = any(exists(directory / "launchers" / "launch_all_prompt_templates.sh") for directory in prompt_full_dirs)
    targeted_binding_pass = (
        targeted_binding_score.get("status") == "PASS"
        and targeted_binding_score.get("completed_answers") == targeted_binding_score.get("question_rows")
        and targeted_binding_score.get("question_rows") == 90
    )
    minimum_dod = (
        strict_ok
        and manual_pass
        and score_exists
        and extractor_report_exists
        and exists(quadrant_dir / "quadrant_report.md")
        and exists(bootstrap_dir / "bootstrap_report.md")
        and any_full_robustness
        and exists(claim_dir / "claim_boundary_audit_report.md")
    )

    rows = [
        row(
            "protocol",
            "strict compare audit PASS with expected counts",
            "PASS" if strict_ok else "FAIL",
            strict_dir / "comparability_audit.json",
            f"verdict={strict_audit.get('verdict', 'NA')}; counts={strict_counts}",
            "rerun strict audit only if counts drift",
            True,
        ),
        row(
            "protocol",
            "same finite slot schema and vocabulary enforced",
            "PASS" if schema_vocab_ok else "FAIL",
            strict_dir / "comparability_audit.json",
            f"schema_ok={strict_audit.get('schema_ok', 'NA')}; vocab_ok={strict_audit.get('vocab_ok', strict_audit.get('vocabulary_ok', 'NA'))}",
            "inspect strict audit if schema/vocabulary fails",
            True,
        ),
        row(
            "extractor_validation",
            "fixed 60-row manual validation sample prepared",
            "PASS" if annotation_counts_ok else "FAIL",
            extractor_dir,
            f"annotation_manifest_lines={line_count(extractor_dir / 'annotation_manifest.csv')}; manual_annotation_lines={line_count(extractor_dir / 'manual_annotations.csv')}",
            "regenerate extractor validation package if counts drift",
            True,
        ),
        row(
            "extractor_validation",
            "manual annotation batches prepared",
            "READY_ONLY" if batch_counts_ok else "FAIL",
            extractor_dir / "annotation_batches",
            f"annotation_batches_manifest_lines={line_count(extractor_dir / 'annotation_batches' / 'annotation_batches_manifest.csv')}",
            "fill batch CSVs, merge, inspect, then write main manual_annotations.csv",
            False,
        ),
        row(
            "extractor_validation",
            "manual annotation progress audit generated",
            "READY_ONLY"
            if exists(manual_progress_dir / "manual_annotation_progress.md")
            and exists(manual_progress_dir / "manual_annotation_progress.csv")
            and exists(manual_progress_dir / "manual_annotation_progress.json")
            else "FAIL",
            manual_progress_dir / "manual_annotation_progress.md",
            f"report_exists={exists(manual_progress_dir / 'manual_annotation_progress.md')}; csv_exists={exists(manual_progress_dir / 'manual_annotation_progress.csv')}; json_exists={exists(manual_progress_dir / 'manual_annotation_progress.json')}",
            "refresh while filling manual_annotations.csv; validator PASS is still required",
            False,
        ),
        row(
            "extractor_validation",
            "portable manual annotation package generated",
            "READY_ONLY"
            if exists(manual_package_dir / "README.md")
            and exists(manual_package_dir / "package_manifest.json")
            and exists(manual_package_dir / "blind_manifest.csv")
            and exists(manual_package_dir.parent / "manual_annotation_package.zip")
            else "FAIL",
            manual_package_dir,
            f"readme_exists={exists(manual_package_dir / 'README.md')}; package_manifest_exists={exists(manual_package_dir / 'package_manifest.json')}; blind_manifest_exists={exists(manual_package_dir / 'blind_manifest.csv')}; zip_exists={exists(manual_package_dir.parent / 'manual_annotation_package.zip')}",
            "send package to annotator; import completed manual_annotations.csv afterward",
            False,
        ),
        row(
            "extractor_validation",
            "portable manual annotation package QA PASS",
            "READY_ONLY"
            if read_json(manual_package_qa_dir / "manual_annotation_package_qa.json").get("status") == "PASS"
            and exists(manual_package_qa_dir / "manual_annotation_package_qa.md")
            else "FAIL",
            manual_package_qa_dir / "manual_annotation_package_qa.md",
            f"status={read_json(manual_package_qa_dir / 'manual_annotation_package_qa.json').get('status', 'NA')}; report_exists={exists(manual_package_qa_dir / 'manual_annotation_package_qa.md')}",
            "rerun after regenerating the annotation package",
            False,
        ),
        row(
            "extractor_validation",
            "returned manual annotation scanner ready",
            "READY_ONLY"
            if exists(Path("scripts/find_semantic_entropy_returned_annotations.py"))
            and exists(manual_return_scan_dir / "manual_annotation_return_scan.md")
            else "FAIL",
            manual_return_scan_dir / "manual_annotation_return_scan.md",
            f"status={manual_return_scan.get('status', 'NA')}; ready_candidates={manual_return_scan.get('ready_to_import_count', 'NA')}; candidate_count={manual_return_scan.get('candidate_count', 'NA')}",
            "place returned CSV in manual_annotation_returns or pass --candidate, then import only after READY_TO_IMPORT",
            False,
        ),
        row(
            "extractor_validation",
            "safe manual annotation importer dry-run generated",
            "READY_ONLY"
            if exists(Path("scripts/import_semantic_entropy_manual_annotations.py"))
            and exists(manual_import_dir / "manual_annotation_import_report.md")
            and exists(manual_import_dir / "manual_annotation_import_report.json")
            else "FAIL",
            manual_import_dir / "manual_annotation_import_report.md",
            f"script_exists={exists(Path('scripts/import_semantic_entropy_manual_annotations.py'))}; report_exists={exists(manual_import_dir / 'manual_annotation_import_report.md')}; json_exists={exists(manual_import_dir / 'manual_annotation_import_report.json')}",
            "rerun on returned annotator CSV; --write only when READY_TO_IMPORT",
            False,
        ),
        row(
            "extractor_validation",
            "returned manual gate runner ready",
            "READY_ONLY"
            if exists(Path("scripts/run_semantic_entropy_returned_manual_gate.py"))
            and exists(returned_manual_gate_dir / "returned_manual_gate_report.md")
            else "FAIL",
            returned_manual_gate_dir / "returned_manual_gate_report.md",
            f"status={returned_manual_gate.get('status', 'NA')}; ready_candidates={returned_manual_gate.get('ready_to_import_count', 'NA')}; wrote={returned_manual_gate.get('wrote_target', 'NA')}",
            "rerun after a returned annotation CSV arrives; use --write only after READY_TO_WRITE",
            False,
        ),
        row(
            "extractor_validation",
            "returned manual watch-and-launch runner ready",
            "READY_ONLY"
            if exists(Path("scripts/watch_semantic_entropy_returned_manual_and_launch.py"))
            and exists(returned_manual_watch_dir / "returned_manual_watch_report.md")
            else "FAIL",
            returned_manual_watch_dir / "returned_manual_watch_report.md",
            f"status={returned_manual_watch.get('status', 'NA')}; gate_ran={returned_manual_watch.get('gate_ran', 'NA')}; run_full={returned_manual_watch.get('run_full_robustness', 'NA')}",
            "run only as a bounded watcher; full robustness still requires a single READY returned CSV and explicit launch flags",
            False,
        ),
        row(
            "extractor_validation",
            "returned manual gate write smoke PASS",
            "READY_ONLY"
            if returned_manual_gate_smoke.get("status") == "PASS"
            and exists(returned_manual_gate_smoke_dir / "returned_manual_gate_smoke_report.md")
            else "FAIL",
            returned_manual_gate_smoke_dir / "returned_manual_gate_smoke_report.md",
            f"gate_status={returned_manual_gate_smoke.get('gate_status', 'NA')}; wrote={returned_manual_gate_smoke.get('wrote_target', 'NA')}; post_manual_ran={returned_manual_gate_smoke.get('post_manual_pipeline_ran', 'NA')}",
            "rerun after editing returned manual gate or importer write behavior",
            False,
        ),
        row(
            "extractor_validation",
            "post-manual import/validate/score smoke complete",
            "READY_ONLY"
            if read_json(post_manual_smoke_dir / "post_manual_smoke_report.json").get("status") == "PASS_SMOKE_ONLY"
            and exists(post_manual_smoke_dir / "post_manual_smoke_report.md")
            else "FAIL",
            post_manual_smoke_dir / "post_manual_smoke_report.md",
            f"status={read_json(post_manual_smoke_dir / 'post_manual_smoke_report.json').get('status', 'NA')}; report_exists={exists(post_manual_smoke_dir / 'post_manual_smoke_report.md')}",
            "rerun after changing importer, validator, scorer, or strict output schema",
            False,
        ),
        row(
            "extractor_validation",
            "post-manual pipeline guard blocks incomplete manual labels",
            "READY_ONLY"
            if read_json(post_manual_guard_dir / "post_manual_pipeline_guard.json").get("status")
            == "PASS_BLOCKED_BY_MANUAL_VALIDATION"
            and exists(post_manual_guard_dir / "post_manual_pipeline_guard.md")
            else "FAIL",
            post_manual_guard_dir / "post_manual_pipeline_guard.md",
            f"status={read_json(post_manual_guard_dir / 'post_manual_pipeline_guard.json').get('status', 'NA')}; report_exists={exists(post_manual_guard_dir / 'post_manual_pipeline_guard.md')}",
            "rerun before handoff or after editing the post-manual pipeline",
            False,
        ),
        row(
            "extractor_validation",
            "manual annotation validator PASS",
            "PASS" if manual_pass else "MISSING",
            extractor_dir / "manual_annotation_validation_report.json",
            f"status={manual_report.get('status', 'NA')}; completed_slot_labels={manual_report.get('completed_slot_labels', 'NA')}/{manual_report.get('required_slot_labels', 'NA')}",
            "fill manual annotations and rerun validate-manual",
            True,
        ),
        row(
            "extractor_validation",
            "official extractor validation score files produced",
            "PASS" if score_exists else "MISSING",
            extractor_dir / "extractor_validation_metrics.csv",
            f"metrics_exists={exists(extractor_dir / 'extractor_validation_metrics.csv')}; joined_exists={exists(extractor_dir / 'extractor_joined_annotations.csv')}",
            "run score after manual validator PASS",
            True,
        ),
        row(
            "quadrant",
            "unknown/error/entropy quadrant report complete",
            "PASS"
            if exists(quadrant_dir / "quadrant_report.md") and exists(quadrant_dir / "concept_route_metrics.csv")
            else "FAIL",
            quadrant_dir / "quadrant_report.md",
            f"report_exists={exists(quadrant_dir / 'quadrant_report.md')}; metrics_exists={exists(quadrant_dir / 'concept_route_metrics.csv')}",
            "rerun quadrant analysis if strict outputs change",
            True,
        ),
        row(
            "bootstrap",
            "concept-level bootstrap stability complete",
            "PASS"
            if exists(bootstrap_dir / "bootstrap_report.md") and exists(bootstrap_dir / "bootstrap_summary.csv")
            else "FAIL",
            bootstrap_dir / "bootstrap_report.md",
            f"report_exists={exists(bootstrap_dir / 'bootstrap_report.md')}; summary_exists={exists(bootstrap_dir / 'bootstrap_summary.csv')}",
            "rerun after extractor validation score or expanded sample size",
            True,
        ),
        row(
            "sample_size",
            "n=5/n=10 sample-size sensitivity complete",
            "PASS" if exists(sample_size_dir / "sample_size_sensitivity_report.md") else "FAIL",
            sample_size_dir / "sample_size_sensitivity_report.md",
            f"report_exists={exists(sample_size_dir / 'sample_size_sensitivity_report.md')}",
            "only expand to n=20/30 after explicit runtime plan",
            False,
        ),
        row(
            "robustness",
            "guarded full option-order launcher ready",
            "READY_ONLY" if option_full_ready else "FAIL",
            "; ".join(str(directory / "launchers") for directory in option_full_dirs),
            f"launcher_exists_any={option_full_ready}",
            "run after manual validation PASS unless completed fallback evidence is already present",
            False,
        ),
        row(
            "robustness",
            "guarded full prompt-template launcher ready",
            "READY_ONLY" if prompt_full_ready else "FAIL",
            "; ".join(str(directory / "launchers") for directory in prompt_full_dirs),
            f"launcher_exists_any={prompt_full_ready}",
            "run after manual validation PASS unless completed fallback evidence is already present",
            False,
        ),
        row(
            "robustness",
            "full robustness guard preflight PASS",
            "READY_ONLY"
            if robustness_preflight.get("status") in {"PASS_GUARDED", "READY_TO_RUN_AFTER_MANUAL_PASS"}
            and exists(robustness_preflight_dir / "robustness_preflight_report.md")
            else "FAIL",
            robustness_preflight_dir / "robustness_preflight_report.md",
            f"status={robustness_preflight.get('status', 'NA')}; report_exists={exists(robustness_preflight_dir / 'robustness_preflight_report.md')}",
            "rerun before full robustness launch or after editing launcher scripts",
            False,
        ),
        row(
            "robustness",
            "full robustness GPU availability guard smoke PASS",
            "READY_ONLY"
            if full_robustness_gpu_guard_smoke.get("status") == "PASS"
            and exists(full_robustness_gpu_guard_smoke_dir / "full_robustness_gpu_guard_smoke_report.md")
            else "FAIL",
            full_robustness_gpu_guard_smoke_dir / "full_robustness_gpu_guard_smoke_report.md",
            f"gate_status={full_robustness_gpu_guard_smoke.get('gate_status', 'NA')}; gpu_ok={full_robustness_gpu_guard_smoke.get('gpu_availability_ok_for_launch', 'NA')}; worker_outputs={full_robustness_gpu_guard_smoke.get('worker_outputs_present', 'NA')}",
            "rerun after editing full robustness preflight, launcher generation, or returned-manual gate pipeline branch",
            False,
        ),
        row(
            "robustness",
            "full robustness positive dry-run smoke PASS",
            "READY_ONLY"
            if full_robustness_dry_run_smoke.get("status") == "PASS"
            and exists(full_robustness_dry_run_smoke_dir / "full_robustness_dry_run_smoke_report.md")
            else "FAIL",
            full_robustness_dry_run_smoke_dir / "full_robustness_dry_run_smoke_report.md",
            f"gate_status={full_robustness_dry_run_smoke.get('gate_status', 'NA')}; dry_run={full_robustness_dry_run_smoke.get('dry_run_status', 'NA')}; worker_outputs={full_robustness_dry_run_smoke.get('worker_outputs_present', 'NA')}",
            "rerun after editing dry-run launch path, full robustness launcher generation, or returned-manual gate pipeline branch",
            False,
        ),
        row(
            "robustness",
            "full robustness launchers cover all detected GPUs",
            "PASS" if robustness_preflight.get("launcher_gpu_coverage_ok") is True else "FAIL",
            robustness_preflight_dir / "robustness_preflight_report.md",
            f"detected={robustness_preflight.get('detected_gpu_indices', 'NA')}; option={robustness_preflight.get('option_covered_gpu_indices', 'NA')}; prompt={robustness_preflight.get('prompt_covered_gpu_indices', 'NA')}",
            "regenerate full robustness launchers with --gpu-pairs auto if coverage fails",
            not any_full_robustness,
        ),
        row(
            "robustness",
            "all detected GPUs available for full robustness launch",
            "PASS"
            if robustness_preflight.get("gpu_availability_ok_for_launch") is True
            else ("READY_ONLY" if any_full_robustness else "MISSING"),
            robustness_preflight_dir / "robustness_preflight_report.md",
            f"busy_gpus={robustness_preflight.get('busy_gpu_indices', 'NA')}; detected={robustness_preflight.get('detected_gpu_indices', 'NA')}",
            "rerun only before launching another full robustness family",
            not any_full_robustness,
        ),
        row(
            "robustness",
            "lb-gzs all-GPU launch runbook generated",
            "READY_ONLY"
            if all_gpu_runbook.get("status")
            in {"BLOCKED_MANUAL_VALIDATION", "READY_TO_WAIT_FOR_GPUS", "READY_TO_LAUNCH_NOW"}
            and exists(all_gpu_runbook_dir / "all_gpu_execution_runbook.md")
            else "FAIL",
            all_gpu_runbook_dir / "all_gpu_execution_runbook.md",
            f"status={all_gpu_runbook.get('status', 'NA')}; launch_permitted={all_gpu_runbook.get('launch_permitted', 'NA')}; immediate_ready={all_gpu_runbook.get('immediate_launch_ready', 'NA')}",
            "refresh after any launcher, manual-validation, or GPU-state change",
            False,
        ),
        row(
            "robustness",
            "at least one full robustness family complete",
            "PASS" if any_full_robustness else "MISSING",
            f"option={option_full_evidence}; prompt={prompt_full_evidence}",
            f"option_full_complete={option_full_complete}; {option_full_observed}; prompt_full_complete={prompt_full_complete}; {prompt_full_observed}",
            "run one full robustness family after manual validation PASS",
            True,
        ),
        row(
            "object_binding",
            "object-binding retrospective candidate index complete",
            "PASS" if exists(object_binding_dir / "object_binding_sanity_report.md") else "FAIL",
            object_binding_dir / "object_binding_sanity_report.md",
            f"report_exists={exists(object_binding_dir / 'object_binding_sanity_report.md')}",
            "use for targeted prompts only; not final robustness",
            False,
        ),
        row(
            "object_binding",
            "targeted object-binding question package scored",
            "PASS" if targeted_binding_pass else "READY_ONLY"
            if exists(object_binding_dir / "targeted_questions" / "targeted_binding_questions.csv")
            and exists(object_binding_dir / "targeted_questions" / "targeted_binding_score_report.md")
            else "FAIL",
            object_binding_dir / "targeted_questions",
            f"question_lines={line_count(object_binding_dir / 'targeted_questions' / 'targeted_binding_questions.csv')}; metrics_lines={line_count(object_binding_dir / 'targeted_questions' / 'targeted_binding_metrics.csv')}; score_status={targeted_binding_score.get('status', 'NA')}; completed_answers={targeted_binding_score.get('completed_answers', 'NA')}/{targeted_binding_score.get('question_rows', 'NA')}",
            "rerun scorer after changing targeted questions or outputs",
            False,
        ),
        row(
            "mode_collapse",
            "exact/perceptual-hash duplicate check complete",
            "PASS" if exists(image_duplicate_dir / "image_duplicate_mode_collapse_report.md") else "FAIL",
            image_duplicate_dir / "image_duplicate_mode_collapse_report.md",
            f"report_exists={exists(image_duplicate_dir / 'image_duplicate_mode_collapse_report.md')}",
            "treat as pixel-hash evidence only",
            False,
        ),
        row(
            "mode_collapse",
            "offline image-embedding fallback duplicate check complete",
            "PASS" if exists(image_embedding_dir / "image_embedding_duplicate_report.md") else "FAIL",
            image_embedding_dir / "image_embedding_duplicate_report.md",
            f"backend={embedding_preflight.get('embedding_backend', 'NA')}; pairwise_lines={line_count(image_embedding_dir / 'generated_embedding_pairwise_similarity.csv')}; high_similarity_lines={line_count(image_embedding_dir / 'generated_embedding_high_similarity_cases.csv')}",
            "do not claim CLIP/SigLIP/DINO semantic diversity from fallback",
            False,
        ),
        row(
            "mode_collapse",
            "CLIP/SigLIP/DINO semantic embedding duplicate check complete",
            "BLOCKED",
            image_embedding_dir / "embedding_preflight.json",
            f"semantic_embedding_model_available={embedding_preflight.get('semantic_embedding_model_available', 'NA')}; incomplete_cache_candidates={embedding_preflight.get('incomplete_cache_candidates', [])}",
            "provide complete local model or explicitly approve external fetch/use",
            False,
        ),
        row(
            "alternative_extractor",
            "alternative extractor feasibility audited",
            "PASS" if exists(alternative_dir / "alternative_extractor_runbook.md") else "FAIL",
            alternative_dir / "alternative_extractor_runbook.md",
            f"feasibility_status={alt_feasibility.get('status', 'NA')}; complete_local_candidates={alt_feasibility.get('complete_local_candidates', 'NA')}; external_api_credentials_configured={alt_feasibility.get('external_api_credentials_configured', 'NA')}",
            "choose complete local model or approve external service before running",
            False,
        ),
        row(
            "alternative_extractor",
            "alternative extractor scorer ready",
            "READY_ONLY" if exists(alternative_dir / "alternative_extractor_score_report.md") else "FAIL",
            alternative_dir / "alternative_extractor_score_report.md",
            f"template_lines={line_count(alternative_dir / 'alternative_extractor_outputs_template.csv')}; metrics_lines={line_count(alternative_dir / 'alternative_extractor_metrics.csv')}; score_status={alt_score.get('status', 'NA')}; completed_slot_labels={alt_score.get('completed_slot_labels', 'NA')}/{alt_score.get('required_slot_labels', 'NA')}",
            "fill outputs from approved extractor and rerun scorer",
            False,
        ),
        row(
            "alternative_extractor",
            "alternative extractor robustness complete",
            "BLOCKED",
            alternative_dir / "alternative_extractor_feasibility.json",
            f"status={alt_feasibility.get('status', 'NA')}; missing_critical_modules={alt_feasibility.get('missing_critical_modules', [])}",
            "blocked until complete local alternative model or approved external service is available",
            False,
        ),
        row(
            "claim_boundary",
            "static claim-boundary audit generated",
            "PASS" if exists(claim_dir / "claim_boundary_audit_report.md") else "FAIL",
            claim_dir / "claim_boundary_audit_report.md",
            f"report_exists={exists(claim_dir / 'claim_boundary_audit_report.md')}",
            "rerun after report/doc edits",
            True,
        ),
        row(
            "reporting",
            "stage report completeness audit generated",
            "READY_ONLY"
            if exists(stage_report_audit_dir / "stage_report_audit.md")
            and exists(stage_report_audit_dir / "stage_report_audit.csv")
            and exists(stage_report_audit_dir / "stage_report_audit.json")
            else "FAIL",
            stage_report_audit_dir / "stage_report_audit.md",
            f"status={read_json(stage_report_audit_dir / 'stage_report_audit.json').get('status', 'NA')}; missing_reports={read_json(stage_report_audit_dir / 'stage_report_audit.json').get('reports_with_missing_sections', 'NA')}; dod_critical_missing={read_json(stage_report_audit_dir / 'stage_report_audit.json').get('dod_critical_missing_sections', 'NA')}",
            "fill missing required report sections before paper drafting",
            False,
        ),
        row(
            "paper_tables",
            "preliminary paper tables and evidence index generated",
            "PASS" if exists(paper_dir / "paper_tables_preliminary.md") and exists(paper_dir / "evidence_index.md") else "FAIL",
            paper_dir / "paper_tables_preliminary.md",
            f"paper_tables_exists={exists(paper_dir / 'paper_tables_preliminary.md')}; evidence_index_exists={exists(paper_dir / 'evidence_index.md')}",
            "regenerate after manual score or full robustness",
            False,
        ),
        row(
            "definition_of_done",
            "minimum paper-draft DoD achieved",
            "PASS" if minimum_dod else "MISSING",
            "docs/semantic_entropy_i2t_t2i_execution_plan.md#definition-of-done",
            f"strict_ok={strict_ok}; manual_pass={manual_pass}; extractor_report={extractor_report_exists}; manual_score_exists={score_exists}; quadrant={exists(quadrant_dir / 'quadrant_report.md')}; bootstrap={exists(bootstrap_dir / 'bootstrap_report.md')}; any_full_robustness={any_full_robustness}; claim_boundary={exists(claim_dir / 'claim_boundary_audit_report.md')}",
            "preserve claim boundary and use generated paper tables/failure appendix for drafting" if minimum_dod else "complete manual annotations, score extractor validation, then run full robustness",
            True,
        ),
        row(
            "claim_boundary",
            "final empirical claim upgraded beyond controlled protocol comparability",
            "PASS" if minimum_dod else "NOT_ALLOWED",
            "docs/semantic_entropy_i2t_t2i_execution_plan.md#claim-boundary",
            "DoD PASS and claim-boundary audit present" if minimum_dod else "manual validation and full robustness are not complete",
            "keep raw text/image entropy and unified internal entropy claims forbidden",
            False,
        ),
    ]
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: list[dict[str, str]]) -> None:
    counts = status_counts(rows)
    blockers = [item for item in rows if item["blocks_goal"] == "true" and item["status"] != "PASS"]
    payload = {
        "status": "PAPER_DRAFT_DOD_PASS" if not blockers else "ACTIVE_NOT_COMPLETE",
        "summary": counts,
        "blocking_goal_items": blockers,
        "rows": rows,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def md_table(rows: list[dict[str, str]], fields: list[str]) -> list[str]:
    if not rows:
        return ["(none)"]
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for item in rows:
        safe = {field: str(item.get(field, "")).replace("|", "\\|").replace("\n", " ") for field in fields}
        lines.append("| " + " | ".join(safe[field] for field in fields) + " |")
    return lines


def write_report(path: Path, rows: list[dict[str, str]]) -> None:
    counts = status_counts(rows)
    blockers = [item for item in rows if item["blocks_goal"] == "true" and item["status"] != "PASS"]
    incomplete = [item for item in rows if item["status"] in {"MISSING", "FAIL", "BLOCKED", "NOT_ALLOWED"}]
    report_status = "PAPER_DRAFT_DOD_PASS" if not blockers else "ACTIVE_NOT_COMPLETE"
    dod_pass = any(
        item["category"] == "definition_of_done"
        and item["requirement"] == "minimum paper-draft DoD achieved"
        and item["status"] == "PASS"
        for item in rows
    )
    fields = ["category", "requirement", "status", "evidence", "observed", "next_step", "blocks_goal"]
    lines = [
        "# Semantic Entropy Requirement Audit",
        "",
        f"Status: {report_status}",
        "",
        "This audit maps the execution plan and Definition of Done to current evidence. It is a traceability report, not a scientific result.",
        "",
        "## Summary",
        "",
        f"- PASS: {counts.get('PASS', 0)}",
        f"- READY_ONLY: {counts.get('READY_ONLY', 0)}",
        f"- MISSING: {counts.get('MISSING', 0)}",
        f"- BLOCKED: {counts.get('BLOCKED', 0)}",
        f"- NOT_ALLOWED: {counts.get('NOT_ALLOWED', 0)}",
        f"- FAIL: {counts.get('FAIL', 0)}",
        "",
        "## Blocking Goal Items",
        "",
        *md_table(blockers, fields),
        "",
        "## Incomplete Or Constrained Items",
        "",
        *md_table(incomplete, fields),
        "",
        "## Full Requirement Matrix",
        "",
        *md_table(rows, fields),
        "",
        "## Claim Allowed Now",
        "",
        (
            "DoD evidence supports the controlled semantic-state route-level entropy claim, subject to the report's extractor-validation and robustness caveats."
            if dod_pass
            else "Current progress supports only controlled semantic-state protocol comparability with extractor-limited summaries."
        ),
        "",
        "## Claim Still Not Allowed",
        "",
        "Do not claim direct text/image entropy comparability or a unified internal entropy space without hidden-state, causal-intervention, or path-level evidence.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows(args)
    write_csv(out_dir / "requirement_audit.csv", rows)
    write_json(out_dir / "requirement_audit.json", rows)
    write_report(out_dir / "requirement_audit.md", rows)
    print(f"[INFO] wrote requirement audit under {out_dir}")


if __name__ == "__main__":
    main()
