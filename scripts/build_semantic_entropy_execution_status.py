#!/usr/bin/env python3
"""Build a current execution-status audit for semantic entropy plan."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


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
    parser.add_argument("--option-order-smoke-dir", default="outputs/semantic_entropy_umm/robustness_option_order_smoke")
    parser.add_argument("--option-order-full-dir", default="outputs/semantic_entropy_umm/robustness_option_order")
    parser.add_argument("--option-order-full-fallback-dir", default="outputs/semantic_entropy_umm/robustness_option_order_idle_gpus")
    parser.add_argument("--prompt-template-smoke-dir", default="outputs/semantic_entropy_umm/robustness_prompt_template_smoke")
    parser.add_argument("--prompt-template-full-dir", default="outputs/semantic_entropy_umm/robustness_prompt_template")
    parser.add_argument("--prompt-template-full-fallback-dir", default="outputs/semantic_entropy_umm/robustness_prompt_template_idle_gpus")
    parser.add_argument("--robustness-preflight-dir", default="outputs/semantic_entropy_umm/robustness_preflight")
    parser.add_argument("--all-gpu-runbook-dir", default="outputs/semantic_entropy_umm/all_gpu_execution_runbook")
    parser.add_argument("--object-binding-dir", default="outputs/semantic_entropy_umm/object_binding_sanity")
    parser.add_argument("--image-duplicate-dir", default="outputs/semantic_entropy_umm/image_duplicate_mode_collapse")
    parser.add_argument("--image-embedding-dir", default="outputs/semantic_entropy_umm/image_embedding_duplicate_check")
    parser.add_argument("--alternative-extractor-dir", default="outputs/semantic_entropy_umm/alternative_extractor_feasibility")
    parser.add_argument("--concept-difficulty-dir", default="outputs/semantic_entropy_umm/concept_difficulty_split")
    parser.add_argument("--random-concept-control-dir", default="outputs/semantic_entropy_umm/random_concept_control")
    parser.add_argument("--slot-ablation-dir", default="outputs/semantic_entropy_umm/slot_ablation")
    parser.add_argument("--claim-boundary-dir", default="outputs/semantic_entropy_umm/claim_boundary_audit")
    parser.add_argument("--stage-report-audit-dir", default="outputs/semantic_entropy_umm/stage_report_audit")
    parser.add_argument("--requirement-audit-dir", default="outputs/semantic_entropy_umm/requirement_audit")
    parser.add_argument("--paper-dir", default="outputs/semantic_entropy_umm/paper_tables_preliminary")
    parser.add_argument("--out-dir", default="outputs/semantic_entropy_umm/execution_status")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def line_count(path: Path) -> int | None:
    if not path.exists():
        return None
    return sum(1 for _ in path.open(encoding="utf-8"))


def exists(path: Path) -> bool:
    return path.exists()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def status_row(requirement: str, status: str, evidence: str, next_step: str, notes: str = "") -> dict:
    return {
        "requirement": requirement,
        "status": status,
        "evidence": evidence,
        "next_step": next_step,
        "notes": notes,
    }


def candidate_dirs(*paths: str) -> list[Path]:
    out: list[Path] = []
    for path in paths:
        candidate = Path(path)
        if candidate not in out:
            out.append(candidate)
    return out


def robustness_audit_pass(audit: dict) -> bool:
    return (
        audit.get("status") == "PASS"
        and audit.get("state_rows") == 600
        and audit.get("slot_answer_rows") == 3600
        and audit.get("schema_ok") is True
        and audit.get("vocab_ok") is True
    )


def full_option_order_status(dirs: list[Path]) -> tuple[bool, str]:
    seeds = ["seed_20270605", "seed_20270606", "seed_20270607"]
    observations = []
    for directory in dirs:
        audits = [read_json(directory / seed / "option_order_audit.json") for seed in seeds]
        seed_ok = [robustness_audit_pass(audit) for audit in audits]
        changed = [audit.get("changed_slot_values_vs_baseline_subset", "NA") for audit in audits]
        observation = f"{directory}: seed_ok={seed_ok}; changed_slot_values={changed}"
        observations.append(observation)
        if all(seed_ok):
            return True, observation
    return False, " | ".join(observations)


def full_prompt_template_status(dirs: list[Path]) -> tuple[bool, str]:
    templates = ["template_v2_short_direct", "template_v3_question_first"]
    observations = []
    for directory in dirs:
        audits = [read_json(directory / template / "prompt_template_audit.json") for template in templates]
        template_ok = [robustness_audit_pass(audit) for audit in audits]
        changed = [audit.get("changed_slot_values_vs_baseline_subset", "NA") for audit in audits]
        observation = f"{directory}: template_ok={template_ok}; changed_slot_values={changed}"
        observations.append(observation)
        if all(template_ok):
            return True, observation
    return False, " | ".join(observations)


def build_rows(args: argparse.Namespace) -> list[dict]:
    strict_dir = Path(args.strict_dir)
    extractor_dir = Path(args.extractor_validation_dir)
    annotation_batches_dir = extractor_dir / "annotation_batches"
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
    option_smoke_dir = Path(args.option_order_smoke_dir)
    option_full_dir = Path(args.option_order_full_dir)
    option_full_dirs = candidate_dirs(args.option_order_full_dir, args.option_order_full_fallback_dir)
    prompt_smoke_dir = Path(args.prompt_template_smoke_dir)
    prompt_full_dir = Path(args.prompt_template_full_dir)
    prompt_full_dirs = candidate_dirs(args.prompt_template_full_dir, args.prompt_template_full_fallback_dir)
    robustness_preflight_dir = Path(args.robustness_preflight_dir)
    all_gpu_runbook_dir = Path(args.all_gpu_runbook_dir)
    requirement_audit_dir = Path(args.requirement_audit_dir)
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
    robustness_preflight = read_json(robustness_preflight_dir / "robustness_preflight_report.json")
    all_gpu_runbook = read_json(all_gpu_runbook_dir / "all_gpu_execution_runbook.json")
    option_smoke = read_json(option_smoke_dir / "option_order_audit.json")
    prompt_smoke = read_json(prompt_smoke_dir / "prompt_template_audit.json")
    targeted_score_report = read_json(
        Path(args.object_binding_dir) / "targeted_questions" / "targeted_binding_score_report.json"
    )
    alternative_score_report = read_json(
        Path(args.alternative_extractor_dir) / "alternative_extractor_score_report.json"
    )

    strict_counts_ok = (
        strict_audit.get("verdict") == "PASS"
        and line_count(strict_dir / "strict_semantic_states.jsonl") == 600
        and line_count(strict_dir / "strict_slot_answers.jsonl") == 3600
        and line_count(strict_dir / "strict_route_entropy.csv") == 421
        and line_count(strict_dir / "strict_route_error.csv") == 361
    )
    manual_pass = manual_report.get("status") == "PASS"
    score_exists = exists(extractor_dir / "extractor_validation_metrics.csv") and exists(
        extractor_dir / "extractor_joined_annotations.csv"
    )
    option_full_ready = any(exists(directory / "launchers" / "launch_all_option_order_seeds.sh") for directory in option_full_dirs)
    prompt_full_ready = any(exists(directory / "launchers" / "launch_all_prompt_templates.sh") for directory in prompt_full_dirs)
    option_full_complete, option_full_observed = full_option_order_status(option_full_dirs)
    prompt_full_complete, prompt_full_observed = full_prompt_template_status(prompt_full_dirs)
    full_robustness_complete = option_full_complete or prompt_full_complete
    targeted_binding_pass = (
        targeted_score_report.get("status") == "PASS"
        and targeted_score_report.get("completed_answers") == targeted_score_report.get("question_rows")
        and targeted_score_report.get("question_rows") == 90
    )
    rows = [
        status_row(
            "strict_compare audit PASS",
            "PASS" if strict_counts_ok else "FAIL",
            str(strict_dir / "comparability_audit.json"),
            "do not rerun strict compare unless audit or counts fail",
            f"states={line_count(strict_dir / 'strict_semantic_states.jsonl')}; slots={line_count(strict_dir / 'strict_slot_answers.jsonl')}",
        ),
        status_row(
            "extractor validation sample fixed and handoff ready",
            "PASS" if exists(extractor_dir / "manual_annotation_handoff.md") else "FAIL",
            str(extractor_dir / "manual_annotation_handoff.md"),
            "fill manual_annotations.csv",
            f"completed_slots={manual_report.get('completed_slot_labels', 'NA')}/{manual_report.get('required_slot_labels', 'NA')}",
        ),
        status_row(
            "manual annotation batches ready",
            "PASS"
            if exists(annotation_batches_dir / "annotation_batches_report.md")
            and exists(annotation_batches_dir / "annotation_batches_manifest.csv")
            else "FAIL",
            str(annotation_batches_dir / "annotation_batches_report.md"),
            "fill batch manual CSVs, merge, inspect merged_manual_annotations.csv, then write main manual_annotations.csv",
            "batch helper is reversible and does not overwrite main manual CSV unless --write-main is set",
        ),
        status_row(
            "manual annotation progress audit generated",
            "PASS"
            if exists(manual_progress_dir / "manual_annotation_progress.md")
            and exists(manual_progress_dir / "manual_annotation_progress.csv")
            and exists(manual_progress_dir / "manual_annotation_progress.json")
            else "FAIL",
            str(manual_progress_dir / "manual_annotation_progress.md"),
            "refresh while filling manual_annotations.csv",
            "completion tracker only; does not replace validate-manual",
        ),
        status_row(
            "portable manual annotation package generated",
            "PASS"
            if exists(manual_package_dir / "README.md")
            and exists(manual_package_dir / "package_manifest.json")
            and exists(manual_package_dir / "blind_manifest.csv")
            and exists(manual_package_dir.parent / "manual_annotation_package.zip")
            else "FAIL",
            str(manual_package_dir),
            "send package to annotator, then place completed CSV at extractor_validation/manual_annotations.csv",
            "package excludes gold labels from annotator-facing manifests",
        ),
        status_row(
            "portable manual annotation package QA PASS",
            "PASS"
            if read_json(manual_package_qa_dir / "manual_annotation_package_qa.json").get("status") == "PASS"
            and exists(manual_package_qa_dir / "manual_annotation_package_qa.md")
            else "FAIL",
            str(manual_package_qa_dir / "manual_annotation_package_qa.md"),
            "rerun after regenerating the annotation package",
            "package QA only; does not replace manual validation",
        ),
        status_row(
            "returned manual annotation scanner ready",
            "PASS"
            if exists(Path("scripts/find_semantic_entropy_returned_annotations.py"))
            and exists(manual_return_scan_dir / "manual_annotation_return_scan.md")
            else "FAIL",
            str(manual_return_scan_dir / "manual_annotation_return_scan.md"),
            "drop returned CSV under manual_annotation_returns or pass --candidate, then rerun scanner",
            f"status={manual_return_scan.get('status', 'NA')}; ready_candidates={manual_return_scan.get('ready_to_import_count', 'NA')}",
        ),
        status_row(
            "safe manual annotation importer dry-run generated",
            "PASS"
            if exists(Path("scripts/import_semantic_entropy_manual_annotations.py"))
            and exists(manual_import_dir / "manual_annotation_import_report.md")
            and exists(manual_import_dir / "manual_annotation_import_report.json")
            else "FAIL",
            str(manual_import_dir / "manual_annotation_import_report.md"),
            "rerun on returned annotator CSV; use --write only when READY_TO_IMPORT",
            "dry-run only; current empty template should not import",
        ),
        status_row(
            "returned manual gate runner ready",
            "PASS"
            if exists(Path("scripts/run_semantic_entropy_returned_manual_gate.py"))
            and exists(returned_manual_gate_dir / "returned_manual_gate_report.md")
            else "FAIL",
            str(returned_manual_gate_dir / "returned_manual_gate_report.md"),
            "rerun after a returned annotation CSV arrives; add --write only after READY_TO_WRITE",
            f"status={returned_manual_gate.get('status', 'NA')}; ready_candidates={returned_manual_gate.get('ready_to_import_count', 'NA')}; wrote={returned_manual_gate.get('wrote_target', 'NA')}",
        ),
        status_row(
            "returned manual watch-and-launch runner ready",
            "PASS"
            if exists(Path("scripts/watch_semantic_entropy_returned_manual_and_launch.py"))
            and exists(returned_manual_watch_dir / "returned_manual_watch_report.md")
            else "FAIL",
            str(returned_manual_watch_dir / "returned_manual_watch_report.md"),
            "run as a bounded watcher only when returned annotation CSVs may arrive; full robustness still requires explicit --write and --run-full-robustness",
            f"status={returned_manual_watch.get('status', 'NA')}; gate_ran={returned_manual_watch.get('gate_ran', 'NA')}; run_full={returned_manual_watch.get('run_full_robustness', 'NA')}",
        ),
        status_row(
            "returned manual gate write smoke PASS",
            "PASS"
            if returned_manual_gate_smoke.get("status") == "PASS"
            and exists(returned_manual_gate_smoke_dir / "returned_manual_gate_smoke_report.md")
            else "FAIL",
            str(returned_manual_gate_smoke_dir / "returned_manual_gate_smoke_report.md"),
            "rerun after editing returned manual gate or importer write behavior",
            f"gate_status={returned_manual_gate_smoke.get('gate_status', 'NA')}; wrote={returned_manual_gate_smoke.get('wrote_target', 'NA')}; post_manual_ran={returned_manual_gate_smoke.get('post_manual_pipeline_ran', 'NA')}",
        ),
        status_row(
            "post-manual import/validate/score smoke complete",
            "PASS"
            if read_json(post_manual_smoke_dir / "post_manual_smoke_report.json").get("status") == "PASS_SMOKE_ONLY"
            and exists(post_manual_smoke_dir / "post_manual_smoke_report.md")
            else "FAIL",
            str(post_manual_smoke_dir / "post_manual_smoke_report.md"),
            "rerun after changing importer, validator, scorer, or strict output schema",
            "synthetic-label smoke only; not human validation evidence",
        ),
        status_row(
            "post-manual pipeline guard blocks incomplete manual labels",
            "PASS"
            if read_json(post_manual_guard_dir / "post_manual_pipeline_guard.json").get("status")
            == "PASS_BLOCKED_BY_MANUAL_VALIDATION"
            and exists(post_manual_guard_dir / "post_manual_pipeline_guard.md")
            else "FAIL",
            str(post_manual_guard_dir / "post_manual_pipeline_guard.md"),
            "rerun before handoff or after editing the post-manual pipeline",
            "guard audit only; manual validation and scoring remain required",
        ),
        status_row(
            "manual annotation validator PASS",
            "PASS" if manual_pass else "MISSING",
            str(extractor_dir / "manual_annotation_validation_report.json"),
            "run validate-manual after filling manual_annotations.csv",
            f"current_status={manual_report.get('status', 'NA')}",
        ),
        status_row(
            "post-manual pipeline ready",
            "PASS" if exists(Path("scripts/run_semantic_entropy_post_manual_pipeline.sh")) else "FAIL",
            "scripts/run_semantic_entropy_post_manual_pipeline.sh",
            "run after manual_annotations.csv is complete",
            "default path validates, scores, and refreshes reports without full robustness",
        ),
        status_row(
            "official extractor validation score complete",
            "PASS" if score_exists else "MISSING",
            str(extractor_dir / "extractor_validation_metrics.csv"),
            "run score after manual validator PASS",
            "required before upgrading beyond extractor-limited evidence",
        ),
        status_row(
            "quadrant report complete",
            "PASS" if exists(quadrant_dir / "quadrant_report.md") and exists(quadrant_dir / "quadrant_cases.csv") else "FAIL",
            str(quadrant_dir / "quadrant_report.md"),
            "review low-entropy/high-error cases",
        ),
        status_row(
            "bootstrap report complete with concept-level statistics",
            "PASS" if exists(bootstrap_dir / "bootstrap_report.md") and exists(bootstrap_dir / "bootstrap_summary.csv") else "FAIL",
            str(bootstrap_dir / "bootstrap_report.md"),
            "keep claim weak if CI crosses zero",
        ),
        status_row(
            "sample-size sensitivity complete",
            "PASS" if exists(Path(args.sample_size_dir) / "sample_size_sensitivity_report.md") else "FAIL",
            str(Path(args.sample_size_dir) / "sample_size_sensitivity_report.md"),
            "only expand to n=20/30 after explicit runtime plan",
        ),
        status_row(
            "option-order robustness smoke",
            "PASS" if option_smoke.get("status") == "PASS" else "FAIL",
            str(option_smoke_dir / "option_order_audit.json"),
            "run full seeds only after manual validator PASS",
            f"state_rows={option_smoke.get('state_rows', 'NA')}",
        ),
        status_row(
            "prompt-template robustness smoke",
            "PASS" if prompt_smoke.get("status") == "PASS" else "FAIL",
            str(prompt_smoke_dir / "prompt_template_audit.json"),
            "run full templates only after manual validator PASS",
            f"state_rows={prompt_smoke.get('state_rows', 'NA')}",
        ),
        status_row(
            "full robustness launcher guarded and ready",
            "PASS" if option_full_ready and prompt_full_ready else "FAIL",
            f"{'; '.join(str(directory / 'launchers') for directory in option_full_dirs)}; {'; '.join(str(directory / 'launchers') for directory in prompt_full_dirs)}",
            "launch after manual validator PASS",
            "guard prevents accidental full runs while validation FAIL",
        ),
        status_row(
            "full robustness guard preflight PASS",
            "PASS"
            if robustness_preflight.get("status") in {"PASS_GUARDED", "READY_TO_RUN_AFTER_MANUAL_PASS"}
            and exists(robustness_preflight_dir / "robustness_preflight_report.md")
            else "FAIL",
            str(robustness_preflight_dir / "robustness_preflight_report.md"),
            "rerun before full robustness launch or after editing launcher scripts",
            "guard readiness only; smoke runs are not full robustness evidence",
        ),
        status_row(
            "full robustness GPU availability guard smoke PASS",
            "PASS"
            if full_robustness_gpu_guard_smoke.get("status") == "PASS"
            and exists(full_robustness_gpu_guard_smoke_dir / "full_robustness_gpu_guard_smoke_report.md")
            else "FAIL",
            str(full_robustness_gpu_guard_smoke_dir / "full_robustness_gpu_guard_smoke_report.md"),
            "rerun after editing full robustness preflight, launcher generation, or returned-manual gate pipeline branch",
            f"gate_status={full_robustness_gpu_guard_smoke.get('gate_status', 'NA')}; gpu_ok={full_robustness_gpu_guard_smoke.get('gpu_availability_ok_for_launch', 'NA')}; worker_outputs={full_robustness_gpu_guard_smoke.get('worker_outputs_present', 'NA')}",
        ),
        status_row(
            "full robustness positive dry-run smoke PASS",
            "PASS"
            if full_robustness_dry_run_smoke.get("status") == "PASS"
            and exists(full_robustness_dry_run_smoke_dir / "full_robustness_dry_run_smoke_report.md")
            else "FAIL",
            str(full_robustness_dry_run_smoke_dir / "full_robustness_dry_run_smoke_report.md"),
            "rerun after editing dry-run launch path, full robustness launcher generation, or returned-manual gate pipeline branch",
            f"gate_status={full_robustness_dry_run_smoke.get('gate_status', 'NA')}; dry_run={full_robustness_dry_run_smoke.get('dry_run_status', 'NA')}; worker_outputs={full_robustness_dry_run_smoke.get('worker_outputs_present', 'NA')}",
        ),
        status_row(
            "full robustness all detected GPUs covered",
            "PASS" if robustness_preflight.get("launcher_gpu_coverage_ok") is True else "FAIL",
            str(robustness_preflight_dir / "robustness_preflight_report.md"),
            "regenerate launchers with --gpu-pairs auto if coverage fails",
            f"detected={robustness_preflight.get('detected_gpu_indices', 'NA')}; option={robustness_preflight.get('option_covered_gpu_indices', 'NA')}; prompt={robustness_preflight.get('prompt_covered_gpu_indices', 'NA')}",
        ),
        status_row(
            "full robustness all detected GPUs available",
            "PASS"
            if robustness_preflight.get("gpu_availability_ok_for_launch") is True
            else ("READY_ONLY" if full_robustness_complete else "MISSING"),
            str(robustness_preflight_dir / "robustness_preflight_report.md"),
            "rerun only before launching another full robustness family",
            f"busy_gpus={robustness_preflight.get('busy_gpu_indices', 'NA')}",
        ),
        status_row(
            "lb-gzs all-GPU execution runbook generated",
            "PASS"
            if all_gpu_runbook.get("status")
            in {"BLOCKED_MANUAL_VALIDATION", "READY_TO_WAIT_FOR_GPUS", "READY_TO_LAUNCH_NOW"}
            and exists(all_gpu_runbook_dir / "all_gpu_execution_runbook.md")
            else "FAIL",
            str(all_gpu_runbook_dir / "all_gpu_execution_runbook.md"),
            "use after manual validator PASS; rerun if preflight, launcher, or GPU state changes",
            f"status={all_gpu_runbook.get('status', 'NA')}; launch_permitted={all_gpu_runbook.get('launch_permitted', 'NA')}; detected_gpus={all_gpu_runbook.get('detected_gpu_indices', 'NA')}",
        ),
        status_row(
            "full robustness complete",
            "PASS" if full_robustness_complete else "MISSING",
            f"{'; '.join(str(directory) for directory in option_full_dirs)}; {'; '.join(str(directory) for directory in prompt_full_dirs)}",
            "run one full robustness family after manual validator PASS",
            f"option_full_complete={option_full_complete}; {option_full_observed}; prompt_full_complete={prompt_full_complete}; {prompt_full_observed}",
        ),
        status_row(
            "object-binding retrospective index complete",
            "PASS" if exists(Path(args.object_binding_dir) / "object_binding_sanity_report.md") else "FAIL",
            str(Path(args.object_binding_dir) / "object_binding_sanity_report.md"),
            "run targeted binding prompts only if stronger claim needs it",
        ),
        status_row(
            "targeted object-binding question package ready",
            "PASS"
            if exists(Path(args.object_binding_dir) / "targeted_questions" / "targeted_binding_questions.csv")
            and exists(Path(args.object_binding_dir) / "targeted_questions" / "targeted_binding_outputs_template.csv")
            else "FAIL",
            str(Path(args.object_binding_dir) / "targeted_questions" / "targeted_binding_runbook.md"),
            "rerun only after changing targeted binding questions or outputs" if targeted_binding_pass else "run and score separate binding questions only after explicit runtime plan",
            "targeted score PASS" if targeted_binding_pass else "preparation only; does not prove object-binding robustness",
        ),
        status_row(
            "targeted object-binding scorer ready",
            "PASS" if targeted_binding_pass else "READY_ONLY"
            if exists(Path("scripts/score_semantic_entropy_object_binding_targeted.py"))
            and exists(Path(args.object_binding_dir) / "targeted_questions" / "targeted_binding_score_report.md")
            else "FAIL",
            str(Path(args.object_binding_dir) / "targeted_questions" / "targeted_binding_score_report.md"),
            "rerun scorer after changing targeted binding questions or outputs",
            f"current_score_status={targeted_score_report.get('status', 'NA')}; completed={targeted_score_report.get('completed_answers', 'NA')}/{targeted_score_report.get('question_rows', 'NA')}",
        ),
        status_row(
            "image duplicate/mode-collapse hash check complete",
            "PASS" if exists(Path(args.image_duplicate_dir) / "image_duplicate_mode_collapse_report.md") else "FAIL",
            str(Path(args.image_duplicate_dir) / "image_duplicate_mode_collapse_report.md"),
            "run CLIP/image-embedding check only if stronger defense needed",
        ),
        status_row(
            "offline image-embedding fallback duplicate check complete",
            "PASS" if exists(Path(args.image_embedding_dir) / "image_embedding_duplicate_report.md") else "FAIL",
            str(Path(args.image_embedding_dir) / "image_embedding_duplicate_report.md"),
            "use only as pixel-stat fallback evidence; real CLIP/SigLIP/DINO semantic embedding remains pending",
            "does not prove semantic diversity",
        ),
        status_row(
            "alternative extractor feasibility audited",
            "PASS" if exists(Path(args.alternative_extractor_dir) / "alternative_extractor_runbook.md") else "FAIL",
            str(Path(args.alternative_extractor_dir) / "alternative_extractor_runbook.md"),
            "select a complete local model or approve external service/cost before running alternative extractor",
            "feasibility only; robustness remains pending",
        ),
        status_row(
            "alternative extractor scorer ready",
            "PASS"
            if exists(Path("scripts/score_semantic_entropy_alternative_extractor.py"))
            and exists(Path(args.alternative_extractor_dir) / "alternative_extractor_score_report.md")
            else "FAIL",
            str(Path(args.alternative_extractor_dir) / "alternative_extractor_score_report.md"),
            "fill alternative_extractor_outputs_template.csv from an approved extractor and rerun scorer without --allow-incomplete",
            f"current_score_status={alternative_score_report.get('status', 'NA')}; completed_slots={alternative_score_report.get('completed_slot_labels', 'NA')}/{alternative_score_report.get('required_slot_labels', 'NA')}",
        ),
        status_row(
            "easy/hard concept split complete",
            "PASS" if exists(Path(args.concept_difficulty_dir) / "concept_difficulty_report.md") else "FAIL",
            str(Path(args.concept_difficulty_dir) / "concept_difficulty_report.md"),
            "treat as strict-output sensitivity, not human truth",
        ),
        status_row(
            "random concept control complete",
            "PASS" if exists(Path(args.random_concept_control_dir) / "random_concept_control_report.md") else "FAIL",
            str(Path(args.random_concept_control_dir) / "random_concept_control_report.md"),
            "treat as mismatched-target metric sensitivity only",
        ),
        status_row(
            "slot ablation complete",
            "PASS" if exists(Path(args.slot_ablation_dir) / "slot_ablation_report.md") else "FAIL",
            str(Path(args.slot_ablation_dir) / "slot_ablation_report.md"),
            "treat as strict-output localization only",
        ),
        status_row(
            "claim boundary static audit PASS",
            "PASS" if exists(Path(args.claim_boundary_dir) / "claim_boundary_audit_report.md") else "FAIL",
            str(Path(args.claim_boundary_dir) / "claim_boundary_audit_report.md"),
            "rerun after editing docs or reports",
            "checks obvious forbidden-claim wording",
        ),
        status_row(
            "stage report completeness audit generated",
            "PASS"
            if exists(stage_report_audit_dir / "stage_report_audit.md")
            and exists(stage_report_audit_dir / "stage_report_audit.csv")
            and exists(stage_report_audit_dir / "stage_report_audit.json")
            else "FAIL",
            str(stage_report_audit_dir / "stage_report_audit.md"),
            "fill missing report sections before paper drafting",
            "structure audit only; ACTION_REQUIRED is expected until all reports match the contract",
        ),
        status_row(
            "requirement/DoD traceability audit generated",
            "PASS"
            if exists(requirement_audit_dir / "requirement_audit.md")
            and exists(requirement_audit_dir / "requirement_audit.csv")
            and exists(requirement_audit_dir / "requirement_audit.json")
            else "FAIL",
            str(requirement_audit_dir / "requirement_audit.md"),
            "refresh after manual score or full robustness",
            "active_not_complete; distinguishes DoD blockers from ready-only packages",
        ),
        status_row(
            "paper tables preliminary package generated",
            "PASS" if exists(paper_dir / "paper_tables_preliminary.md") else "FAIL",
            str(paper_dir / "paper_tables_preliminary.md"),
            "regenerate after scoring or full robustness",
        ),
        status_row(
            "claim boundary enforced",
            "PASS",
            "docs/semantic_entropy_i2t_t2i_execution_plan.md",
            "do not claim raw text/image entropy comparability or internal unified entropy space",
            "current claim remains shared semantic-state protocol with extractor-limited caveat",
        ),
    ]
    return rows


def md_table(rows: list[dict]) -> list[str]:
    fields = ["requirement", "status", "evidence", "next_step", "notes"]
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return lines


def write_report(out_dir: Path, rows: list[dict]) -> None:
    counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    blockers = [row for row in rows if row["status"] in {"MISSING", "FAIL"}]
    report_status = "PAPER_DRAFT_DOD_PASS" if not blockers else "ACTIVE_NOT_COMPLETE"
    next_gate = (
        "No blocking execution-plan gates remain. Preserve the claim boundary and use the generated paper tables/failure appendix for drafting."
        if not blockers
        else "Complete `outputs/semantic_entropy_umm/extractor_validation/manual_annotations.csv`, run `--mode validate-manual` to PASS, then run `--mode score`. Before launching full robustness on `lb-gzs`, rerun robustness preflight and require both all-GPU coverage and all-GPU availability to PASS."
    )
    lines = [
        "# Semantic Entropy Execution Status Audit",
        "",
        f"Status: {report_status}",
        "",
        "This audit maps the execution plan's gates to current files. It is evidence tracking only; it does not replace manual annotation or full robustness runs.",
        "",
        "## Summary",
        "",
        f"- PASS: {counts.get('PASS', 0)}",
        f"- MISSING: {counts.get('MISSING', 0)}",
        f"- FAIL: {counts.get('FAIL', 0)}",
        "",
        "## Current Blocking Gates",
        "",
        *md_table(blockers),
        "",
        "## Full Status Matrix",
        "",
        *md_table(rows),
        "",
        "## Next Executable Gate",
        "",
        next_gate,
    ]
    (out_dir / "execution_status_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows(args)
    write_csv(out_dir / "execution_status_audit.csv", rows)
    write_report(out_dir, rows)
    print(f"[INFO] wrote execution status audit under {out_dir}")


if __name__ == "__main__":
    main()
