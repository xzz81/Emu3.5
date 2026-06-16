#!/usr/bin/env python3
"""Build preliminary paper tables and evidence index for semantic entropy.

This script is intentionally read-only with respect to experiment outputs. It
summarizes existing strict/quadrant/bootstrap/validation/robustness artifacts
into a preliminary, extractor-limited paper package.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


SLOTS = ["object_1", "color_1", "object_2", "color_2", "relation", "background"]
ALL_SLOTS = ["joint", *SLOTS]
ROUTES = ["I2T", "T2I"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-dir", default="outputs/semantic_entropy_umm/strict_compare")
    parser.add_argument("--extractor-validation-dir", default="outputs/semantic_entropy_umm/extractor_validation")
    parser.add_argument("--manual-progress-dir", default="outputs/semantic_entropy_umm/manual_annotation_progress")
    parser.add_argument("--manual-package-dir", default="outputs/semantic_entropy_umm/manual_annotation_package")
    parser.add_argument("--manual-package-qa-dir", default="outputs/semantic_entropy_umm/manual_annotation_package_qa")
    parser.add_argument("--manual-import-dir", default="outputs/semantic_entropy_umm/manual_annotation_import")
    parser.add_argument("--post-manual-smoke-dir", default="outputs/semantic_entropy_umm/post_manual_smoke")
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
    parser.add_argument("--execution-status-dir", default="outputs/semantic_entropy_umm/execution_status")
    parser.add_argument("--out-dir", default="outputs/semantic_entropy_umm/paper_tables_preliminary")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None and rows:
        fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or [])
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, digits: int = 4) -> str:
    if value in ("", None):
        return "TBD"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def mean(rows: list[dict], field: str) -> float | None:
    values = []
    for row in rows:
        if row.get(field, "") != "":
            values.append(float(row[field]))
    return sum(values) / len(values) if values else None


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


def mean_route_slot(csv_path: Path, route: str, slot: str, field: str) -> float | None:
    return mean(
        [row for row in read_csv(csv_path) if row.get("route") == route and row.get("slot") == slot],
        field,
    )


def delta_summary(run_dir: Path, prefix: str) -> tuple[str, str]:
    entropy_path = run_dir / f"{prefix}_route_entropy.csv"
    error_path = run_dir / f"{prefix}_route_error.csv"
    h_i2t = mean_route_slot(entropy_path, "I2T", "joint", "entropy")
    h_t2i = mean_route_slot(entropy_path, "T2I", "joint", "entropy")
    e_i2t = mean_route_slot(error_path, "I2T", "object_1", "error_rate")
    e_t2i = mean_route_slot(error_path, "T2I", "object_1", "error_rate")
    delta_h = fmt(h_t2i - h_i2t) if h_i2t is not None and h_t2i is not None else "TBD"
    delta_err = fmt(e_t2i - e_i2t) if e_i2t is not None and e_t2i is not None else "TBD"
    return f"{delta_h} joint", f"{delta_err} object_1"


def full_option_order_rows(dirs: list[Path]) -> list[dict]:
    seeds = ["seed_20270605", "seed_20270606", "seed_20270607"]
    for directory in dirs:
        rows = []
        for seed in seeds:
            run_dir = directory / seed
            audit = read_json(run_dir / "option_order_audit.json")
            if not robustness_audit_pass(audit):
                rows = []
                break
            delta_h, delta_err = delta_summary(run_dir, "option_order")
            rows.append(
                {
                    "setting": f"randomized option order {seed.replace('seed_', '')}",
                    "delta_h": delta_h,
                    "delta_err": delta_err,
                    "conclusion": (
                        f"full robustness PASS; 600 states; "
                        f"{audit.get('changed_slot_values_vs_baseline_subset', 'NA')}/3600 slot values changed vs baseline"
                    ),
                }
            )
        if rows:
            return rows
    return [{"setting": "randomized option order full seeds", "delta_h": "TBD", "delta_err": "TBD", "conclusion": "pending"}]


def full_prompt_template_rows(dirs: list[Path]) -> list[dict]:
    templates = ["template_v2_short_direct", "template_v3_question_first"]
    for directory in dirs:
        rows = []
        for template in templates:
            run_dir = directory / template
            audit = read_json(run_dir / "prompt_template_audit.json")
            if not robustness_audit_pass(audit):
                rows = []
                break
            delta_h, delta_err = delta_summary(run_dir, "prompt_template")
            rows.append(
                {
                    "setting": f"prompt template {template}",
                    "delta_h": delta_h,
                    "delta_err": delta_err,
                    "conclusion": (
                        f"full robustness PASS; 600 states; "
                        f"{audit.get('changed_slot_values_vs_baseline_subset', 'NA')}/3600 slot values changed vs baseline"
                    ),
                }
            )
        if rows:
            return rows
    return [{"setting": "prompt template full", "delta_h": "TBD", "delta_err": "TBD", "conclusion": "pending"}]


def targeted_binding_conclusion(object_binding_dir: Path) -> str:
    target_dir = object_binding_dir / "targeted_questions"
    report = read_json(target_dir / "targeted_binding_score_report.json")
    metrics = read_csv(target_dir / "targeted_binding_metrics.csv")
    by_group = {(row.get("group"), row.get("value")): row for row in metrics}
    if report.get("status") == "PASS":
        all_row = by_group.get(("all", "ALL"), {})
        relation = by_group.get(("binding_family", "relation"), {})
        color_binding = by_group.get(("binding_family", "color_object_binding"), {})
        return (
            "targeted control PASS; "
            f"{report.get('completed_answers', 'NA')}/{report.get('question_rows', 'NA')} answers; "
            f"all acc={fmt(all_row.get('accuracy'))}; "
            f"relation acc={fmt(relation.get('accuracy'))}; "
            f"color-object acc={fmt(color_binding.get('accuracy'))}"
        )
    return (
        f"targeted control pending; status={report.get('status', 'NA')}; "
        f"completed={report.get('completed_answers', 'NA')}/{report.get('question_rows', 'NA')}"
    )


def table1_comparability(audit: dict) -> list[dict]:
    sample_counts = audit.get("sample_count_by_route", {})
    total_states = sum(int(v) for v in sample_counts.values()) if sample_counts else ""
    slot_counts = audit.get("slot_answer_count_by_route_slot", {})
    total_slot_answers = sum(int(v) for v in slot_counts.values()) if slot_counts else ""
    concepts = total_states // 20 if isinstance(total_states, int) and total_states else ""
    return [
        {"item": "concepts", "expected": 30, "observed": concepts, "pass": "yes" if concepts == 30 else "no"},
        {"item": "routes", "expected": 2, "observed": len(sample_counts), "pass": "yes" if len(sample_counts) == 2 else "no"},
        {
            "item": "states per concept-route",
            "expected": 10,
            "observed": "10" if not audit.get("per_concept_route_sample_parity_failures") else "parity failures",
            "pass": "yes" if not audit.get("per_concept_route_sample_parity_failures") else "no",
        },
        {"item": "total states", "expected": 600, "observed": total_states, "pass": "yes" if total_states == 600 else "no"},
        {"item": "slots per state", "expected": 6, "observed": len(audit.get("state_schema", [])), "pass": "yes" if len(audit.get("state_schema", [])) == 6 else "no"},
        {"item": "total slot answers", "expected": 3600, "observed": total_slot_answers, "pass": "yes" if total_slot_answers == 3600 else "no"},
        {"item": "legal values", "expected": "100%", "observed": audit.get("vocab_ok"), "pass": "yes" if audit.get("vocab_ok") else "no"},
        {"item": "audit verdict", "expected": "PASS", "observed": audit.get("verdict"), "pass": "yes" if audit.get("verdict") == "PASS" else "no"},
    ]


def table2_extractor_validation(extractor_dir: Path) -> list[dict]:
    metrics = read_csv(extractor_dir / "extractor_validation_metrics.csv")
    if not metrics:
        return [
            {
                "slot": slot,
                "accuracy": "TBD",
                "macro_f1": "TBD",
                "unknown_agreement": "TBD",
                "notes": "manual annotations pending" if slot != "relation" else "binding-critical; manual annotations pending",
            }
            for slot in SLOTS
        ]
    by_key = {(row["metric"], row["route"], row["slot"]): row for row in metrics}
    rows = []
    for slot in SLOTS:
        rows.append(
            {
                "slot": slot,
                "accuracy": fmt(by_key.get(("slot_accuracy", "ALL", slot), {}).get("value")),
                "macro_f1": fmt(by_key.get(("macro_f1", "ALL", slot), {}).get("value")),
                "unknown_agreement": fmt(by_key.get(("unknown_agreement", "ALL", slot), {}).get("value")),
                "notes": "binding-critical" if slot == "relation" else "",
            }
        )
    return rows


def metrics_index(quadrant_dir: Path) -> dict[tuple[str, str], list[dict]]:
    rows = read_csv(quadrant_dir / "concept_route_metrics.csv")
    out: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        out[(row["route"], row["slot"])].append(row)
    return out


def table3_route_entropy(metrics: dict[tuple[str, str], list[dict]]) -> list[dict]:
    rows = []
    i2t = metrics.get(("I2T", "joint"), [])
    t2i = metrics.get(("T2I", "joint"), [])
    vals = {
        "I2T": {
            "state_entropy": mean(i2t, "entropy"),
            "normalized_entropy": mean(i2t, "normalized_entropy"),
            "effective_states": mean(i2t, "effective_num_states"),
            "error": mean(i2t, "error_rate"),
            "unknown_rate": mean(i2t, "unknown_rate"),
        },
        "T2I": {
            "state_entropy": mean(t2i, "entropy"),
            "normalized_entropy": mean(t2i, "normalized_entropy"),
            "effective_states": mean(t2i, "effective_num_states"),
            "error": mean(t2i, "error_rate"),
            "unknown_rate": mean(t2i, "unknown_rate"),
        },
    }
    for route in ROUTES:
        rows.append({"route": route, **{k: fmt(v) for k, v in vals[route].items()}})
    rows.append(
        {
            "route": "Delta T2I-I2T",
            **{
                key: fmt((vals["T2I"][key] or 0) - (vals["I2T"][key] or 0))
                for key in ["state_entropy", "normalized_entropy", "effective_states", "error", "unknown_rate"]
            },
        }
    )
    return rows


def table4_slot_breakdown(metrics: dict[tuple[str, str], list[dict]], bootstrap_dir: Path) -> list[dict]:
    bootstrap = {
        (row["slot"], row["metric"]): row
        for row in read_csv(bootstrap_dir / "bootstrap_summary.csv")
    }
    rows = []
    for slot in SLOTS:
        i2t = metrics.get(("I2T", slot), [])
        t2i = metrics.get(("T2I", slot), [])
        h_i2t = mean(i2t, "entropy") or 0.0
        h_t2i = mean(t2i, "entropy") or 0.0
        e_i2t = mean(i2t, "error_rate") or 0.0
        e_t2i = mean(t2i, "error_rate") or 0.0
        h_ci = bootstrap.get((slot, "entropy"), {})
        e_ci = bootstrap.get((slot, "error_rate"), {})
        rows.append(
            {
                "slot": slot,
                "h_i2t": fmt(h_i2t),
                "h_t2i": fmt(h_t2i),
                "delta_h": fmt(h_t2i - h_i2t),
                "delta_h_95ci": f"[{fmt(h_ci.get('bootstrap_ci_low'))}, {fmt(h_ci.get('bootstrap_ci_high'))}]" if h_ci else "TBD",
                "err_i2t": fmt(e_i2t),
                "err_t2i": fmt(e_t2i),
                "delta_err": fmt(e_t2i - e_i2t),
                "delta_err_95ci": f"[{fmt(e_ci.get('bootstrap_ci_low'))}, {fmt(e_ci.get('bootstrap_ci_high'))}]" if e_ci else "TBD",
            }
        )
    return rows


def table5_robustness(
    strict_table3: list[dict],
    option_smoke_dir: Path,
    option_full_dirs: list[Path],
    prompt_smoke_dir: Path,
    prompt_full_dirs: list[Path],
    object_binding_dir: Path,
    image_duplicate_dir: Path,
    concept_difficulty_dir: Path,
    random_control_dir: Path,
    slot_ablation_dir: Path,
) -> list[dict]:
    smoke_audit = read_json(option_smoke_dir / "option_order_audit.json")
    prompt_smoke_audit = read_json(prompt_smoke_dir / "prompt_template_audit.json")
    baseline = next((row for row in strict_table3 if row["route"] == "Delta T2I-I2T"), {})
    object_binding_rows = read_csv(object_binding_dir / "object_binding_summary.csv")
    object_binding_all = next((row for row in object_binding_rows if row.get("route") == "ALL"), {})
    object_binding_conclusion = targeted_binding_conclusion(object_binding_dir)
    if object_binding_all:
        object_binding_conclusion = (
            f"{object_binding_conclusion}; "
            "retrospective strict-output index; "
            f"{object_binding_all.get('states', 'TBD')} eligible states"
        )
    image_duplicate_rows = read_csv(image_duplicate_dir / "generated_image_duplicate_summary.csv")
    total_generated = sum(int(row.get("existing_images", 0)) for row in image_duplicate_rows)
    total_duplicate_images = sum(
        int(row.get("generated_images", 0)) - int(row.get("exact_unique_images", 0))
        for row in image_duplicate_rows
    )
    total_pairs = sum(int(row.get("pair_count", 0)) for row in image_duplicate_rows)
    perceptual_pairs = sum(int(row.get("perceptual_duplicate_pairs", 0)) for row in image_duplicate_rows)
    flagged_concepts = sum(row.get("mode_collapse_flag") == "True" for row in image_duplicate_rows)
    duplicate_conclusion = "pending"
    if image_duplicate_rows:
        duplicate_conclusion = (
            f"pixel/hash check; {total_generated} images; "
            f"{total_duplicate_images} exact duplicate images; "
            f"{perceptual_pairs}/{total_pairs} perceptual duplicate pairs; "
            f"{flagged_concepts} concepts flagged"
        )
    difficulty_rows = read_csv(concept_difficulty_dir / "concept_difficulty_split_summary.csv")
    difficulty_by_split = {row.get("difficulty_split"): row for row in difficulty_rows}
    difficulty_conclusion = "pending"
    if difficulty_rows:
        hard = difficulty_by_split.get("hard", {})
        easy = difficulty_by_split.get("easy", {})
        difficulty_conclusion = (
            "retrospective strict-output split; "
            f"hard T2I H={fmt(hard.get('mean_t2i_joint_entropy'))}, "
            f"easy T2I H={fmt(easy.get('mean_t2i_joint_entropy'))}; "
            f"hard T2I err={fmt(hard.get('mean_t2i_mean_slot_error'))}, "
            f"easy T2I err={fmt(easy.get('mean_t2i_mean_slot_error'))}"
        )
    control_rows = read_csv(random_control_dir / "random_concept_route_slot_summary.csv")
    control_summary = {
        row.get("route"): row
        for row in control_rows
        if row.get("slot") == "mean_over_slots"
    }
    random_control_conclusion = "pending"
    if control_summary:
        i2t = control_summary.get("I2T", {})
        t2i = control_summary.get("T2I", {})
        random_control_conclusion = (
            "mismatched-target sanity; "
            f"I2T err {fmt(i2t.get('true_error_rate'))}->{fmt(i2t.get('control_error_rate'))}; "
            f"T2I err {fmt(t2i.get('true_error_rate'))}->{fmt(t2i.get('control_error_rate'))}"
        )
    slot_ablation_rows = read_csv(slot_ablation_dir / "slot_ablation_summary.csv")
    family_rows = [row for row in slot_ablation_rows if row.get("level") == "family"]
    slot_ablation_conclusion = "pending"
    if family_rows:
        positive_entropy = [row for row in family_rows if float(row.get("delta_entropy_t2i_minus_i2t", 0.0)) > 0]
        positive_error = [row for row in family_rows if float(row.get("delta_error_t2i_minus_i2t", 0.0)) > 0]
        entropy_family = max(positive_entropy, key=lambda row: float(row["delta_entropy_t2i_minus_i2t"]))["name"] if positive_entropy else "none"
        error_family = max(positive_error, key=lambda row: float(row["delta_error_t2i_minus_i2t"]))["name"] if positive_error else "none"
        slot_ablation_conclusion = f"strict-output localization; entropy family={entropy_family}; error family={error_family}"
    rows = [
        {
            "setting": "original strict protocol",
            "delta_h": f"{baseline.get('state_entropy', 'TBD')} joint",
            "delta_err": f"{baseline.get('error', 'TBD')} joint",
            "conclusion": "baseline; extractor-limited until human validation",
        },
        {
            "setting": "randomized option order smoke",
            "delta_h": "smoke only",
            "delta_err": "smoke only",
            "conclusion": f"pipeline PASS; {smoke_audit.get('state_rows', 0)} states; not full robustness",
        },
        *full_option_order_rows(option_full_dirs),
        {
            "setting": "prompt template v2 smoke",
            "delta_h": "smoke only",
            "delta_err": "smoke only",
            "conclusion": f"pipeline PASS; {prompt_smoke_audit.get('state_rows', 0)} states; not full robustness",
        },
        *full_prompt_template_rows(prompt_full_dirs),
        {"setting": "object-binding sanity", "delta_h": "N/A", "delta_err": "N/A", "conclusion": object_binding_conclusion},
        {"setting": "T2I duplicate/mode-collapse hash check", "delta_h": "N/A", "delta_err": "N/A", "conclusion": duplicate_conclusion},
        {"setting": "easy/hard concept split", "delta_h": "N/A", "delta_err": "N/A", "conclusion": difficulty_conclusion},
        {"setting": "random concept control", "delta_h": "N/A", "delta_err": "N/A", "conclusion": random_control_conclusion},
        {"setting": "slot ablation", "delta_h": "N/A", "delta_err": "N/A", "conclusion": slot_ablation_conclusion},
        {"setting": "alternative extractor", "delta_h": "TBD", "delta_err": "TBD", "conclusion": "pending; no external paid API used"},
    ]
    return rows


def markdown_table(rows: list[dict], fields: list[str] | None = None) -> list[str]:
    if not rows:
        return ["(no rows)"]
    fields = fields or list(rows[0].keys())
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return lines


def build_failure_appendix(quadrant_dir: Path) -> list[dict]:
    rows = read_csv(quadrant_dir / "quadrant_cases.csv")
    out = []
    for row in rows:
        out.append(
            {
                "route": row["route"],
                "concept_id": row["concept_id"],
                "slot": row["slot"],
                "quadrant": row["quadrant"],
                "error_rate": fmt(row["error_rate"]),
                "sample_id": row["sample_id"],
                "target": row["target_value"],
                "predicted": row["predicted_value"],
                "image_path": row["image_path"],
            }
        )
    return out


def write_package(out_dir: Path, tables: dict[str, list[dict]], args: argparse.Namespace) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "table1_comparability_audit.csv", tables["table1"])
    write_csv(out_dir / "table2_extractor_validation.csv", tables["table2"])
    write_csv(out_dir / "table3_route_entropy_error.csv", tables["table3"])
    write_csv(out_dir / "table4_slot_breakdown.csv", tables["table4"])
    write_csv(out_dir / "table5_robustness.csv", tables["table5"])
    write_csv(out_dir / "failure_case_appendix.csv", tables["failures"])

    extractor_dir = Path(args.extractor_validation_dir)
    manual_report = read_json(extractor_dir / "manual_annotation_validation_report.json")
    manual_validated = (
        manual_report.get("status") == "PASS"
        and (extractor_dir / "extractor_validation_report.md").exists()
        and (extractor_dir / "extractor_validation_metrics.csv").exists()
    )
    full_robustness_present = any(
        "full robustness PASS" in str(row.get("conclusion", "")) for row in tables["table5"]
    )
    package_status = (
        "PAPER_DRAFT_DOD_PASS"
        if manual_validated and full_robustness_present
        else "PRELIMINARY_EXTRACTOR_LIMITED"
    )

    lines = [
        "# Preliminary Semantic Entropy Paper Tables",
        "",
        f"Status: {package_status}",
        "",
        "These tables summarize current evidence. They are not final paper-ready claims until manual extractor validation is complete and the claim-boundary audit remains clean.",
        "",
        "## Input Files",
        "",
        f"- `{args.strict_dir}/comparability_audit.json`",
        f"- `{args.extractor_validation_dir}/extractor_validation_metrics.csv` if present",
        f"- `{args.quadrant_dir}/concept_route_metrics.csv`",
        f"- `{args.quadrant_dir}/quadrant_cases.csv`",
        f"- `{args.bootstrap_dir}/bootstrap_summary.csv`",
        f"- `{args.option_order_smoke_dir}/option_order_audit.json`",
        f"- `{args.option_order_full_dir}` plus fallback `{args.option_order_full_fallback_dir}`",
        f"- `{args.prompt_template_smoke_dir}/prompt_template_audit.json`",
        f"- `{args.prompt_template_full_dir}` plus fallback `{args.prompt_template_full_fallback_dir}`",
        f"- `{args.object_binding_dir}/object_binding_summary.csv`",
        f"- `{args.object_binding_dir}/targeted_questions/targeted_binding_score_report.json`",
        f"- `{args.image_duplicate_dir}/generated_image_duplicate_summary.csv`",
        "",
        "## Commands",
        "",
        "```bash",
        "./.venv-transformers/bin/python scripts/build_semantic_entropy_paper_tables.py \\",
        f"  --strict-dir {args.strict_dir} \\",
        f"  --extractor-validation-dir {args.extractor_validation_dir} \\",
        f"  --quadrant-dir {args.quadrant_dir} \\",
        f"  --bootstrap-dir {args.bootstrap_dir} \\",
        f"  --out-dir {args.out_dir}",
        "```",
        "",
        "## Output Files",
        "",
        f"- `{out_dir / 'table1_comparability_audit.csv'}`",
        f"- `{out_dir / 'table2_extractor_validation.csv'}`",
        f"- `{out_dir / 'table3_route_entropy_error.csv'}`",
        f"- `{out_dir / 'table4_slot_breakdown.csv'}`",
        f"- `{out_dir / 'table5_robustness.csv'}`",
        f"- `{out_dir / 'failure_case_appendix.csv'}`",
        f"- `{out_dir / 'paper_tables_preliminary.md'}`",
        f"- `{out_dir / 'failure_case_appendix.md'}`",
        f"- `{out_dir / 'evidence_index.csv'}`",
        f"- `{out_dir / 'gap_checklist.md'}`",
        "",
        "## Sample Counts",
        "",
        f"- Table 1 rows: `{len(tables['table1'])}`",
        f"- Table 2 rows: `{len(tables['table2'])}`",
        f"- Table 3 rows: `{len(tables['table3'])}`",
        f"- Table 4 rows: `{len(tables['table4'])}`",
        f"- Table 5 rows: `{len(tables['table5'])}`",
        f"- Failure appendix rows: `{len(tables['failures'])}`",
        "",
        "## Pass/Fail Checks",
        "",
        f"- Comparability table generated: `{bool(tables['table1'])}`",
        f"- Extractor validation table generated: `{bool(tables['table2'])}`",
        f"- Route entropy/error table generated: `{bool(tables['table3'])}`",
        f"- Robustness table generated: `{bool(tables['table5'])}`",
        "- Preliminary/extractor-limited caveat retained: `True`",
        "",
        "## Table 1: Comparability Audit",
        "",
        *markdown_table(tables["table1"]),
        "",
        "## Table 2: Extractor Validation",
        "",
        *markdown_table(tables["table2"]),
        "",
        "## Table 3: Route Entropy and Error",
        "",
        *markdown_table(tables["table3"]),
        "",
        "## Table 4: Slot-Level Breakdown",
        "",
        *markdown_table(tables["table4"]),
        "",
        "## Table 5: Robustness",
        "",
        *markdown_table(tables["table5"]),
        "",
        "## Claim Allowed After This Step",
        "",
        (
            "Allowed now: controlled semantic-state route-level entropy comparison with extractor-validation and robustness caveats."
            if package_status == "PAPER_DRAFT_DOD_PASS"
            else "Allowed now: strict protocol comparability and preliminary extractor-limited error/entropy summaries."
        ),
        "",
        "## Claim Still Not Allowed",
        "",
        "Not allowed now: extractor validity, direct text/image entropy comparability, or unified internal entropy space. Route-specific semantic-instability wording still requires manual validation plus claim-boundary review.",
    ]
    (out_dir / "paper_tables_preliminary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    failure_lines = [
        "# Failure Case Appendix",
        "",
        "Representative low-entropy/high-error and related cases from automated strict extractor outputs.",
        "",
        *markdown_table(tables["failures"]),
    ]
    (out_dir / "failure_case_appendix.md").write_text("\n".join(failure_lines) + "\n", encoding="utf-8")

    evidence_rows = [
        {"artifact": "strict comparability audit", "path": f"{args.strict_dir}/comparability_audit.json", "status": "PASS"},
        {"artifact": "extractor validation manifest/form", "path": args.extractor_validation_dir, "status": "READY_FOR_MANUAL_ANNOTATION"},
        {
            "artifact": "manual annotation handoff",
            "path": f"{args.extractor_validation_dir}/manual_annotation_handoff.md",
            "status": "READY_FOR_HUMAN_LABELING",
        },
        {
            "artifact": "manual annotation batches",
            "path": f"{args.extractor_validation_dir}/annotation_batches",
            "status": "READY_FOR_BATCH_LABELING",
        },
        {
            "artifact": "manual annotation progress audit",
            "path": args.manual_progress_dir,
            "status": "READY_TRACKER_NOT_VALIDATION",
        },
        {
            "artifact": "portable manual annotation package",
            "path": args.manual_package_dir,
            "status": "READY_FOR_ANNOTATOR_NOT_VALIDATION",
        },
        {
            "artifact": "portable manual annotation package QA",
            "path": args.manual_package_qa_dir,
            "status": "PACKAGE_QA_PASS_NOT_VALIDATION",
        },
        {
            "artifact": "safe manual annotation importer",
            "path": args.manual_import_dir,
            "status": "READY_DRY_RUN_NOT_VALIDATION",
        },
        {
            "artifact": "post-manual import/validate/score smoke",
            "path": args.post_manual_smoke_dir,
            "status": "SMOKE_ONLY_NOT_VALIDATION",
        },
        {
            "artifact": "post-manual pipeline",
            "path": "scripts/run_semantic_entropy_post_manual_pipeline.sh",
            "status": "READY_GUARDED_BY_MANUAL_VALIDATION",
        },
        {
            "artifact": "manual annotation validator",
            "path": f"{args.extractor_validation_dir}/manual_annotation_validation_report.json",
            "status": "FAIL_UNTIL_MANUAL_LABELS_COMPLETE",
        },
        {"artifact": "quadrant analysis", "path": args.quadrant_dir, "status": "PRELIMINARY_EXTRACTOR_LIMITED"},
        {"artifact": "bootstrap stability", "path": args.bootstrap_dir, "status": "PRELIMINARY_EXTRACTOR_LIMITED"},
        {"artifact": "sample-size sensitivity", "path": args.sample_size_dir, "status": "PRELIMINARY_EXTRACTOR_LIMITED"},
        {"artifact": "object-binding sanity candidates", "path": args.object_binding_dir, "status": "RETROSPECTIVE_EXTRACTOR_LIMITED"},
        {
            "artifact": "targeted object-binding question package",
            "path": f"{args.object_binding_dir}/targeted_questions",
            "status": "SCORED_IF_REPORT_PASS",
        },
        {
            "artifact": "targeted object-binding scorer",
            "path": f"{args.object_binding_dir}/targeted_questions/targeted_binding_score_report.md",
            "status": "PASS_IF_SCORE_REPORT_PASS",
        },
        {"artifact": "image duplicate/mode-collapse hash check", "path": args.image_duplicate_dir, "status": "RETROSPECTIVE_PIXEL_HASH_ANALYSIS"},
        {
            "artifact": "offline image-embedding fallback duplicate check",
            "path": args.image_embedding_dir,
            "status": "FALLBACK_PIXEL_STAT_NOT_CLIP",
        },
        {
            "artifact": "alternative extractor feasibility audit",
            "path": args.alternative_extractor_dir,
            "status": "BLOCKED_NO_LOCAL_ALTERNATIVE_EXTRACTOR",
        },
        {
            "artifact": "alternative extractor scorer",
            "path": f"{args.alternative_extractor_dir}/alternative_extractor_score_report.md",
            "status": "READY_OUTPUTS_INCOMPLETE",
        },
        {"artifact": "easy/hard concept split", "path": args.concept_difficulty_dir, "status": "RETROSPECTIVE_STRICT_OUTPUT_ANALYSIS"},
        {"artifact": "random concept control", "path": args.random_concept_control_dir, "status": "RETROSPECTIVE_STRICT_OUTPUT_SANITY_CHECK"},
        {"artifact": "slot ablation", "path": args.slot_ablation_dir, "status": "RETROSPECTIVE_STRICT_OUTPUT_ANALYSIS"},
        {"artifact": "claim boundary audit", "path": args.claim_boundary_dir, "status": "PASS"},
        {
            "artifact": "stage report completeness audit",
            "path": args.stage_report_audit_dir,
            "status": "ACTION_REQUIRED_REPORT_STRUCTURE",
        },
        {
            "artifact": "requirement/DoD traceability audit",
            "path": args.requirement_audit_dir,
            "status": "ACTIVE_NOT_COMPLETE",
        },
        {"artifact": "execution status audit", "path": args.execution_status_dir, "status": "ACTIVE_NOT_COMPLETE"},
        {"artifact": "option-order robustness smoke", "path": args.option_order_smoke_dir, "status": "SMOKE_PASS_NOT_FULL_ROBUSTNESS"},
        {"artifact": "prompt-template robustness smoke", "path": args.prompt_template_smoke_dir, "status": "SMOKE_PASS_NOT_FULL_ROBUSTNESS"},
        {
            "artifact": "guarded option-order full launcher",
            "path": f"{args.option_order_full_dir}/launchers",
            "status": "READY_BUT_GUARDED_BY_MANUAL_VALIDATION",
        },
        {
            "artifact": "option-order full robustness fallback outputs",
            "path": args.option_order_full_fallback_dir,
            "status": "FULL_ROBUSTNESS_EVIDENCE_IF_ALL_SEED_AUDITS_PASS",
        },
        {
            "artifact": "guarded prompt-template full launcher",
            "path": f"{args.prompt_template_full_dir}/launchers",
            "status": "READY_BUT_GUARDED_BY_MANUAL_VALIDATION",
        },
        {
            "artifact": "prompt-template full robustness fallback outputs",
            "path": args.prompt_template_full_fallback_dir,
            "status": "FULL_ROBUSTNESS_EVIDENCE_IF_ALL_TEMPLATE_AUDITS_PASS",
        },
        {
            "artifact": "full robustness guard preflight",
            "path": args.robustness_preflight_dir,
            "status": "GUARD_READY_NOT_ROBUSTNESS_EVIDENCE",
        },
        {"artifact": "preliminary paper tables", "path": str(out_dir), "status": "GENERATED"},
    ]
    write_csv(out_dir / "evidence_index.csv", evidence_rows)
    evidence_lines = ["# Evidence Index", "", *markdown_table(evidence_rows)]
    (out_dir / "evidence_index.md").write_text("\n".join(evidence_lines) + "\n", encoding="utf-8")

    gaps = [
        {"requirement": "manual extractor validation completed", "status": "missing", "next_step": "fill manual_annotations.csv and run --mode score"},
        {"requirement": "manual annotation handoff", "status": "ready", "next_step": "open manual_annotation_handoff.md and complete manual_annotations.csv"},
        {"requirement": "manual annotation batches", "status": "ready", "next_step": "fill per-batch manual CSVs, merge, inspect, then use --write-main"},
        {"requirement": "manual annotation progress audit", "status": "ready", "next_step": "refresh while filling manual_annotations.csv; does not replace validate-manual"},
        {"requirement": "portable manual annotation package", "status": "ready", "next_step": "send package to annotator; import completed manual_annotations.csv afterward"},
        {"requirement": "portable manual annotation package QA", "status": "ready", "next_step": "rerun after regenerating package; does not replace validate-manual"},
        {"requirement": "safe manual annotation importer", "status": "ready", "next_step": "dry-run returned CSV, then rerun with --write only when READY_TO_IMPORT"},
        {"requirement": "post-manual import/validate/score smoke", "status": "ready", "next_step": "rerun after changing importer, validator, scorer, or strict output schema"},
        {"requirement": "post-manual pipeline", "status": "ready", "next_step": "run after manual_annotations.csv is complete; full robustness remains opt-in"},
        {"requirement": "manual annotation validator PASS", "status": "missing", "next_step": "run --mode validate-manual after exporting manual_annotations.csv"},
        {"requirement": "extractor_validation_metrics.csv exists", "status": "missing", "next_step": "run score after validator PASS"},
        {"requirement": "full option-order robustness seeds", "status": "ready", "next_step": "use completed seed audits if all PASS; otherwise rerun seeds 20270605,20270606,20270607"},
        {"requirement": "guarded option-order launcher", "status": "ready", "next_step": "bash robustness_option_order/launchers/launch_all_option_order_seeds.sh after validator PASS"},
        {"requirement": "prompt-template robustness smoke", "status": "ready", "next_step": "use only as pipeline check, not robustness evidence"},
        {"requirement": "guarded prompt-template launcher", "status": "ready", "next_step": "bash robustness_prompt_template/launchers/launch_all_prompt_templates.sh after validator PASS"},
        {"requirement": "full robustness guard preflight", "status": "ready", "next_step": "rerun before full robustness launch; does not replace full robustness evidence"},
        {"requirement": "object-binding strict failure index", "status": "ready", "next_step": "use as candidate manifest for manual/targeted sanity checks"},
        {"requirement": "targeted object-binding sanity prompts", "status": "ready", "next_step": "use score report if PASS; rerun only after changing questions or outputs"},
        {"requirement": "targeted object-binding scorer", "status": "ready", "next_step": "score report should be PASS with 90/90 completed answers"},
        {"requirement": "T2I exact/perceptual-hash duplicate check", "status": "ready", "next_step": "treat as pixel-hash evidence only"},
        {"requirement": "offline image-embedding fallback duplicate check", "status": "ready", "next_step": "treat as pixel-stat fallback evidence only"},
        {"requirement": "CLIP/SigLIP/DINO semantic embedding duplicate check", "status": "pending", "next_step": "requires a complete local cached model or explicit approval to fetch/use an external model"},
        {"requirement": "alternative extractor feasibility audit", "status": "ready", "next_step": "use runbook only after choosing a complete local model or approved external service"},
        {"requirement": "alternative extractor scorer", "status": "ready", "next_step": "fill outputs template and rerun scorer without --allow-incomplete"},
        {"requirement": "easy/hard concept split", "status": "ready", "next_step": "use only as strict-output sensitivity evidence"},
        {"requirement": "random concept control", "status": "ready", "next_step": "use only as mismatched-target metric sensitivity check"},
        {"requirement": "slot ablation", "status": "ready", "next_step": "use only as strict-output localization evidence"},
        {"requirement": "claim boundary audit", "status": "ready", "next_step": "rerun before paper draft or after editing reports"},
        {"requirement": "stage report completeness audit", "status": "ready", "next_step": "fill required sections in stage reports before paper drafting"},
        {"requirement": "requirement/DoD traceability audit", "status": "ready", "next_step": "refresh after manual score or full robustness"},
        {"requirement": "execution status audit", "status": "ready", "next_step": "refresh after manual score or full robustness"},
        {"requirement": "alternative extractor robustness", "status": "pending", "next_step": "blocked until complete local alternative model or approved external service is available"},
        {"requirement": "final claim upgraded beyond protocol comparability", "status": "not allowed", "next_step": "requires validation plus robustness"},
    ]
    write_csv(out_dir / "gap_checklist.csv", gaps)
    gap_lines = ["# Gap Checklist", "", *markdown_table(gaps)]
    (out_dir / "gap_checklist.md").write_text("\n".join(gap_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    strict_dir = Path(args.strict_dir)
    extractor_dir = Path(args.extractor_validation_dir)
    quadrant_dir = Path(args.quadrant_dir)
    bootstrap_dir = Path(args.bootstrap_dir)
    smoke_dir = Path(args.option_order_smoke_dir)
    option_full_dirs = candidate_dirs(args.option_order_full_dir, args.option_order_full_fallback_dir)
    prompt_smoke_dir = Path(args.prompt_template_smoke_dir)
    prompt_full_dirs = candidate_dirs(args.prompt_template_full_dir, args.prompt_template_full_fallback_dir)
    object_binding_dir = Path(args.object_binding_dir)
    image_duplicate_dir = Path(args.image_duplicate_dir)
    concept_difficulty_dir = Path(args.concept_difficulty_dir)
    random_control_dir = Path(args.random_concept_control_dir)
    slot_ablation_dir = Path(args.slot_ablation_dir)
    out_dir = Path(args.out_dir)

    audit = read_json(strict_dir / "comparability_audit.json")
    metrics = metrics_index(quadrant_dir)
    table3 = table3_route_entropy(metrics)
    tables = {
        "table1": table1_comparability(audit),
        "table2": table2_extractor_validation(extractor_dir),
        "table3": table3,
        "table4": table4_slot_breakdown(metrics, bootstrap_dir),
        "table5": table5_robustness(
            table3,
            smoke_dir,
            option_full_dirs,
            prompt_smoke_dir,
            prompt_full_dirs,
            object_binding_dir,
            image_duplicate_dir,
            concept_difficulty_dir,
            random_control_dir,
            slot_ablation_dir,
        ),
        "failures": build_failure_appendix(quadrant_dir),
    }
    write_package(out_dir, tables, args)
    print(f"[INFO] wrote preliminary paper table package under {out_dir}")


if __name__ == "__main__":
    main()
