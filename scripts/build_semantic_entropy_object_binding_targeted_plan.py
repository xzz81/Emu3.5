#!/usr/bin/env python3
"""Prepare a targeted object-binding sanity-check question package.

This does not run inference. It freezes a traceable case set and writes the
questions, expected answers, and blank output template needed for a later
targeted object-binding control.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


QUESTION_FIELDS = [
    "question_id",
    "case_id",
    "question_type",
    "concept_id",
    "route",
    "sample_id",
    "image_path",
    "source_subset",
    "failure_types",
    "target_description",
    "question",
    "allowed_values",
    "expected_answer",
]

OUTPUT_FIELDS = ["question_id", "model_or_annotator", "answer", "raw_output", "notes"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--object-binding-dir", default="outputs/semantic_entropy_umm/object_binding_sanity")
    parser.add_argument("--out-dir", default="outputs/semantic_entropy_umm/object_binding_sanity/targeted_questions")
    parser.add_argument("--failure-cases", type=int, default=20)
    parser.add_argument("--control-cases", type=int, default=10)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def concept_prompt(row: dict) -> str:
    return (
        f"{row['target_color_1']} {row['target_object_1']} "
        f"{relation_to_words(row['target_relation'])} "
        f"{row['target_color_2']} {row['target_object_2']}"
    )


def relation_to_words(value: str) -> str:
    return {
        "object_1_left_of_object_2": "left of",
        "object_1_right_of_object_2": "right of",
        "object_1_above_object_2": "above",
        "object_1_below_object_2": "below",
    }.get(value, value)


def pair_value(color: str, obj: str) -> str:
    return f"{color}_{obj}"


def select_failure_cases(failures: list[dict], limit: int) -> list[dict]:
    selected = []
    seen_concepts = set()
    for row in failures:
        if row["concept_id"] in seen_concepts:
            continue
        selected.append(row)
        seen_concepts.add(row["concept_id"])
        if len(selected) >= limit:
            return selected
    for row in failures:
        key = (row["concept_id"], row["route"], row["sample_id"])
        if key in {(item["concept_id"], item["route"], item["sample_id"]) for item in selected}:
            continue
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def select_control_cases(candidates: list[dict], selected_failures: list[dict], limit: int) -> list[dict]:
    failure_concepts = {row["concept_id"] for row in selected_failures}
    controls = []
    by_route: dict[str, list[dict]] = defaultdict(list)
    for row in candidates:
        if row["any_binding_failure"] != "False":
            continue
        if row["concept_id"] in failure_concepts:
            continue
        by_route[row["route"]].append(row)
    route_order = ["I2T", "T2I"]
    while len(controls) < limit and any(by_route.values()):
        progressed = False
        for route in route_order:
            if by_route[route]:
                controls.append(by_route[route].pop(0))
                progressed = True
                if len(controls) >= limit:
                    break
        if not progressed:
            break
    return controls


def build_questions(case_rows: list[dict]) -> list[dict]:
    questions = []
    for case_idx, row in enumerate(case_rows, start=1):
        case_id = f"binding_case_{case_idx:03d}"
        prompt = concept_prompt(row)
        object1_pair = pair_value(row["target_color_1"], row["target_object_1"])
        object2_pair = pair_value(row["target_color_2"], row["target_object_2"])
        pair_choices = [object1_pair, object2_pair, "unknown"]
        relation_choices = [
            "object_1_left_of_object_2",
            "object_1_right_of_object_2",
            "object_1_above_object_2",
            "object_1_below_object_2",
            "unknown",
        ]
        specs = [
            (
                "object_1_color_shape",
                f"Target description: {prompt} Which colored object is object_1, the first object named in the target description? Choose exactly one label from: {', '.join(pair_choices)}. Answer with the label only.",
                pair_choices,
                object1_pair,
            ),
            (
                "object_2_color_shape",
                f"Target description: {prompt} Which colored object is object_2, the second object named in the target description? Choose exactly one label from: {', '.join(pair_choices)}. Answer with the label only.",
                pair_choices,
                object2_pair,
            ),
            (
                "relation_object1_to_object2",
                f"Target description: {prompt} What is the spatial relation from object_1 to object_2? Choose exactly one label from: {', '.join(relation_choices)}. Answer with the label only.",
                relation_choices,
                row["target_relation"],
            ),
        ]
        for q_idx, (question_type, question, choices, expected) in enumerate(specs, start=1):
            questions.append(
                {
                    "question_id": f"{case_id}_q{q_idx:02d}",
                    "case_id": case_id,
                    "question_type": question_type,
                    "concept_id": row["concept_id"],
                    "route": row["route"],
                    "sample_id": row["sample_id"],
                    "image_path": row["image_path"],
                    "source_subset": row["source_subset"],
                    "failure_types": row.get("failure_types", ""),
                    "target_description": prompt,
                    "question": question,
                    "allowed_values": ";".join(choices),
                    "expected_answer": expected,
                }
            )
    return questions


def md_table(rows: list[dict], fields: list[str]) -> list[str]:
    if not rows:
        return ["(no rows)"]
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return lines


def write_report(out_dir: Path, cases: list[dict], questions: list[dict], args: argparse.Namespace) -> None:
    source_counts = defaultdict(int)
    route_counts = defaultdict(int)
    for row in cases:
        source_counts[row["source_subset"]] += 1
        route_counts[row["route"]] += 1
    preview_fields = ["case_id", "source_subset", "route", "concept_id", "sample_id", "failure_types", "image_path"]
    case_preview = [
        {
            "case_id": f"binding_case_{idx:03d}",
            "source_subset": row["source_subset"],
            "route": row["route"],
            "concept_id": row["concept_id"],
            "sample_id": row["sample_id"],
            "failure_types": row.get("failure_types", ""),
            "image_path": row["image_path"],
        }
        for idx, row in enumerate(cases[:12], start=1)
    ]
    lines = [
        "# Targeted Object-Binding Sanity Question Plan",
        "",
        "Status: READY_NOT_RUN",
        "",
        "This package freezes targeted object-binding questions from existing strict outputs. It does not run inference and does not establish object-binding robustness.",
        "",
        "## Input Files",
        "",
        f"- `{Path(args.object_binding_dir) / 'strict_binding_failure_cases.csv'}`",
        f"- `{Path(args.object_binding_dir) / 'object_binding_candidate_manifest.csv'}`",
        "",
        "## Output Files",
        "",
        f"- `{out_dir / 'targeted_binding_cases.csv'}`",
        f"- `{out_dir / 'targeted_binding_questions.csv'}`",
        f"- `{out_dir / 'targeted_binding_outputs_template.csv'}`",
        f"- `{out_dir / 'targeted_binding_score_report.md'}` after scoring",
        f"- `{out_dir / 'targeted_binding_runbook.md'}`",
        "",
        "## Summary",
        "",
        f"- Cases: `{len(cases)}`",
        f"- Questions: `{len(questions)}`",
        f"- Source counts: `{dict(source_counts)}`",
        f"- Route counts: `{dict(route_counts)}`",
        f"- Questions per case: `3`",
        "",
        "## Case Preview",
        "",
        *md_table(case_preview, preview_fields),
        "",
        "## Run And Score Contract",
        "",
        "Run a future worker over `targeted_binding_questions.csv`, writing one row per `question_id` into `targeted_binding_outputs_template.csv` fields. Normalize answers against `allowed_values`, then compare with `expected_answer`.",
        "",
        "Validate the empty or filled output file with:",
        "",
        "```bash",
        "PY=./.venv-transformers/bin/python",
        "$PY scripts/score_semantic_entropy_object_binding_targeted.py \\",
        f"  --questions-csv {out_dir / 'targeted_binding_questions.csv'} \\",
        f"  --outputs-csv {out_dir / 'targeted_binding_outputs_template.csv'} \\",
        f"  --out-dir {out_dir} \\",
        "  --allow-incomplete",
        "```",
        "",
        "The scorer only writes accuracy metrics when all 90 answers are present and legal; empty templates remain `FAIL_INCOMPLETE`.",
        "",
        "Required metrics after running:",
        "",
        "- object_1 color-shape binding accuracy",
        "- object_2 color-shape binding accuracy",
        "- relation accuracy",
        "- route-level accuracy by I2T/T2I",
        "- failure-source vs control-source accuracy",
        "",
        "## Claim Boundary",
        "",
        "Allowed now: the targeted object-binding control has a fixed input package.",
        "",
        "Still not allowed: object-binding robustness or relation/color-object binding accuracy under targeted questions.",
    ]
    (out_dir / "targeted_binding_runbook.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    object_binding_dir = Path(args.object_binding_dir)
    out_dir = Path(args.out_dir)
    failures = read_csv(object_binding_dir / "strict_binding_failure_cases.csv")
    candidates = read_csv(object_binding_dir / "object_binding_candidate_manifest.csv")
    failure_cases = select_failure_cases(failures, args.failure_cases)
    control_cases = select_control_cases(candidates, failure_cases, args.control_cases)
    for row in failure_cases:
        row["source_subset"] = "strict_binding_failure"
    for row in control_cases:
        row["source_subset"] = "matched_binding_control"
    cases = failure_cases + control_cases
    questions = build_questions(cases)
    output_rows = [{"question_id": row["question_id"], "model_or_annotator": "", "answer": "", "raw_output": "", "notes": ""} for row in questions]
    out_dir.mkdir(parents=True, exist_ok=True)
    case_fields = [
        "source_subset",
        "concept_id",
        "route",
        "sample_id",
        "image_path",
        "failure_types",
        "target_object_1",
        "target_color_1",
        "target_object_2",
        "target_color_2",
        "target_relation",
        "pred_object_1",
        "pred_color_1",
        "pred_object_2",
        "pred_color_2",
        "pred_relation",
    ]
    write_csv(out_dir / "targeted_binding_cases.csv", cases, case_fields)
    write_csv(out_dir / "targeted_binding_questions.csv", questions, QUESTION_FIELDS)
    write_csv(out_dir / "targeted_binding_outputs_template.csv", output_rows, OUTPUT_FIELDS)
    write_report(out_dir, cases, questions, args)
    print(f"[INFO] wrote {len(cases)} cases and {len(questions)} questions under {out_dir}")


if __name__ == "__main__":
    main()
