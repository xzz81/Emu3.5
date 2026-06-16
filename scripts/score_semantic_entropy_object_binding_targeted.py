#!/usr/bin/env python3
"""Validate and score targeted object-binding sanity outputs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


QUESTION_TYPES = {
    "object_1_color_shape",
    "object_2_color_shape",
    "relation_object1_to_object2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions-csv", default="outputs/semantic_entropy_umm/object_binding_sanity/targeted_questions/targeted_binding_questions.csv")
    parser.add_argument("--outputs-csv", default="outputs/semantic_entropy_umm/object_binding_sanity/targeted_questions/targeted_binding_outputs_template.csv")
    parser.add_argument("--out-dir", default="outputs/semantic_entropy_umm/object_binding_sanity/targeted_questions")
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0].keys()) if rows else [])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalize(text: str) -> str:
    value = text.lower().strip()
    value = re.sub(r"[^a-z0-9_ ]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value.replace(" ", "_")


def normalize_answer(raw_answer: str, allowed_values: list[str]) -> tuple[str, str]:
    normalized = normalize(raw_answer)
    allowed = {value: value for value in allowed_values}
    allowed.update({normalize(value): value for value in allowed_values})
    if normalized in allowed:
        return allowed[normalized], ""
    text = f" {normalized} "
    hits = [value for value in allowed_values if f" {normalize(value)} " in text]
    if len(hits) == 1:
        return hits[0], ""
    relation_phrases = {
        "left": "object_1_left_of_object_2",
        "right": "object_1_right_of_object_2",
        "above": "object_1_above_object_2",
        "below": "object_1_below_object_2",
    }
    relation_hits = [value for key, value in relation_phrases.items() if key in normalized and value in allowed_values]
    if len(relation_hits) == 1:
        return relation_hits[0], ""
    if "unknown" in normalized and "unknown" in allowed_values:
        return "unknown", ""
    return "", "illegal_value"


def index_rows(rows: list[dict]) -> tuple[dict[str, dict], list[str]]:
    out = {}
    duplicates = []
    for row in rows:
        question_id = row.get("question_id", "")
        if question_id in out:
            duplicates.append(question_id)
        out[question_id] = row
    return out, duplicates


def metric_row(group: str, value: str, rows: list[dict]) -> dict:
    total = len(rows)
    correct = sum(row["is_correct"] == "True" for row in rows)
    unknown = sum(row["normalized_answer"] == "unknown" for row in rows)
    invalid = sum(bool(row["issue"]) for row in rows)
    return {
        "group": group,
        "value": value,
        "n": total,
        "correct": correct,
        "accuracy": correct / total if total else "",
        "unknown_rate": unknown / total if total else "",
        "invalid_rate": invalid / total if total else "",
    }


def build_metrics(scored: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in scored:
        groups[("all", "ALL")].append(row)
        groups[("question_type", row["question_type"])].append(row)
        groups[("route", row["route"])].append(row)
        groups[("source_subset", row["source_subset"])].append(row)
        groups[("route_question_type", f"{row['route']}::{row['question_type']}")].append(row)
        groups[("source_question_type", f"{row['source_subset']}::{row['question_type']}")].append(row)
        if row["question_type"] in {"object_1_color_shape", "object_2_color_shape"}:
            groups[("binding_family", "color_object_binding")].append(row)
        if row["question_type"] == "relation_object1_to_object2":
            groups[("binding_family", "relation")].append(row)
    return [metric_row(group, value, rows) for (group, value), rows in sorted(groups.items())]


def md_table(rows: list[dict]) -> list[str]:
    if not rows:
        return ["(no rows)"]
    fields = list(rows[0].keys())
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return lines


def write_report(out_dir: Path, report: dict, metrics: list[dict], issues: list[dict], args: argparse.Namespace) -> None:
    highlights = [
        row
        for row in metrics
        if row["group"] in {"all", "binding_family", "route", "source_subset", "question_type"}
    ]
    lines = [
        "# Targeted Object-Binding Score Report",
        "",
        f"Status: {report['status']}",
        "",
        "This report validates and scores targeted object-binding outputs against the fixed question package. It is not part of the full route-level robustness gate unless the output file is complete and the protocol is explicitly run.",
        "",
        "## Input Files",
        "",
        f"- Questions: `{args.questions_csv}`",
        f"- Outputs: `{args.outputs_csv}`",
        "",
        "## Summary",
        "",
        f"- Questions: `{report['question_rows']}`",
        f"- Output rows: `{report['output_rows']}`",
        f"- Completed answers: `{report['completed_answers']}` / `{report['question_rows']}`",
        f"- Missing output IDs: `{report['missing_output_ids']}`",
        f"- Unknown output IDs: `{report['unknown_output_ids']}`",
        f"- Duplicate output IDs: `{report['duplicate_output_ids']}`",
        f"- Illegal values: `{report['illegal_values']}`",
        "",
        "## Metrics",
        "",
        *md_table(highlights),
        "",
        "## Issues Preview",
        "",
        *md_table(issues[:20]),
        "",
        "## Claim Boundary",
        "",
        "Allowed now: scoring readiness and validation status for the fixed targeted object-binding package.",
        "",
        "Still not allowed: object-binding robustness unless this report is complete and PASS under the planned runtime protocol.",
    ]
    (out_dir / "targeted_binding_score_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    question_rows = read_csv(Path(args.questions_csv))
    output_rows = read_csv(Path(args.outputs_csv))
    question_index, duplicate_questions = index_rows(question_rows)
    output_index, duplicate_outputs = index_rows(output_rows)
    expected_ids = [row["question_id"] for row in question_rows]
    missing_output_ids = [question_id for question_id in expected_ids if question_id not in output_index]
    unknown_output_ids = sorted(set(output_index) - set(expected_ids))
    scored = []
    issues = []
    for question in question_rows:
        question_id = question["question_id"]
        output = output_index.get(question_id, {})
        answer = output.get("answer", "").strip()
        raw_output = output.get("raw_output", "").strip()
        response = answer or raw_output
        allowed_values = question["allowed_values"].split(";")
        normalized_answer = ""
        issue = ""
        if not response:
            issue = "empty_answer"
        else:
            normalized_answer, issue = normalize_answer(response, allowed_values)
        joined = {
            **question,
            "model_or_annotator": output.get("model_or_annotator", ""),
            "answer": answer,
            "raw_output": raw_output,
            "normalized_answer": normalized_answer,
            "issue": issue,
            "is_correct": str(normalized_answer == question["expected_answer"] and not issue),
        }
        scored.append(joined)
        if issue:
            issues.append({"question_id": question_id, "issue": issue, "answer": answer, "raw_output": raw_output})
    for question_id in missing_output_ids:
        issues.append({"question_id": question_id, "issue": "missing_output_row", "answer": "", "raw_output": ""})
    for question_id in unknown_output_ids:
        issues.append({"question_id": question_id, "issue": "unknown_output_id", "answer": output_index[question_id].get("answer", ""), "raw_output": output_index[question_id].get("raw_output", "")})
    for question_id in duplicate_outputs:
        issues.append({"question_id": question_id, "issue": "duplicate_output_id", "answer": "", "raw_output": ""})
    for question_id in duplicate_questions:
        issues.append({"question_id": question_id, "issue": "duplicate_question_id", "answer": "", "raw_output": ""})

    completed = sum(not row["issue"] for row in scored)
    illegal_values = sum(row["issue"] == "illegal_value" for row in scored)
    complete = (
        completed == len(question_rows)
        and not missing_output_ids
        and not unknown_output_ids
        and not duplicate_outputs
        and not duplicate_questions
        and illegal_values == 0
        and all(row["question_type"] in QUESTION_TYPES for row in question_rows)
    )
    metrics = build_metrics(scored) if complete else []
    status = "PASS" if complete else "FAIL_INCOMPLETE"
    report = {
        "status": status,
        "question_rows": len(question_rows),
        "output_rows": len(output_rows),
        "completed_answers": completed,
        "missing_output_ids": len(missing_output_ids),
        "unknown_output_ids": len(unknown_output_ids),
        "duplicate_output_ids": len(duplicate_outputs),
        "duplicate_question_ids": len(duplicate_questions),
        "illegal_values": illegal_values,
        "question_types": sorted({row["question_type"] for row in question_rows}),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "targeted_binding_joined_outputs.csv", scored)
    write_csv(out_dir / "targeted_binding_metrics.csv", metrics, ["group", "value", "n", "correct", "accuracy", "unknown_rate", "invalid_rate"])
    (out_dir / "targeted_binding_score_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_report(out_dir, report, metrics, issues, args)
    print(json.dumps(report, indent=2))
    if status != "PASS" and not args.allow_incomplete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
