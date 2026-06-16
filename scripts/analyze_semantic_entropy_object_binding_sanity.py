#!/usr/bin/env python3
"""Retrospective object-binding sanity index from strict semantic states."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


SLOTS = ["object_1", "color_1", "object_2", "color_2", "relation", "background"]
BINDING_SLOTS = ["object_1", "color_1", "object_2", "color_2", "relation"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-dir", default="outputs/semantic_entropy_umm/strict_compare")
    parser.add_argument("--out-dir", default="outputs/semantic_entropy_umm/object_binding_sanity")
    parser.add_argument("--case-limit", type=int, default=50)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None and rows:
        fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or [])
        writer.writeheader()
        writer.writerows(rows)


def answer_index(answers: list[dict]) -> dict[tuple[str, str, int, str], dict]:
    out = {}
    for row in answers:
        out[(row["concept_id"], row["route"], int(row["sample_id"]), row["slot"])] = row
    return out


def is_binding_eligible(target: dict) -> bool:
    return target["object_1"] != target["object_2"] and target["color_1"] != target["color_2"]


def failure_types(state: dict) -> list[str]:
    target = state["target_semantics"]
    pred = state["slots"]
    failures = []
    if pred.get("object_1") != target.get("object_1") or pred.get("object_2") != target.get("object_2"):
        failures.append("object_binding")
    if pred.get("color_1") != target.get("color_1") or pred.get("color_2") != target.get("color_2"):
        failures.append("color_object_binding")
    if pred.get("relation") != target.get("relation"):
        failures.append("relation")
    return failures


def build_rows(strict_dir: Path, case_limit: int):
    states = read_jsonl(strict_dir / "strict_semantic_states.jsonl")
    answers = read_jsonl(strict_dir / "strict_slot_answers.jsonl")
    idx = answer_index(answers)
    candidates = []
    failures = []
    route_totals = Counter()
    route_eligible = Counter()
    for row in states:
        concept_id = row["concept_id"]
        route = row["route"]
        sample_id = int(row["sample_id"])
        target = row["target_semantics"]
        pred = row["slots"]
        route_totals[route] += 1
        eligible = is_binding_eligible(target)
        if not eligible:
            continue
        route_eligible[route] += 1
        fails = failure_types(row)
        object_binding_ok = "object_binding" not in fails
        color_binding_ok = "color_object_binding" not in fails
        relation_ok = "relation" not in fails
        candidate = {
            "concept_id": concept_id,
            "route": route,
            "sample_id": sample_id,
            "image_path": row["image_path"],
            "binding_eligible": True,
            "target_object_1": target["object_1"],
            "target_color_1": target["color_1"],
            "target_object_2": target["object_2"],
            "target_color_2": target["color_2"],
            "target_relation": target["relation"],
            "pred_object_1": pred.get("object_1", "unknown"),
            "pred_color_1": pred.get("color_1", "unknown"),
            "pred_object_2": pred.get("object_2", "unknown"),
            "pred_color_2": pred.get("color_2", "unknown"),
            "pred_relation": pred.get("relation", "unknown"),
            "object_binding_ok": object_binding_ok,
            "color_binding_ok": color_binding_ok,
            "relation_ok": relation_ok,
            "any_binding_failure": bool(fails),
            "failure_types": ";".join(fails),
        }
        candidates.append(candidate)
        if fails:
            raw_outputs = {
                slot: idx.get((concept_id, route, sample_id, slot), {}).get("raw_output", "")
                for slot in BINDING_SLOTS
            }
            failures.append(
                {
                    **candidate,
                    "raw_object_1": raw_outputs["object_1"],
                    "raw_color_1": raw_outputs["color_1"],
                    "raw_object_2": raw_outputs["object_2"],
                    "raw_color_2": raw_outputs["color_2"],
                    "raw_relation": raw_outputs["relation"],
                }
            )
    failures.sort(
        key=lambda row: (
            row["route"] != "T2I",
            -len(row["failure_types"].split(";")),
            row["concept_id"],
            int(row["sample_id"]),
        )
    )
    stats = {
        "state_rows": len(states),
        "slot_answer_rows": len(answers),
        "binding_eligible_rows": len(candidates),
        "route_totals": dict(route_totals),
        "route_eligible": dict(route_eligible),
        "report_case_limit": case_limit,
    }
    return candidates, failures, stats


def summarize(candidates: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in candidates:
        groups[row["route"]].append(row)
        groups["ALL"].append(row)
    rows = []
    for route in [*sorted(route for route in groups if route != "ALL"), "ALL"]:
        items = groups[route]
        total = len(items)
        object_ok = sum(row["object_binding_ok"] for row in items)
        color_ok = sum(row["color_binding_ok"] for row in items)
        relation_ok = sum(row["relation_ok"] for row in items)
        any_fail = sum(row["any_binding_failure"] for row in items)
        rows.append(
            {
                "route": route,
                "subset": "binding_eligible",
                "states": total,
                "any_binding_related_failure": any_fail,
                "object_binding_accuracy": object_ok / total if total else 0.0,
                "color_object_binding_accuracy": color_ok / total if total else 0.0,
                "relation_accuracy": relation_ok / total if total else 0.0,
                "any_failure_rate": any_fail / total if total else 0.0,
            }
        )
    return rows


def md_table(rows: list[dict]) -> list[str]:
    if not rows:
        return ["(no rows)"]
    fields = list(rows[0].keys())
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return lines


def write_report(out_dir: Path, summary: list[dict], failures: list[dict], stats: dict, args: argparse.Namespace) -> None:
    preview_fields = [
        "route",
        "concept_id",
        "sample_id",
        "failure_types",
        "target_object_1",
        "pred_object_1",
        "target_object_2",
        "pred_object_2",
        "target_relation",
        "pred_relation",
        "image_path",
    ]
    failure_preview = [{field: row.get(field, "") for field in preview_fields} for row in failures[: args.case_limit]]
    lines = [
        "# Object-Binding Sanity Candidate Index",
        "",
        "Status: RETROSPECTIVE_EXTRACTOR_LIMITED",
        "",
        "This report only indexes existing strict extractor outputs. It does not run separate object-binding questions and does not complete the planned object-binding sanity control.",
        "",
        "## Input Files",
        "",
        f"- `{args.strict_dir}/strict_semantic_states.jsonl`",
        f"- `{args.strict_dir}/strict_slot_answers.jsonl`",
        "",
        "## Commands",
        "",
        "```bash",
        "./.venv-transformers/bin/python scripts/analyze_semantic_entropy_object_binding_sanity.py \\",
        f"  --strict-dir {args.strict_dir} \\",
        f"  --out-dir {args.out_dir} \\",
        f"  --case-limit {args.case_limit}",
        "```",
        "",
        "## Output Files",
        "",
        f"- `{out_dir / 'object_binding_candidate_manifest.csv'}`",
        f"- `{out_dir / 'strict_binding_failure_cases.csv'}`",
        f"- `{out_dir / 'object_binding_summary.csv'}`",
        f"- `{out_dir / 'object_binding_sanity_report.md'}`",
        "",
        "## Sample Counts",
        "",
        f"- Strict states: `{stats['state_rows']}`",
        f"- Strict slot answers: `{stats['slot_answer_rows']}`",
        f"- Binding-eligible candidate states: `{stats['binding_eligible_rows']}`",
        f"- Binding failure cases: `{len(failures)}`",
        f"- Summary rows: `{len(summary)}`",
        "",
        "## Pass/Fail Checks",
        "",
        f"- Input states present: `{stats['state_rows'] > 0}`",
        f"- Input slot answers present: `{stats['slot_answer_rows'] > 0}`",
        f"- Candidate manifest generated: `{bool(summary)}`",
        "- Retrospective-only caveat retained: `True`",
        "",
        "## Summary",
        "",
        f"- Strict states: {stats['state_rows']}",
        f"- Strict slot answers: {stats['slot_answer_rows']}",
        f"- Binding-eligible candidate states: {stats['binding_eligible_rows']}",
        f"- Binding failure cases: {len(failures)}",
        f"- Route totals: {stats['route_totals']}",
        f"- Route binding-eligible totals: {stats['route_eligible']}",
        "",
        *md_table(summary),
        "",
        "## Representative Failure Cases",
        "",
        *md_table(failure_preview),
        "",
        "## Claim Allowed After This Step",
        "",
        "Use this as a traceable candidate and failure-case index for later object-binding sanity checks.",
        "",
        "## Claim Still Not Allowed",
        "",
        "Do not claim object-binding robustness until separate relation and color-object binding questions are run and scored.",
    ]
    (out_dir / "object_binding_sanity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    strict_dir = Path(args.strict_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates, failures, stats = build_rows(strict_dir, args.case_limit)
    summary = summarize(candidates)
    write_csv(out_dir / "object_binding_candidate_manifest.csv", candidates)
    write_csv(out_dir / "strict_binding_failure_cases.csv", failures)
    write_csv(out_dir / "object_binding_summary.csv", summary)
    write_report(out_dir, summary, failures, stats, args)
    print(f"[INFO] wrote {len(candidates)} candidates and {len(failures)} failure cases under {out_dir}")


if __name__ == "__main__":
    main()
