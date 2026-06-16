#!/usr/bin/env python3
"""Random/mismatched concept control for strict semantic entropy outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


SLOTS = ["object_1", "color_1", "object_2", "color_2", "relation", "background"]
ROUTES = ["I2T", "T2I"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-dir", default="outputs/semantic_entropy_umm/pilot")
    parser.add_argument("--strict-dir", default="outputs/semantic_entropy_umm/strict_compare")
    parser.add_argument("--out-dir", default="outputs/semantic_entropy_umm/random_concept_control")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None and rows:
        fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or [])
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def concept_targets(pilot_dir: Path) -> tuple[list[str], dict[str, dict]]:
    rows = read_jsonl(pilot_dir / "concepts.jsonl")
    concept_ids = sorted(row["concept_id"] for row in rows)
    target_by_id = {row["concept_id"]: row["target_semantics"] for row in rows}
    return concept_ids, target_by_id


def build_control_map(concept_ids: list[str], target_by_id: dict[str, dict]) -> list[dict]:
    mapping = []
    n = len(concept_ids)
    for idx, concept_id in enumerate(concept_ids):
        source_target = target_by_id[concept_id]
        best = None
        best_distance = -1
        for offset in range(1, n):
            candidate_id = concept_ids[(idx + offset) % n]
            candidate_target = target_by_id[candidate_id]
            distance = sum(source_target[slot] != candidate_target[slot] for slot in SLOTS)
            if distance > best_distance:
                best = candidate_id
                best_distance = distance
            if distance == len(SLOTS):
                break
        mapping.append(
            {
                "concept_id": concept_id,
                "control_concept_id": best,
                "target_slot_distance": best_distance,
            }
        )
    return mapping


def build_state_rows(strict_dir: Path, target_by_id: dict[str, dict], control_by_id: dict[str, str]) -> list[dict]:
    states = read_jsonl(strict_dir / "strict_semantic_states.jsonl")
    rows = []
    for state in states:
        concept_id = state["concept_id"]
        control_id = control_by_id[concept_id]
        true_target = target_by_id[concept_id]
        control_target = target_by_id[control_id]
        slots = state["slots"]
        true_misses = [slots.get(slot, "unknown") != true_target[slot] for slot in SLOTS]
        control_misses = [slots.get(slot, "unknown") != control_target[slot] for slot in SLOTS]
        row = {
            "concept_id": concept_id,
            "control_concept_id": control_id,
            "route": state["route"],
            "sample_id": int(state["sample_id"]),
            "image_path": state["image_path"],
            "true_joint_match": not any(true_misses),
            "control_joint_match": not any(control_misses),
            "true_mean_slot_error": sum(true_misses) / len(SLOTS),
            "control_mean_slot_error": sum(control_misses) / len(SLOTS),
            "delta_control_minus_true_mean_slot_error": (sum(control_misses) - sum(true_misses)) / len(SLOTS),
        }
        for slot in SLOTS:
            row[f"pred_{slot}"] = slots.get(slot, "unknown")
            row[f"true_{slot}"] = true_target[slot]
            row[f"control_{slot}"] = control_target[slot]
            row[f"true_{slot}_error"] = slots.get(slot, "unknown") != true_target[slot]
            row[f"control_{slot}_error"] = slots.get(slot, "unknown") != control_target[slot]
        rows.append(row)
    return rows


def route_slot_summary(rows: list[dict]) -> list[dict]:
    out = []
    for route in ROUTES:
        route_rows = [row for row in rows if row["route"] == route]
        for slot in SLOTS:
            true_values = [bool(row[f"true_{slot}_error"]) for row in route_rows]
            control_values = [bool(row[f"control_{slot}_error"]) for row in route_rows]
            out.append(
                {
                    "route": route,
                    "slot": slot,
                    "states": len(route_rows),
                    "true_error_rate": mean([float(value) for value in true_values]),
                    "control_error_rate": mean([float(value) for value in control_values]),
                    "delta_control_minus_true_error": mean([float(value) for value in control_values])
                    - mean([float(value) for value in true_values]),
                }
            )
    for route in ROUTES:
        route_rows = [row for row in rows if row["route"] == route]
        out.append(
            {
                "route": route,
                "slot": "mean_over_slots",
                "states": len(route_rows),
                "true_error_rate": mean([float(row["true_mean_slot_error"]) for row in route_rows]),
                "control_error_rate": mean([float(row["control_mean_slot_error"]) for row in route_rows]),
                "delta_control_minus_true_error": mean([float(row["delta_control_minus_true_mean_slot_error"]) for row in route_rows]),
            }
        )
    return out


def concept_summary(rows: list[dict], control_map: list[dict]) -> list[dict]:
    distance_by_id = {row["concept_id"]: row["target_slot_distance"] for row in control_map}
    out = []
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["concept_id"], row["route"])].append(row)
    for (concept_id, route), group in sorted(grouped.items()):
        out.append(
            {
                "concept_id": concept_id,
                "control_concept_id": group[0]["control_concept_id"],
                "route": route,
                "states": len(group),
                "target_slot_distance": distance_by_id[concept_id],
                "true_mean_slot_error": mean([float(row["true_mean_slot_error"]) for row in group]),
                "control_mean_slot_error": mean([float(row["control_mean_slot_error"]) for row in group]),
                "delta_control_minus_true_mean_slot_error": mean(
                    [float(row["delta_control_minus_true_mean_slot_error"]) for row in group]
                ),
                "true_joint_error_rate": mean([float(not row["true_joint_match"]) for row in group]),
                "control_joint_error_rate": mean([float(not row["control_joint_match"]) for row in group]),
            }
        )
    return out


def high_risk_cases(rows: list[dict]) -> list[dict]:
    cases = []
    for row in rows:
        if row["control_mean_slot_error"] <= row["true_mean_slot_error"]:
            cases.append(
                {
                    "concept_id": row["concept_id"],
                    "control_concept_id": row["control_concept_id"],
                    "route": row["route"],
                    "sample_id": row["sample_id"],
                    "true_mean_slot_error": row["true_mean_slot_error"],
                    "control_mean_slot_error": row["control_mean_slot_error"],
                    "image_path": row["image_path"],
                }
            )
    cases.sort(key=lambda row: (row["delta_control_minus_true_mean_slot_error"] if "delta_control_minus_true_mean_slot_error" in row else 0, row["concept_id"]))
    return cases


def md_table(rows: list[dict], fields: list[str] | None = None) -> list[str]:
    if not rows:
        return ["(no rows)"]
    fields = fields or list(rows[0].keys())
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return lines


def write_report(out_dir: Path, summary: list[dict], concept_rows: list[dict], control_map: list[dict], cases: list[dict], args: argparse.Namespace) -> None:
    mean_summary = [row for row in summary if row["slot"] == "mean_over_slots"]
    fields = ["route", "slot", "states", "true_error_rate", "control_error_rate", "delta_control_minus_true_error"]
    control_fields = ["concept_id", "control_concept_id", "target_slot_distance"]
    concept_fields = [
        "concept_id",
        "control_concept_id",
        "route",
        "target_slot_distance",
        "true_mean_slot_error",
        "control_mean_slot_error",
        "delta_control_minus_true_mean_slot_error",
    ]
    lines = [
        "# Random Concept Control",
        "",
        "Status: RETROSPECTIVE_STRICT_OUTPUT_SANITY_CHECK",
        "",
        "This report re-scores existing strict semantic states against intentionally mismatched concept targets. It does not rerun the extractor or generate new images.",
        "",
        "## Input Files",
        "",
        f"- `{args.pilot_dir}/concepts.jsonl`",
        f"- `{args.strict_dir}/strict_semantic_states.jsonl`",
        "",
        "## Output Files",
        "",
        f"- `{out_dir / 'random_concept_control_map.csv'}`",
        f"- `{out_dir / 'random_concept_state_errors.csv'}`",
        f"- `{out_dir / 'random_concept_route_slot_summary.csv'}`",
        f"- `{out_dir / 'random_concept_concept_summary.csv'}`",
        f"- `{out_dir / 'random_concept_control_report.md'}`",
        "",
        "## Control Rule",
        "",
        "For each concept, choose a deterministic mismatched control concept with the largest slot distance in the fixed semantic schema. Ties follow sorted concept order.",
        "",
        "## Mean-Over-Slots Check",
        "",
        *md_table(mean_summary, fields),
        "",
        "## Route-Slot Summary",
        "",
        *md_table(summary, fields),
        "",
        "## Control Map",
        "",
        *md_table(control_map, control_fields),
        "",
        "## Concept Summary",
        "",
        *md_table(concept_rows, concept_fields),
        "",
        "## Cases Where Control Error Did Not Increase",
        "",
        *md_table(cases[:40]),
        "",
        "## Claim Allowed After This Step",
        "",
        "Use this as a metric-sensitivity check: when concept/image semantics are deliberately mismatched, the strict-output error should rise.",
        "",
        "## Claim Still Not Allowed",
        "",
        "Do not claim extractor validity or route-level semantic instability from this control alone; it only tests a mismatched-target sanity condition.",
    ]
    (out_dir / "random_concept_control_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    concept_ids, target_by_id = concept_targets(Path(args.pilot_dir))
    control_map = build_control_map(concept_ids, target_by_id)
    control_by_id = {row["concept_id"]: row["control_concept_id"] for row in control_map}
    state_rows = build_state_rows(Path(args.strict_dir), target_by_id, control_by_id)
    summary = route_slot_summary(state_rows)
    concept_rows = concept_summary(state_rows, control_map)
    cases = high_risk_cases(state_rows)
    write_csv(out_dir / "random_concept_control_map.csv", control_map)
    write_csv(out_dir / "random_concept_state_errors.csv", state_rows)
    write_csv(out_dir / "random_concept_route_slot_summary.csv", summary)
    write_csv(out_dir / "random_concept_concept_summary.csv", concept_rows)
    write_csv(out_dir / "random_concept_non_increase_cases.csv", cases)
    write_report(out_dir, summary, concept_rows, control_map, cases, args)
    print(f"[INFO] wrote random concept control for {len(concept_ids)} concepts and {len(state_rows)} states under {out_dir}")


if __name__ == "__main__":
    main()
