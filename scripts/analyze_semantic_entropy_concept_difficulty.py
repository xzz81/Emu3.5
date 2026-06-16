#!/usr/bin/env python3
"""Easy/hard concept split for strict semantic entropy outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


SLOTS = ["object_1", "color_1", "object_2", "color_2", "relation", "background"]
ROUTES = ["I2T", "T2I"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-dir", default="outputs/semantic_entropy_umm/strict_compare")
    parser.add_argument("--out-dir", default="outputs/semantic_entropy_umm/concept_difficulty_split")
    parser.add_argument("--samples-per-concept-route", type=int, default=10)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def metric_maps(strict_dir: Path) -> tuple[dict[tuple[str, str, str], float], dict[tuple[str, str, str], float]]:
    entropy = {}
    errors = {}
    for row in read_csv(strict_dir / "strict_route_entropy.csv"):
        entropy[(row["concept_id"], row["route"], row["slot"])] = float(row["entropy"])
    for row in read_csv(strict_dir / "strict_route_error.csv"):
        errors[(row["concept_id"], row["route"], row["slot"])] = float(row["error_rate"])
    return entropy, errors


def unknown_rates(states: list[dict]) -> dict[tuple[str, str], float]:
    counts = Counter()
    unknown = Counter()
    for row in states:
        concept_id = row["concept_id"]
        route = row["route"]
        for slot in SLOTS:
            counts[(concept_id, route)] += 1
            if row.get("slots", {}).get(slot) == "unknown":
                unknown[(concept_id, route)] += 1
    return {key: unknown[key] / counts[key] for key in counts}


def build_concept_metrics(strict_dir: Path, samples_per_concept_route: int) -> list[dict]:
    states = read_jsonl(strict_dir / "strict_semantic_states.jsonl")
    entropy, errors = metric_maps(strict_dir)
    unknown = unknown_rates(states)
    concept_ids = sorted({row["concept_id"] for row in states})
    norm_denominator = math.log(samples_per_concept_route) if samples_per_concept_route > 1 else 1.0
    rows = []
    for concept_id in concept_ids:
        route_values = {}
        for route in ROUTES:
            slot_errors = [errors.get((concept_id, route, slot), 0.0) for slot in SLOTS]
            joint_entropy = entropy.get((concept_id, route, "joint"), 0.0)
            route_values[route] = {
                "joint_entropy": joint_entropy,
                "joint_normalized_entropy": joint_entropy / norm_denominator if norm_denominator else 0.0,
                "mean_slot_error": mean(slot_errors),
                "max_slot_error": max(slot_errors) if slot_errors else 0.0,
                "object_1_error": errors.get((concept_id, route, "object_1"), 0.0),
                "object_1_entropy": entropy.get((concept_id, route, "object_1"), 0.0),
                "unknown_rate": unknown.get((concept_id, route), 0.0),
            }
        difficulty_score = route_values["T2I"]["mean_slot_error"] + route_values["T2I"]["joint_normalized_entropy"]
        rows.append(
            {
                "concept_id": concept_id,
                "difficulty_score": difficulty_score,
                "i2t_joint_entropy": route_values["I2T"]["joint_entropy"],
                "t2i_joint_entropy": route_values["T2I"]["joint_entropy"],
                "delta_joint_entropy_t2i_minus_i2t": route_values["T2I"]["joint_entropy"] - route_values["I2T"]["joint_entropy"],
                "i2t_joint_normalized_entropy": route_values["I2T"]["joint_normalized_entropy"],
                "t2i_joint_normalized_entropy": route_values["T2I"]["joint_normalized_entropy"],
                "i2t_mean_slot_error": route_values["I2T"]["mean_slot_error"],
                "t2i_mean_slot_error": route_values["T2I"]["mean_slot_error"],
                "delta_mean_slot_error_t2i_minus_i2t": route_values["T2I"]["mean_slot_error"] - route_values["I2T"]["mean_slot_error"],
                "i2t_max_slot_error": route_values["I2T"]["max_slot_error"],
                "t2i_max_slot_error": route_values["T2I"]["max_slot_error"],
                "i2t_object_1_error": route_values["I2T"]["object_1_error"],
                "t2i_object_1_error": route_values["T2I"]["object_1_error"],
                "i2t_object_1_entropy": route_values["I2T"]["object_1_entropy"],
                "t2i_object_1_entropy": route_values["T2I"]["object_1_entropy"],
                "i2t_unknown_rate": route_values["I2T"]["unknown_rate"],
                "t2i_unknown_rate": route_values["T2I"]["unknown_rate"],
            }
        )
    rows.sort(key=lambda row: (row["difficulty_score"], row["concept_id"]))
    n = len(rows)
    tertile = max(1, n // 3)
    for idx, row in enumerate(rows):
        if idx < tertile:
            split = "easy"
        elif idx >= n - tertile:
            split = "hard"
        else:
            split = "medium"
        row["difficulty_rank"] = idx + 1
        row["difficulty_split"] = split
    return rows


def split_summary(rows: list[dict]) -> list[dict]:
    out = []
    for split in ["easy", "medium", "hard"]:
        subset = [row for row in rows if row["difficulty_split"] == split]
        out.append(
            {
                "difficulty_split": split,
                "concepts": len(subset),
                "mean_difficulty_score": mean([float(row["difficulty_score"]) for row in subset]),
                "mean_t2i_joint_entropy": mean([float(row["t2i_joint_entropy"]) for row in subset]),
                "mean_i2t_joint_entropy": mean([float(row["i2t_joint_entropy"]) for row in subset]),
                "mean_delta_joint_entropy": mean([float(row["delta_joint_entropy_t2i_minus_i2t"]) for row in subset]),
                "mean_t2i_mean_slot_error": mean([float(row["t2i_mean_slot_error"]) for row in subset]),
                "mean_i2t_mean_slot_error": mean([float(row["i2t_mean_slot_error"]) for row in subset]),
                "mean_delta_mean_slot_error": mean([float(row["delta_mean_slot_error_t2i_minus_i2t"]) for row in subset]),
                "mean_t2i_object_1_error": mean([float(row["t2i_object_1_error"]) for row in subset]),
                "mean_t2i_unknown_rate": mean([float(row["t2i_unknown_rate"]) for row in subset]),
            }
        )
    return out


def build_cases(rows: list[dict]) -> list[dict]:
    hard = sorted(
        [row for row in rows if row["difficulty_split"] == "hard"],
        key=lambda row: (-float(row["difficulty_score"]), row["concept_id"]),
    )
    easy = sorted(
        [row for row in rows if row["difficulty_split"] == "easy"],
        key=lambda row: (float(row["difficulty_score"]), row["concept_id"]),
    )
    case_rows = []
    for label, subset in [("hard", hard), ("easy", easy)]:
        for row in subset:
            case_rows.append(
                {
                    "case_type": label,
                    "concept_id": row["concept_id"],
                    "difficulty_score": row["difficulty_score"],
                    "t2i_joint_entropy": row["t2i_joint_entropy"],
                    "t2i_mean_slot_error": row["t2i_mean_slot_error"],
                    "t2i_object_1_error": row["t2i_object_1_error"],
                    "delta_joint_entropy_t2i_minus_i2t": row["delta_joint_entropy_t2i_minus_i2t"],
                    "delta_mean_slot_error_t2i_minus_i2t": row["delta_mean_slot_error_t2i_minus_i2t"],
                    "difficulty_split": row["difficulty_split"],
                }
            )
    return case_rows


def md_table(rows: list[dict], fields: list[str] | None = None) -> list[str]:
    if not rows:
        return ["(no rows)"]
    fields = fields or list(rows[0].keys())
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return lines


def write_report(out_dir: Path, rows: list[dict], summary: list[dict], cases: list[dict], args: argparse.Namespace) -> None:
    fields = [
        "concept_id",
        "difficulty_split",
        "difficulty_score",
        "t2i_joint_entropy",
        "t2i_mean_slot_error",
        "t2i_object_1_error",
        "delta_joint_entropy_t2i_minus_i2t",
        "delta_mean_slot_error_t2i_minus_i2t",
    ]
    hard_rows = [row for row in rows if row["difficulty_split"] == "hard"]
    easy_rows = [row for row in rows if row["difficulty_split"] == "easy"]
    hard_entropy = mean([float(row["t2i_joint_entropy"]) for row in hard_rows])
    easy_entropy = mean([float(row["t2i_joint_entropy"]) for row in easy_rows])
    hard_error = mean([float(row["t2i_mean_slot_error"]) for row in hard_rows])
    easy_error = mean([float(row["t2i_mean_slot_error"]) for row in easy_rows])
    lines = [
        "# Easy vs Hard Concept Split",
        "",
        "Status: RETROSPECTIVE_STRICT_OUTPUT_ANALYSIS",
        "",
        "This report splits concepts by T2I strict-output difficulty. It uses existing automated extractor outputs and does not establish human-level extractor validity.",
        "",
        "## Input Files",
        "",
        f"- `{args.strict_dir}/strict_route_entropy.csv`",
        f"- `{args.strict_dir}/strict_route_error.csv`",
        f"- `{args.strict_dir}/strict_semantic_states.jsonl`",
        "",
        "## Output Files",
        "",
        f"- `{out_dir / 'concept_difficulty_metrics.csv'}`",
        f"- `{out_dir / 'concept_difficulty_split_summary.csv'}`",
        f"- `{out_dir / 'concept_difficulty_cases.csv'}`",
        f"- `{out_dir / 'concept_difficulty_report.md'}`",
        "",
        "## Split Rule",
        "",
        f"- Difficulty score: `T2I mean slot error + T2I joint entropy / log({args.samples_per_concept_route})`.",
        "- Bottom third: easy; top third: hard; remaining concepts: medium.",
        "",
        "## Main Check",
        "",
        f"- Hard split mean T2I joint entropy: {hard_entropy:.4f}",
        f"- Easy split mean T2I joint entropy: {easy_entropy:.4f}",
        f"- Hard split mean T2I slot error: {hard_error:.4f}",
        f"- Easy split mean T2I slot error: {easy_error:.4f}",
        "",
        "## Split Summary",
        "",
        *md_table(summary),
        "",
        "## Concept Metrics",
        "",
        *md_table(rows, fields),
        "",
        "## Representative Cases",
        "",
        *md_table(cases),
        "",
        "## Claim Allowed After This Step",
        "",
        "Use this to show whether strict-output entropy/error concentrates in harder concepts rather than being uniform across the concept set.",
        "",
        "## Claim Still Not Allowed",
        "",
        "Do not claim true concept difficulty or measurement validity until manual extractor validation is complete.",
    ]
    (out_dir / "concept_difficulty_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    strict_dir = Path(args.strict_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = build_concept_metrics(strict_dir, args.samples_per_concept_route)
    summary = split_summary(rows)
    cases = build_cases(rows)
    write_csv(out_dir / "concept_difficulty_metrics.csv", rows)
    write_csv(out_dir / "concept_difficulty_split_summary.csv", summary)
    write_csv(out_dir / "concept_difficulty_cases.csv", cases)
    write_report(out_dir, rows, summary, cases, args)
    print(f"[INFO] wrote concept difficulty split for {len(rows)} concepts under {out_dir}", flush=True)


if __name__ == "__main__":
    main()
