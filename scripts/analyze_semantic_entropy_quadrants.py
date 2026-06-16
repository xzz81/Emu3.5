#!/usr/bin/env python3
"""Analyze strict semantic entropy, error, unknowns, and quadrants.

This is a no-model analysis pass over strict_compare outputs. Until human
extractor validation is scored, its report should be treated as
extractor-limited evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


SLOTS = ["object_1", "color_1", "object_2", "color_2", "relation", "background"]
ROUTES = ["I2T", "T2I"]
VOCAB = {
    "object_1": ["cube", "sphere", "cone", "unknown"],
    "object_2": ["cube", "sphere", "cone", "unknown"],
    "color_1": ["red", "blue", "green", "yellow", "unknown"],
    "color_2": ["red", "blue", "green", "yellow", "unknown"],
    "relation": [
        "object_1_left_of_object_2",
        "object_1_right_of_object_2",
        "object_1_above_object_2",
        "object_1_below_object_2",
        "unknown",
    ],
    "background": ["white", "other", "unknown"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-dir", default="outputs/semantic_entropy_umm/strict_compare")
    parser.add_argument("--out-dir", default="outputs/semantic_entropy_umm/quadrant_analysis")
    parser.add_argument("--error-threshold", type=float, default=0.5)
    parser.add_argument("--case-limit-per-route", type=int, default=5)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict]:
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


def support_size(slot: str) -> int:
    if slot == "joint":
        value = 1
        for name in SLOTS:
            value *= len(VOCAB[name])
        return value
    return len(VOCAB[slot])


def normalized_entropy(entropy: float, slot: str, num_samples: int) -> float:
    # With n sampled states, the empirical entropy cannot exceed log(n). Use
    # min(vocabulary support, n) as the achievable maximum for this estimate.
    denom = math.log(max(1, min(support_size(slot), num_samples)))
    return entropy / denom if denom > 0 else 0.0


def state_key(row: dict) -> tuple[str, str, int]:
    return row["concept_id"], row["route"], int(row["sample_id"])


def build_indexes(strict_dir: Path):
    states = read_jsonl(strict_dir / "strict_semantic_states.jsonl")
    answers = read_jsonl(strict_dir / "strict_slot_answers.jsonl")
    entropy_rows = read_csv(strict_dir / "strict_route_entropy.csv")
    error_rows = read_csv(strict_dir / "strict_route_error.csv")
    states_by_cr: dict[tuple[str, str], list[dict]] = defaultdict(list)
    answers_by_crs: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in states:
        row["sample_id"] = int(row["sample_id"])
        states_by_cr[(row["concept_id"], row["route"])].append(row)
    for row in answers:
        row["sample_id"] = int(row["sample_id"])
        answers_by_crs[(row["concept_id"], row["route"], row["slot"])].append(row)
    for rows in states_by_cr.values():
        rows.sort(key=lambda row: row["sample_id"])
    for rows in answers_by_crs.values():
        rows.sort(key=lambda row: row["sample_id"])
    error_by_crs = {(row["concept_id"], row["route"], row["slot"]): row for row in error_rows}
    return states_by_cr, answers_by_crs, entropy_rows, error_by_crs


def invalid_parse_rate(answer_rows: list[dict]) -> float:
    if not answer_rows:
        return 0.0
    bad = 0
    for row in answer_rows:
        slot = row["slot"]
        allowed = row.get("allowed_values") or VOCAB[slot]
        if isinstance(allowed, str):
            try:
                allowed = json.loads(allowed)
            except json.JSONDecodeError:
                allowed = VOCAB[slot]
        bad += row.get("value") not in allowed
    return bad / len(answer_rows)


def error_for_state(state: dict, slot: str) -> bool:
    target = state["target_semantics"]
    if slot == "joint":
        return any(state["slots"].get(name, "unknown") != target.get(name) for name in SLOTS)
    return state["slots"].get(slot, "unknown") != target.get(slot)


def unknown_for_state(state: dict, slot: str) -> bool:
    if slot == "joint":
        return any(state["slots"].get(name, "unknown") == "unknown" for name in SLOTS)
    return state["slots"].get(slot, "unknown") == "unknown"


def build_concept_route_metrics(strict_dir: Path, error_threshold: float) -> list[dict]:
    states_by_cr, answers_by_crs, entropy_rows, error_by_crs = build_indexes(strict_dir)
    metrics = []
    for row in entropy_rows:
        concept_id = row["concept_id"]
        route = row["route"]
        slot = row["slot"]
        cr_states = states_by_cr[(concept_id, route)]
        num_samples = int(row["num_samples"])
        entropy = float(row["entropy"])
        if slot == "joint":
            error_rate = sum(error_for_state(state, slot) for state in cr_states) / len(cr_states)
            unknown_rate = sum(unknown_for_state(state, slot) for state in cr_states) / len(cr_states)
            answer_rows = [
                answer
                for name in SLOTS
                for answer in answers_by_crs.get((concept_id, route, name), [])
            ]
            invalid_rate = invalid_parse_rate(answer_rows)
        else:
            err = error_by_crs[(concept_id, route, slot)]
            error_rate = float(err["error_rate"])
            answer_rows = answers_by_crs.get((concept_id, route, slot), [])
            unknown_rate = sum(answer.get("value") == "unknown" for answer in answer_rows) / len(answer_rows) if answer_rows else 0.0
            invalid_rate = invalid_parse_rate(answer_rows)
        norm = normalized_entropy(entropy, slot, num_samples)
        metrics.append(
            {
                "concept_id": concept_id,
                "route": route,
                "slot": slot,
                "entropy": entropy,
                "normalized_entropy": norm,
                "effective_num_states": math.exp(entropy),
                "error_rate": error_rate,
                "unknown_rate": unknown_rate,
                "invalid_parse_rate": invalid_rate,
                "quadrant": "",
                "num_samples": num_samples,
                "distribution": row["distribution"],
            }
        )
    medians = {}
    for slot in ["joint", *SLOTS]:
        values = [row["normalized_entropy"] for row in metrics if row["slot"] == slot]
        medians[slot] = statistics.median(values) if values else 0.0
    for row in metrics:
        entropy_band = "low" if row["normalized_entropy"] <= medians[row["slot"]] else "high"
        error_band = "low" if row["error_rate"] <= error_threshold else "high"
        row["quadrant"] = f"{entropy_band}_entropy_{error_band}_error"
    return metrics


def summarize_quadrants(metrics: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in metrics:
        groups[(row["route"], row["slot"], row["quadrant"])].append(row)
    out = []
    for (route, slot, quadrant), rows in sorted(groups.items()):
        out.append(
            {
                "route": route,
                "slot": slot,
                "quadrant": quadrant,
                "count": len(rows),
                "mean_entropy": sum(row["entropy"] for row in rows) / len(rows),
                "mean_normalized_entropy": sum(row["normalized_entropy"] for row in rows) / len(rows),
                "mean_effective_num_states": sum(row["effective_num_states"] for row in rows) / len(rows),
                "mean_error_rate": sum(row["error_rate"] for row in rows) / len(rows),
                "mean_unknown_rate": sum(row["unknown_rate"] for row in rows) / len(rows),
                "mean_invalid_parse_rate": sum(row["invalid_parse_rate"] for row in rows) / len(rows),
            }
        )
    return out


def representative_cases(strict_dir: Path, metrics: list[dict], limit_per_route: int) -> list[dict]:
    states_by_cr, answers_by_crs, _, _ = build_indexes(strict_dir)
    candidates = sorted(
        metrics,
        key=lambda row: (
            row["quadrant"] != "low_entropy_high_error",
            -row["error_rate"],
            row["normalized_entropy"],
            row["route"],
            row["concept_id"],
            row["slot"],
        ),
    )
    counts = Counter()
    cases = []
    for metric in candidates:
        route = metric["route"]
        if counts[route] >= limit_per_route:
            continue
        concept_id = metric["concept_id"]
        slot = metric["slot"]
        rows = states_by_cr[(concept_id, route)]
        bad_states = [row for row in rows if error_for_state(row, slot)]
        if not bad_states:
            continue
        state = bad_states[0]
        if slot == "joint":
            wrong_slots = [name for name in SLOTS if error_for_state(state, name)]
            first_slot = wrong_slots[0] if wrong_slots else SLOTS[0]
            answer = next((row for row in answers_by_crs[(concept_id, route, first_slot)] if row["sample_id"] == state["sample_id"]), {})
            target_value = json.dumps(state["target_semantics"], ensure_ascii=False)
            predicted_value = json.dumps(state["slots"], ensure_ascii=False)
            raw_output = answer.get("raw_output", "")
        else:
            answer = next((row for row in answers_by_crs[(concept_id, route, slot)] if row["sample_id"] == state["sample_id"]), {})
            target_value = state["target_semantics"].get(slot)
            predicted_value = state["slots"].get(slot, "unknown")
            raw_output = answer.get("raw_output", "")
        cases.append(
            {
                "route": route,
                "concept_id": concept_id,
                "slot": slot,
                "quadrant": metric["quadrant"],
                "error_rate": metric["error_rate"],
                "normalized_entropy": metric["normalized_entropy"],
                "sample_id": state["sample_id"],
                "image_path": state["image_path"],
                "target_value": target_value,
                "predicted_value": predicted_value,
                "raw_output": raw_output,
            }
        )
        counts[route] += 1
    return cases


def draw_scatter(path: Path, metrics: list[dict]) -> None:
    w, h = 900, 620
    margin_l, margin_b, margin_t, margin_r = 80, 70, 50, 30
    plot_w = w - margin_l - margin_r
    plot_h = h - margin_t - margin_b
    image = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((margin_l, 16), "Semantic entropy quadrants: normalized entropy vs error", fill=(0, 0, 0), font=font)
    draw.line((margin_l, margin_t + plot_h, margin_l + plot_w, margin_t + plot_h), fill=(0, 0, 0), width=2)
    draw.line((margin_l, margin_t, margin_l, margin_t + plot_h), fill=(0, 0, 0), width=2)
    for tick in range(6):
        x = margin_l + plot_w * tick / 5
        y = margin_t + plot_h - plot_h * tick / 5
        draw.line((x, margin_t + plot_h, x, margin_t + plot_h + 5), fill=(0, 0, 0))
        draw.text((x - 10, margin_t + plot_h + 12), f"{tick/5:.1f}", fill=(0, 0, 0), font=font)
        draw.line((margin_l - 5, y, margin_l, y), fill=(0, 0, 0))
        draw.text((margin_l - 42, y - 6), f"{tick/5:.1f}", fill=(0, 0, 0), font=font)
    draw.text((margin_l + plot_w // 2 - 60, h - 28), "normalized entropy", fill=(0, 0, 0), font=font)
    draw.text((8, margin_t + plot_h // 2), "error", fill=(0, 0, 0), font=font)
    colors = {"I2T": (42, 93, 180), "T2I": (202, 76, 43)}
    for row in metrics:
        x = margin_l + min(1.0, max(0.0, row["normalized_entropy"])) * plot_w
        y = margin_t + plot_h - min(1.0, max(0.0, row["error_rate"])) * plot_h
        color = colors.get(row["route"], (80, 80, 80))
        radius = 3 if row["slot"] != "joint" else 5
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    draw.text((margin_l + plot_w - 170, 18), "I2T", fill=colors["I2T"], font=font)
    draw.text((margin_l + plot_w - 120, 18), "T2I", fill=colors["T2I"], font=font)
    image.save(path)


def mean_by_route_slot(metrics: list[dict], field: str) -> dict[tuple[str, str], float]:
    out = {}
    for route in ROUTES:
        for slot in ["joint", *SLOTS]:
            vals = [row[field] for row in metrics if row["route"] == route and row["slot"] == slot]
            if vals:
                out[(route, slot)] = sum(vals) / len(vals)
    return out


def write_report(out_dir: Path, metrics: list[dict], summary: list[dict], cases: list[dict], args: argparse.Namespace) -> None:
    err = mean_by_route_slot(metrics, "error_rate")
    unk = mean_by_route_slot(metrics, "unknown_rate")
    ent = mean_by_route_slot(metrics, "entropy")
    norm = mean_by_route_slot(metrics, "normalized_entropy")
    concept_ids = sorted({row["concept_id"] for row in metrics})
    route_slot_rows = len(metrics)
    expected_route_slot_rows = len(concept_ids) * len(ROUTES) * (len(SLOTS) + 1)
    joint_rows = [row for row in metrics if row["slot"] == "joint"]
    all_required_metrics = all(
        key in row and row[key] not in ("", None)
        for row in metrics
        for key in ["entropy", "error_rate", "unknown_rate", "invalid_parse_rate", "normalized_entropy"]
    )
    route_case_counts = Counter(row["route"] for row in cases)
    object1_delta = err.get(("T2I", "object_1"), 0.0) - err.get(("I2T", "object_1"), 0.0)
    other_slots = [slot for slot in SLOTS if slot != "object_1"]
    other_delta_max = max((abs(err.get(("T2I", slot), 0.0) - err.get(("I2T", slot), 0.0)) for slot in other_slots), default=0.0)
    lines = [
        "# Semantic Entropy Quadrant Analysis",
        "",
        "Status: PRELIMINARY_EXTRACTOR_LIMITED",
        "",
        "This report uses automated strict extractor outputs. It should be updated after human/extractor validation is scored.",
        "",
        "## Input Files",
        "",
        f"- `{args.strict_dir}/strict_route_entropy.csv`",
        f"- `{args.strict_dir}/strict_route_error.csv`",
        f"- `{args.strict_dir}/strict_semantic_states.jsonl`",
        f"- `{args.strict_dir}/strict_slot_answers.jsonl`",
        "",
        "## Commands",
        "",
        "```bash",
        f"./.venv-transformers/bin/python scripts/analyze_semantic_entropy_quadrants.py --strict-dir {args.strict_dir} --out-dir {out_dir} --error-threshold {args.error_threshold} --case-limit-per-route {args.case_limit_per_route}",
        "```",
        "",
        "## Output Files",
        "",
        f"- `{out_dir / 'concept_route_metrics.csv'}`",
        f"- `{out_dir / 'slot_quadrant_metrics.csv'}`",
        f"- `{out_dir / 'quadrant_cases.csv'}`",
        f"- `{out_dir / 'quadrant_scatter.png'}`",
        f"- `{out_dir / 'quadrant_report.md'}`",
        "",
        "## Sample Counts",
        "",
        f"- Concepts: `{len(concept_ids)}`",
        f"- Routes: `{len(ROUTES)}`",
        f"- Slots including joint: `{len(SLOTS) + 1}`",
        f"- Concept-route-slot metric rows: `{route_slot_rows}` / expected `{expected_route_slot_rows}`",
        f"- Joint concept-route rows: `{len(joint_rows)}`",
        f"- Representative failure cases: `{len(cases)}`",
        "",
        "## Pass/Fail Checks",
        "",
        f"- Every concept-route-slot has entropy/error/unknown/invalid/normalized metrics: {'PASS' if all_required_metrics else 'FAIL'}",
        f"- Concept-route-slot row count matches protocol: {'PASS' if route_slot_rows == expected_route_slot_rows else 'FAIL'}",
        f"- Joint and six slots represented: {'PASS' if sorted({row['slot'] for row in metrics}) == sorted(['joint', *SLOTS]) else 'FAIL'}",
        f"- At least {args.case_limit_per_route} representative cases per route when available: {'PASS' if all(route_case_counts.get(route, 0) >= args.case_limit_per_route for route in ROUTES) else 'FAIL'}",
        "",
        "## Method",
        "",
        "- `normalized_entropy = H / log(min(slot_support_size, num_samples))`.",
        f"- Error threshold for low/high: `{args.error_threshold}`.",
        "- Entropy threshold for low/high: dataset median within each slot.",
        "- `joint` error means any slot in the state mismatched gold semantics.",
        "- `joint` unknown means any slot in the state was `unknown`.",
        "",
        "## Route Summary",
        "",
        "| Slot | H I2T | H T2I | NormH I2T | NormH T2I | Err I2T | Err T2I | Unknown I2T | Unknown T2I |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for slot in ["joint", *SLOTS]:
        lines.append(
            f"| {slot} | {ent.get(('I2T', slot), 0):.4f} | {ent.get(('T2I', slot), 0):.4f} | "
            f"{norm.get(('I2T', slot), 0):.4f} | {norm.get(('T2I', slot), 0):.4f} | "
            f"{err.get(('I2T', slot), 0):.4f} | {err.get(('T2I', slot), 0):.4f} | "
            f"{unk.get(('I2T', slot), 0):.4f} | {unk.get(('T2I', slot), 0):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Concentration Checks",
            "",
            f"- T2I object_1 error minus I2T object_1 error: `{object1_delta:.4f}`.",
            f"- Max absolute route error delta among other slots: `{other_delta_max:.4f}`.",
            f"- Unknown rate max: `{max((row['unknown_rate'] for row in metrics), default=0.0):.4f}`.",
            f"- Invalid parse rate max: `{max((row['invalid_parse_rate'] for row in metrics), default=0.0):.4f}`.",
            "",
            "## Quadrant Distribution",
            "",
            "| Route | Slot | Quadrant | Count | Mean NormH | Mean Err | Mean Unknown |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary:
        lines.append(
            f"| {row['route']} | {row['slot']} | {row['quadrant']} | {row['count']} | "
            f"{row['mean_normalized_entropy']:.4f} | {row['mean_error_rate']:.4f} | {row['mean_unknown_rate']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Representative Failure Cases",
            "",
            "| Route | Concept | Slot | Quadrant | Error | NormH | Sample | Target | Predicted | Image |",
            "|---|---|---|---|---:|---:|---:|---|---|---|",
        ]
    )
    for row in cases:
        lines.append(
            f"| {row['route']} | {row['concept_id']} | {row['slot']} | {row['quadrant']} | "
            f"{row['error_rate']:.4f} | {row['normalized_entropy']:.4f} | {row['sample_id']} | "
            f"`{row['target_value']}` | `{row['predicted_value']}` | `{row['image_path']}` |"
        )
    lines.extend(
        [
            "",
            "## Claim Allowed After This Step",
            "",
            "The strict outputs can be described in error/unknown/entropy quadrants under the automated extractor.",
            "",
            "## Claim Still Not Allowed",
            "",
            "Do not claim extractor validity or route-specific semantic instability until human validation and bootstrap/robustness checks are complete.",
        ]
    )
    (out_dir / "quadrant_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    strict_dir = Path(args.strict_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = build_concept_route_metrics(strict_dir, args.error_threshold)
    summary = summarize_quadrants(metrics)
    cases = representative_cases(strict_dir, metrics, args.case_limit_per_route)
    fields = [
        "concept_id",
        "route",
        "slot",
        "entropy",
        "normalized_entropy",
        "effective_num_states",
        "error_rate",
        "unknown_rate",
        "invalid_parse_rate",
        "quadrant",
        "num_samples",
        "distribution",
    ]
    write_csv(out_dir / "concept_route_metrics.csv", metrics, fields)
    write_csv(out_dir / "slot_quadrant_metrics.csv", summary)
    write_csv(out_dir / "quadrant_cases.csv", cases)
    draw_scatter(out_dir / "quadrant_scatter.png", metrics)
    write_report(out_dir, metrics, summary, cases, args)
    print(f"[INFO] wrote {len(metrics)} concept-route-slot rows under {out_dir}")


if __name__ == "__main__":
    main()
