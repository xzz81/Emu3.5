#!/usr/bin/env python3
"""Slot/family ablation summary for strict semantic entropy outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


SLOTS = ["object_1", "color_1", "object_2", "color_2", "relation", "background"]
ROUTES = ["I2T", "T2I"]
FAMILIES = {
    "object": ["object_1", "object_2"],
    "color": ["color_1", "color_2"],
    "relation": ["relation"],
    "background": ["background"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-dir", default="outputs/semantic_entropy_umm/strict_compare")
    parser.add_argument("--out-dir", default="outputs/semantic_entropy_umm/slot_ablation")
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


def route_slot_metrics(strict_dir: Path) -> list[dict]:
    entropy_rows = read_csv(strict_dir / "strict_route_entropy.csv")
    error_rows = read_csv(strict_dir / "strict_route_error.csv")
    states = read_jsonl(strict_dir / "strict_semantic_states.jsonl")
    error_by_key = {(row["concept_id"], row["route"], row["slot"]): float(row["error_rate"]) for row in error_rows}
    unknown_counts = Counter()
    total_counts = Counter()
    for row in states:
        for slot in SLOTS:
            key = (row["concept_id"], row["route"], slot)
            total_counts[key] += 1
            if row.get("slots", {}).get(slot) == "unknown":
                unknown_counts[key] += 1
    out = []
    for row in entropy_rows:
        slot = row["slot"]
        if slot == "joint":
            continue
        key = (row["concept_id"], row["route"], slot)
        out.append(
            {
                "concept_id": row["concept_id"],
                "route": row["route"],
                "slot": slot,
                "family": next(name for name, slots in FAMILIES.items() if slot in slots),
                "entropy": float(row["entropy"]),
                "error_rate": error_by_key.get(key, 0.0),
                "unknown_rate": unknown_counts[key] / total_counts[key] if total_counts[key] else 0.0,
                "num_samples": int(row["num_samples"]),
                "distribution": row["distribution"],
            }
        )
    return out


def summarize_slots(rows: list[dict]) -> list[dict]:
    out = []
    for slot in SLOTS:
        route_values = {}
        for route in ROUTES:
            subset = [row for row in rows if row["slot"] == slot and row["route"] == route]
            route_values[route] = {
                "entropy": mean([row["entropy"] for row in subset]),
                "error_rate": mean([row["error_rate"] for row in subset]),
                "unknown_rate": mean([row["unknown_rate"] for row in subset]),
            }
        out.append(
            {
                "level": "slot",
                "name": slot,
                "slots": slot,
                "i2t_entropy": route_values["I2T"]["entropy"],
                "t2i_entropy": route_values["T2I"]["entropy"],
                "delta_entropy_t2i_minus_i2t": route_values["T2I"]["entropy"] - route_values["I2T"]["entropy"],
                "i2t_error_rate": route_values["I2T"]["error_rate"],
                "t2i_error_rate": route_values["T2I"]["error_rate"],
                "delta_error_t2i_minus_i2t": route_values["T2I"]["error_rate"] - route_values["I2T"]["error_rate"],
                "i2t_unknown_rate": route_values["I2T"]["unknown_rate"],
                "t2i_unknown_rate": route_values["T2I"]["unknown_rate"],
                "delta_unknown_t2i_minus_i2t": route_values["T2I"]["unknown_rate"] - route_values["I2T"]["unknown_rate"],
            }
        )
    return out


def summarize_families(rows: list[dict]) -> list[dict]:
    out = []
    for family, slots in FAMILIES.items():
        route_values = {}
        for route in ROUTES:
            subset = [row for row in rows if row["slot"] in slots and row["route"] == route]
            route_values[route] = {
                "entropy": mean([row["entropy"] for row in subset]),
                "error_rate": mean([row["error_rate"] for row in subset]),
                "unknown_rate": mean([row["unknown_rate"] for row in subset]),
            }
        out.append(
            {
                "level": "family",
                "name": family,
                "slots": "+".join(slots),
                "i2t_entropy": route_values["I2T"]["entropy"],
                "t2i_entropy": route_values["T2I"]["entropy"],
                "delta_entropy_t2i_minus_i2t": route_values["T2I"]["entropy"] - route_values["I2T"]["entropy"],
                "i2t_error_rate": route_values["I2T"]["error_rate"],
                "t2i_error_rate": route_values["T2I"]["error_rate"],
                "delta_error_t2i_minus_i2t": route_values["T2I"]["error_rate"] - route_values["I2T"]["error_rate"],
                "i2t_unknown_rate": route_values["I2T"]["unknown_rate"],
                "t2i_unknown_rate": route_values["T2I"]["unknown_rate"],
                "delta_unknown_t2i_minus_i2t": route_values["T2I"]["unknown_rate"] - route_values["I2T"]["unknown_rate"],
            }
        )
    return out


def dominant(rows: list[dict], metric: str) -> dict:
    positives = [row for row in rows if row["level"] == "family" and float(row[metric]) > 0]
    if not positives:
        return {"name": "none", metric: 0.0, "share": 0.0}
    total = sum(float(row[metric]) for row in positives)
    top = max(positives, key=lambda row: float(row[metric]))
    return {"name": top["name"], metric: float(top[metric]), "share": float(top[metric]) / total if total else 0.0}


def md_table(rows: list[dict], fields: list[str] | None = None) -> list[str]:
    if not rows:
        return ["(no rows)"]
    fields = fields or list(rows[0].keys())
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return lines


def write_report(out_dir: Path, summary: list[dict], concept_rows: list[dict], args: argparse.Namespace) -> None:
    family_rows = [row for row in summary if row["level"] == "family"]
    slot_rows = [row for row in summary if row["level"] == "slot"]
    entropy_dom = dominant(summary, "delta_entropy_t2i_minus_i2t")
    error_dom = dominant(summary, "delta_error_t2i_minus_i2t")
    fields = [
        "level",
        "name",
        "slots",
        "i2t_entropy",
        "t2i_entropy",
        "delta_entropy_t2i_minus_i2t",
        "i2t_error_rate",
        "t2i_error_rate",
        "delta_error_t2i_minus_i2t",
        "i2t_unknown_rate",
        "t2i_unknown_rate",
    ]
    lines = [
        "# Slot Ablation Summary",
        "",
        "Status: RETROSPECTIVE_STRICT_OUTPUT_ANALYSIS",
        "",
        "This report groups existing strict outputs by semantic slot and slot family. It does not rerun extraction or establish human-level validity.",
        "",
        "## Input Files",
        "",
        f"- `{args.strict_dir}/strict_route_entropy.csv`",
        f"- `{args.strict_dir}/strict_route_error.csv`",
        f"- `{args.strict_dir}/strict_semantic_states.jsonl`",
        "",
        "## Output Files",
        "",
        f"- `{out_dir / 'slot_ablation_concept_metrics.csv'}`",
        f"- `{out_dir / 'slot_ablation_summary.csv'}`",
        f"- `{out_dir / 'slot_ablation_report.md'}`",
        "",
        "## Main Finding",
        "",
        f"- Dominant positive delta entropy family: `{entropy_dom['name']}` ({entropy_dom['share']:.4f} share among positive family deltas).",
        f"- Dominant positive delta error family: `{error_dom['name']}` ({error_dom['share']:.4f} share among positive family deltas).",
        "",
        "## Family Summary",
        "",
        *md_table(family_rows, fields),
        "",
        "## Slot Summary",
        "",
        *md_table(slot_rows, fields),
        "",
        "## Claim Allowed After This Step",
        "",
        "Use this to localize current strict-output entropy/error deltas to object/color/relation/background families.",
        "",
        "## Claim Still Not Allowed",
        "",
        "Do not claim the localized source is a true model behavior rather than extractor behavior until manual extractor validation passes.",
    ]
    (out_dir / "slot_ablation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    concept_rows = route_slot_metrics(Path(args.strict_dir))
    summary = summarize_families(concept_rows) + summarize_slots(concept_rows)
    write_csv(out_dir / "slot_ablation_concept_metrics.csv", concept_rows)
    write_csv(out_dir / "slot_ablation_summary.csv", summary)
    write_report(out_dir, summary, concept_rows, args)
    print(f"[INFO] wrote slot ablation for {len(concept_rows)} concept-route-slot rows under {out_dir}")


if __name__ == "__main__":
    main()
