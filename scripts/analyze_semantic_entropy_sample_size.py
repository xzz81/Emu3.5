#!/usr/bin/env python3
"""Sample-size sensitivity from existing strict semantic states.

This script does not run any model. It repeatedly subsamples existing
state-level outputs within each concept-route and recomputes paired route
deltas at the concept level.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path


SLOTS = ["object_1", "color_1", "object_2", "color_2", "relation", "background"]
ALL_SLOTS = ["joint", *SLOTS]
METRICS = ["entropy", "error_rate", "unknown_rate"]
ROUTES = ["I2T", "T2I"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-dir", default="outputs/semantic_entropy_umm/strict_compare")
    parser.add_argument("--out-dir", default="outputs/semantic_entropy_umm/sample_size_sensitivity")
    parser.add_argument("--sample-sizes", default="5,10")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20270605)
    return parser.parse_args()


def parse_int_list(text: str) -> list[int]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise ValueError("at least one sample size is required")
    return values


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


def entropy(values: list[tuple | str]) -> float:
    counts = Counter(values)
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    out = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            out -= p * math.log(p)
    return out


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def group_states(states: list[dict]) -> dict[tuple[str, str], list[dict]]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in states:
        row["sample_id"] = int(row["sample_id"])
        groups[(row["concept_id"], row["route"])].append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: row["sample_id"])
    return groups


def metrics_for_rows(rows: list[dict], slot: str) -> dict[str, float]:
    if not rows:
        return {"entropy": 0.0, "error_rate": 0.0, "unknown_rate": 0.0}
    target = rows[0]["target_semantics"]
    if slot == "joint":
        values = [tuple(row["slots"].get(name, "unknown") for name in SLOTS) for row in rows]
        errors = [any(row["slots"].get(name, "unknown") != target.get(name) for name in SLOTS) for row in rows]
        unknowns = [any(row["slots"].get(name, "unknown") == "unknown" for name in SLOTS) for row in rows]
    else:
        values = [row["slots"].get(slot, "unknown") for row in rows]
        errors = [row["slots"].get(slot, "unknown") != target.get(slot) for row in rows]
        unknowns = [row["slots"].get(slot, "unknown") == "unknown" for row in rows]
    return {
        "entropy": entropy(values),
        "error_rate": sum(errors) / len(errors),
        "unknown_rate": sum(unknowns) / len(unknowns),
    }


def stable_offset(*parts: object) -> int:
    text = ":".join(str(part) for part in parts)
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(text))


def run_sensitivity(states: list[dict], sample_sizes: list[int], iterations: int, seed: int):
    groups = group_states(states)
    concept_ids = sorted({concept_id for concept_id, _ in groups})
    concept_rows = []
    distribution_rows = []
    for sample_size in sample_sizes:
        for iteration in range(iterations):
            sampled_metrics: dict[tuple[str, str, str], dict[str, float]] = {}
            for concept_id in concept_ids:
                for route in ROUTES:
                    rows = groups[(concept_id, route)]
                    if len(rows) < sample_size:
                        raise ValueError(f"{concept_id}:{route} has {len(rows)} rows, cannot sample n={sample_size}")
                    rng = random.Random(seed + stable_offset(sample_size, iteration, concept_id, route))
                    sampled = rng.sample(rows, sample_size)
                    for slot in ALL_SLOTS:
                        sampled_metrics[(concept_id, route, slot)] = metrics_for_rows(sampled, slot)
            for concept_id in concept_ids:
                for slot in ALL_SLOTS:
                    for metric in METRICS:
                        i2t = sampled_metrics[(concept_id, "I2T", slot)][metric]
                        t2i = sampled_metrics[(concept_id, "T2I", slot)][metric]
                        concept_rows.append(
                            {
                                "sample_size": sample_size,
                                "iteration": iteration,
                                "concept_id": concept_id,
                                "slot": slot,
                                "metric": metric,
                                "i2t": i2t,
                                "t2i": t2i,
                                "delta_t2i_minus_i2t": t2i - i2t,
                            }
                        )
            by_slot_metric: dict[tuple[str, str], list[float]] = defaultdict(list)
            for concept_id in concept_ids:
                for slot in ALL_SLOTS:
                    for metric in METRICS:
                        i2t = sampled_metrics[(concept_id, "I2T", slot)][metric]
                        t2i = sampled_metrics[(concept_id, "T2I", slot)][metric]
                        by_slot_metric[(slot, metric)].append(t2i - i2t)
            for (slot, metric), values in by_slot_metric.items():
                distribution_rows.append(
                    {
                        "sample_size": sample_size,
                        "iteration": iteration,
                        "slot": slot,
                        "metric": metric,
                        "mean_delta_t2i_minus_i2t": sum(values) / len(values),
                        "median_delta_t2i_minus_i2t": statistics.median(values),
                    }
                )
    return concept_rows, distribution_rows


def summarize(distribution_rows: list[dict]) -> list[dict]:
    groups: dict[tuple[int, str, str], list[dict]] = defaultdict(list)
    for row in distribution_rows:
        groups[(int(row["sample_size"]), row["slot"], row["metric"])].append(row)
    summary = []
    for (sample_size, slot, metric), rows in sorted(groups.items()):
        values = [float(row["mean_delta_t2i_minus_i2t"]) for row in rows]
        summary.append(
            {
                "sample_size": sample_size,
                "slot": slot,
                "metric": metric,
                "iterations": len(values),
                "mean_delta": sum(values) / len(values),
                "median_delta": statistics.median(values),
                "ci_low": percentile(values, 0.025),
                "ci_high": percentile(values, 0.975),
                "min_delta": min(values),
                "max_delta": max(values),
            }
        )
    return summary


def write_report(out_dir: Path, summary: list[dict], states: list[dict], args: argparse.Namespace) -> None:
    key_rows = [
        row
        for row in summary
        if (row["slot"], row["metric"]) in {
            ("joint", "entropy"),
            ("object_1", "entropy"),
            ("joint", "error_rate"),
            ("object_1", "error_rate"),
        }
    ]
    concept_ids = sorted({row["concept_id"] for row in states})
    route_counts = Counter(row["route"] for row in states)
    lines = [
        "# Semantic Entropy Sample-Size Sensitivity",
        "",
        "Status: PRELIMINARY_EXTRACTOR_LIMITED",
        "",
        "This analysis reuses existing strict states only; it does not run Emu3.5.",
        "",
        "## Input Files",
        "",
        f"- `{args.strict_dir}/strict_semantic_states.jsonl`",
        "",
        "## Commands",
        "",
        "```bash",
        "./.venv-transformers/bin/python scripts/analyze_semantic_entropy_sample_size.py \\",
        f"  --strict-dir {args.strict_dir} \\",
        f"  --out-dir {args.out_dir} \\",
        f"  --sample-sizes {args.sample_sizes} \\",
        f"  --iterations {args.iterations} \\",
        f"  --seed {args.seed}",
        "```",
        "",
        "## Output Files",
        "",
        f"- `{args.out_dir}/sample_size_concept_iteration_metrics.csv`",
        f"- `{args.out_dir}/sample_size_delta_distributions.csv`",
        f"- `{args.out_dir}/sample_size_sensitivity_summary.csv`",
        f"- `{args.out_dir}/sample_size_sensitivity_report.md`",
        "",
        "## Sample Counts",
        "",
        f"- Strict state rows: `{len(states)}`",
        f"- Concepts: `{len(concept_ids)}`",
        f"- I2T state rows: `{route_counts.get('I2T', 0)}`",
        f"- T2I state rows: `{route_counts.get('T2I', 0)}`",
        f"- Summary rows: `{len(summary)}`",
        "",
        "## Method",
        "",
        f"- Input: `{args.strict_dir}/strict_semantic_states.jsonl`",
        f"- Sample sizes: `{args.sample_sizes}`",
        f"- Iterations per sample size: `{args.iterations}`",
        f"- Seed: `{args.seed}`",
        "- Sampling: without replacement within each concept-route from the existing 10 states.",
        "- Statistical unit: concept; route delta is T2I - I2T.",
        "",
        "## Pass/Fail Checks",
        "",
        f"- Input states present: `{bool(states)}`",
        f"- Both routes present: `{all(route_counts.get(route, 0) > 0 for route in ROUTES)}`",
        f"- Summary rows written: `{bool(summary)}`",
        f"- Requested sample sizes: `{args.sample_sizes}`",
        "",
        "## Key Metrics",
        "",
        "| N | Slot | Metric | Mean Delta | Median Delta | 95% CI | Min | Max |",
        "|---:|---|---|---:|---:|---|---:|---:|",
    ]
    for row in key_rows:
        lines.append(
            f"| {row['sample_size']} | {row['slot']} | {row['metric']} | "
            f"{row['mean_delta']:.4f} | {row['median_delta']:.4f} | "
            f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}] | {row['min_delta']:.4f} | {row['max_delta']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Claim Allowed After This Step",
            "",
            "This can support sample-size stability of the current automated-extractor pilot only. It does not replace human validation or full robustness controls.",
            "",
            "## Claim Still Not Allowed",
            "",
            "Do not claim final route-specific semantic instability, validated extractor quality, or full robustness from this sample-size resampling alone.",
        ]
    )
    (out_dir / "sample_size_sensitivity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    strict_dir = Path(args.strict_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    states = read_jsonl(strict_dir / "strict_semantic_states.jsonl")
    sample_sizes = parse_int_list(args.sample_sizes)
    concept_rows, distribution_rows = run_sensitivity(states, sample_sizes, args.iterations, args.seed)
    summary = summarize(distribution_rows)
    write_csv(
        out_dir / "sample_size_concept_iteration_metrics.csv",
        concept_rows,
        ["sample_size", "iteration", "concept_id", "slot", "metric", "i2t", "t2i", "delta_t2i_minus_i2t"],
    )
    write_csv(
        out_dir / "sample_size_delta_distributions.csv",
        distribution_rows,
        ["sample_size", "iteration", "slot", "metric", "mean_delta_t2i_minus_i2t", "median_delta_t2i_minus_i2t"],
    )
    write_csv(out_dir / "sample_size_sensitivity_summary.csv", summary)
    write_report(out_dir, summary, states, args)
    print(f"[INFO] wrote sample-size sensitivity for n={sample_sizes} under {out_dir}")


if __name__ == "__main__":
    main()
