#!/usr/bin/env python3
"""Bootstrap concept-level paired I2T/T2I semantic entropy metrics."""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


SLOTS = ["joint", "object_1", "color_1", "object_2", "color_2", "relation", "background"]
METRICS = [
    ("entropy", "entropy"),
    ("error_rate", "error_rate"),
    ("unknown_rate", "unknown_rate"),
]
ROUTES = ["I2T", "T2I"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quadrant-dir", default="outputs/semantic_entropy_umm/quadrant_analysis")
    parser.add_argument("--out-dir", default="outputs/semantic_entropy_umm/bootstrap_stability")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20270605)
    return parser.parse_args()


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


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def wilcoxon_signed_rank(deltas: list[float]) -> tuple[float, float, str, int]:
    nonzero = [delta for delta in deltas if abs(delta) > 1e-12]
    n = len(nonzero)
    if n == 0:
        return 0.0, 1.0, "all_zero_deltas", 0
    try:
        from scipy.stats import wilcoxon  # type: ignore

        result = wilcoxon(nonzero, zero_method="wilcox", alternative="two-sided", method="auto")
        return float(result.statistic), float(result.pvalue), "scipy", n
    except Exception:
        abs_values = [abs(delta) for delta in nonzero]
        ranks = average_ranks(abs_values)
        w_plus = sum(rank for rank, delta in zip(ranks, nonzero) if delta > 0)
        w_minus = sum(rank for rank, delta in zip(ranks, nonzero) if delta < 0)
        statistic = min(w_plus, w_minus)
        mean = n * (n + 1) / 4
        var = n * (n + 1) * (2 * n + 1) / 24
        z = (statistic - mean) / math.sqrt(var) if var > 0 else 0.0
        p = min(1.0, 2 * normal_cdf(z))
        return statistic, p, "normal_approx_no_tie_correction", n


def build_paired_rows(quadrant_dir: Path) -> list[dict]:
    metrics = read_csv(quadrant_dir / "concept_route_metrics.csv")
    by_key: dict[tuple[str, str, str], dict] = {}
    for row in metrics:
        by_key[(row["concept_id"], row["route"], row["slot"])] = row
    concept_ids = sorted({row["concept_id"] for row in metrics})
    paired = []
    for concept_id in concept_ids:
        for slot in SLOTS:
            i2t = by_key[(concept_id, "I2T", slot)]
            t2i = by_key[(concept_id, "T2I", slot)]
            for metric_name, field in METRICS:
                i2t_value = float(i2t[field])
                t2i_value = float(t2i[field])
                paired.append(
                    {
                        "concept_id": concept_id,
                        "slot": slot,
                        "metric": metric_name,
                        "i2t": i2t_value,
                        "t2i": t2i_value,
                        "delta_t2i_minus_i2t": t2i_value - i2t_value,
                    }
                )
    return paired


def bootstrap(values: list[dict], iterations: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    n = len(values)
    out = []
    deltas = [float(row["delta_t2i_minus_i2t"]) for row in values]
    for _ in range(iterations):
        total = 0.0
        for _ in range(n):
            total += deltas[rng.randrange(n)]
        out.append(total / n if n else 0.0)
    return out


def stable_seed_offset(slot: str, metric: str) -> int:
    text = f"{slot}:{metric}"
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(text))


def summarize(paired: list[dict], iterations: int, seed: int) -> tuple[list[dict], list[dict]]:
    by_metric: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in paired:
        by_metric[(row["slot"], row["metric"])].append(row)
    summary = []
    distributions = []
    for (slot, metric), rows in sorted(by_metric.items()):
        deltas = [float(row["delta_t2i_minus_i2t"]) for row in rows]
        samples = bootstrap(rows, iterations, seed + stable_seed_offset(slot, metric))
        statistic, pvalue, method, n_eff = wilcoxon_signed_rank(deltas)
        mean_i2t = sum(float(row["i2t"]) for row in rows) / len(rows)
        mean_t2i = sum(float(row["t2i"]) for row in rows) / len(rows)
        mean_delta = sum(deltas) / len(deltas)
        summary.append(
            {
                "slot": slot,
                "metric": metric,
                "n_concepts": len(rows),
                "mean_i2t": mean_i2t,
                "mean_t2i": mean_t2i,
                "mean_delta_t2i_minus_i2t": mean_delta,
                "median_delta_t2i_minus_i2t": statistics.median(deltas),
                "bootstrap_ci_low": percentile(samples, 0.025),
                "bootstrap_ci_high": percentile(samples, 0.975),
                "wilcoxon_statistic": statistic,
                "wilcoxon_pvalue": pvalue,
                "wilcoxon_method": method,
                "wilcoxon_n_nonzero": n_eff,
                "effective_states_i2t": math.exp(mean_i2t) if metric == "entropy" else "",
                "effective_states_t2i": math.exp(mean_t2i) if metric == "entropy" else "",
            }
        )
        for iteration, value in enumerate(samples):
            distributions.append(
                {
                    "slot": slot,
                    "metric": metric,
                    "iteration": iteration,
                    "delta_t2i_minus_i2t": value,
                }
            )
    return summary, distributions


def write_report(out_dir: Path, paired: list[dict], summary: list[dict], args: argparse.Namespace) -> None:
    lookup = {(row["slot"], row["metric"]): row for row in summary}
    concept_ids = sorted({row["concept_id"] for row in paired})
    expected_paired_rows = len(concept_ids) * len(SLOTS) * len(METRICS)
    summary_keys = {(row["slot"], row["metric"]) for row in summary}
    expected_summary_keys = {(slot, metric_name) for slot in SLOTS for metric_name, _ in METRICS}
    required_key_pairs = [("joint", "entropy"), ("object_1", "entropy"), ("joint", "error_rate"), ("object_1", "error_rate")]
    required_keys_present = all(key in lookup for key in required_key_pairs)
    lines = [
        "# Semantic Entropy Bootstrap Stability",
        "",
        "Status: PRELIMINARY_EXTRACTOR_LIMITED",
        "",
        "This report uses concept-level paired metrics from strict automated extractor outputs. It should be updated after human/extractor validation is scored.",
        "",
        "## Input Files",
        "",
        f"- `{args.quadrant_dir}/concept_route_metrics.csv`",
        "",
        "## Commands",
        "",
        "```bash",
        f"./.venv-transformers/bin/python scripts/analyze_semantic_entropy_bootstrap.py --iterations {args.iterations} --seed {args.seed}",
        "```",
        "",
        "## Output Files",
        "",
        f"- `{out_dir / 'concept_level_paired_metrics.csv'}`",
        f"- `{out_dir / 'bootstrap_summary.csv'}`",
        f"- `{out_dir / 'bootstrap_delta_distributions.csv'}`",
        f"- `{out_dir / 'bootstrap_report.md'}`",
        "",
        "## Sample Counts",
        "",
        f"- Concepts: `{len(concept_ids)}`",
        f"- Slots including joint: `{len(SLOTS)}`",
        f"- Metrics per slot: `{len(METRICS)}`",
        f"- Concept-level paired rows: `{len(paired)}` / expected `{expected_paired_rows}`",
        f"- Bootstrap summary rows: `{len(summary)}` / expected `{len(expected_summary_keys)}`",
        f"- Bootstrap iterations per slot-metric: `{args.iterations}`",
        "",
        "## Pass/Fail Checks",
        "",
        f"- Statistical unit is concept: PASS",
        f"- Paired row count matches concept-slot-metric design: {'PASS' if len(paired) == expected_paired_rows else 'FAIL'}",
        f"- Summary covers joint and all six slots for entropy/error/unknown: {'PASS' if summary_keys == expected_summary_keys else 'FAIL'}",
        f"- Joint and object_1 entropy/error entries present: {'PASS' if required_keys_present else 'FAIL'}",
        f"- Bootstrap iterations configured to at least 10000: {'PASS' if args.iterations >= 10000 else 'FAIL'}",
        "",
        "## Method",
        "",
        "- Statistical unit: concept.",
        "- Paired delta: metric(T2I) - metric(I2T).",
        f"- Bootstrap iterations: `{args.iterations}`.",
        f"- Seed: `{args.seed}`.",
        "- Wilcoxon signed-rank test uses SciPy when available, otherwise a normal approximation.",
        "",
        "## Key Metrics",
        "",
        "| Slot | Metric | Mean I2T | Mean T2I | Mean Delta | Median Delta | 95% CI Low | 95% CI High | Wilcoxon p |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    key_pairs = [("joint", "entropy"), ("object_1", "entropy"), ("joint", "error_rate"), ("object_1", "error_rate")]
    for key in key_pairs:
        row = lookup[key]
        lines.append(
            f"| {row['slot']} | {row['metric']} | {row['mean_i2t']:.4f} | {row['mean_t2i']:.4f} | "
            f"{row['mean_delta_t2i_minus_i2t']:.4f} | {row['median_delta_t2i_minus_i2t']:.4f} | "
            f"{row['bootstrap_ci_low']:.4f} | {row['bootstrap_ci_high']:.4f} | {row['wilcoxon_pvalue']:.4g} |"
        )
    lines.extend(
        [
            "",
            "## Full Summary",
            "",
            "| Slot | Metric | N | Mean Delta | 95% CI | Wilcoxon p | Method |",
            "|---|---|---:|---:|---|---:|---|",
        ]
    )
    for row in summary:
        lines.append(
            f"| {row['slot']} | {row['metric']} | {row['n_concepts']} | {row['mean_delta_t2i_minus_i2t']:.4f} | "
            f"[{row['bootstrap_ci_low']:.4f}, {row['bootstrap_ci_high']:.4f}] | {row['wilcoxon_pvalue']:.4g} | {row['wilcoxon_method']} |"
        )
    lines.extend(
        [
            "",
            "## Claim Allowed After This Step",
            "",
            "The strict automated-extractor outputs can be summarized with concept-level paired bootstrap uncertainty under the fixed semantic-state protocol.",
            "",
            "## Claim Implication",
            "",
            "If a confidence interval crosses 0, the route-difference claim must be downgraded to protocol comparability with a weak pilot trend.",
            "",
            "## Claim Still Not Allowed",
            "",
            "Do not claim route-specific semantic instability until extractor validation and at least one robustness control are complete.",
        ]
    )
    (out_dir / "bootstrap_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paired = build_paired_rows(Path(args.quadrant_dir))
    summary, distributions = summarize(paired, args.iterations, args.seed)
    write_csv(
        out_dir / "concept_level_paired_metrics.csv",
        paired,
        ["concept_id", "slot", "metric", "i2t", "t2i", "delta_t2i_minus_i2t"],
    )
    write_csv(out_dir / "bootstrap_summary.csv", summary)
    write_csv(out_dir / "bootstrap_delta_distributions.csv", distributions)
    write_report(out_dir, paired, summary, args)
    print(f"[INFO] wrote {len(summary)} bootstrap summaries and {len(distributions)} samples under {out_dir}")


if __name__ == "__main__":
    main()
