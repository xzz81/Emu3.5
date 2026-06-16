#!/usr/bin/env python3
"""Join Codex visual GT with T2I semantic-uncertainty datapoints.

The existing T2I detector analysis uses automatic readback/slot errors as the
hallucination label. This script evaluates the same uncertainty scores against
first-pass visual GT labels, so the detector target is true image faithfulness
rather than readback mismatch.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCORE_SPECS = [
    ("semantic_entropy", "cluster_assignment_entropy"),
    ("normalized_entropy", "normalized_entropy"),
    ("effective_semantic_clusters", "effective_semantic_clusters"),
    ("auto_readback_error_rate", "error_rate"),
    ("auto_readback_unknown_rate", "unknown_rate"),
]

VISUAL_LABEL_SPECS = [
    ("strict_any_visual_hallucination", "strict_any_visual_hallucination"),
    ("core_any_visual_hallucination", "core_any_visual_hallucination"),
    ("background_any_visual_issue", "background_any_visual_issue"),
    ("strict_majority_visual_hallucination", "strict_majority_visual_hallucination"),
    ("core_majority_visual_hallucination", "core_majority_visual_hallucination"),
]

AUTO_LABEL_SPECS = [
    ("auto_any_error", "any_error_label"),
    ("auto_majority_error", "majority_error_label"),
    ("auto_mode_error", "mode_error_label"),
    ("auto_first_sample_error", "first_sample_error_label"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt-csv",
        default="outputs/semantic_entropy_umm/codex_visual_gt_first_pass/codex_visual_gt_first_pass.csv",
    )
    parser.add_argument(
        "--run-datapoints",
        action="append",
        required=True,
        help="Run datapoints in the form run_name=path/to/t2i_uncertainty_datapoints.csv",
    )
    parser.add_argument(
        "--out-dir",
        default=(
            "outputs/semantic_entropy_umm/codex_visual_gt_first_pass/"
            "true_visual_hallucination_uncertainty"
        ),
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None and rows:
        fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or [])
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result) or math.isinf(result):
        return default
    return result


def to_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def fmt(value: Any, digits: int = 4) -> str:
    if value in ("", None):
        return "NA"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def auroc(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j
    pos_rank_sum = sum(rank for rank, (_, label) in zip(ranks, pairs) if label == 1)
    return (pos_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def average_precision(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    if positives == 0:
        return None
    pairs = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    tp = 0
    precision_sum = 0.0
    for idx, (_, label) in enumerate(pairs, start=1):
        if label:
            tp += 1
            precision_sum += tp / idx
    return precision_sum / positives


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - idx) + ordered[hi] * (idx - lo)


def accuracy_at_quantile(accuracies: list[int], uncertainties: list[float], q: float) -> float:
    cutoff = quantile(uncertainties, q)
    selected = [acc for acc, unc in zip(accuracies, uncertainties) if unc <= cutoff]
    return sum(selected) / len(selected) if selected else 0.0


def area_under_thresholded_accuracy(accuracies: list[int], uncertainties: list[float]) -> float:
    qs = [0.1 + idx * (0.9 / 19) for idx in range(20)]
    dx = qs[1] - qs[0]
    return sum(accuracy_at_quantile(accuracies, uncertainties, q) * dx for q in qs)


def parse_run_datapoints(specs: list[str]) -> list[tuple[str, Path]]:
    parsed = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--run-datapoints must be run=path, got {spec!r}")
        run, path = spec.split("=", 1)
        parsed.append((run, Path(path)))
    return parsed


def build_concept_visual_summary(gt_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in gt_rows:
        grouped[(row["run"], row["concept_id"])].append(row)

    summaries = []
    for (run, concept_id), rows in sorted(grouped.items()):
        n = len(rows)
        strict_bad = [row for row in rows if row["strict_visual_match"] != "yes"]
        core_bad = [row for row in rows if row["core_scene_match"] != "yes"]
        background_bad = [row for row in rows if row["error_type"] == "background_mismatch"]
        type_counts = Counter(row["error_type"] for row in rows if row["error_type"] != "none")
        problem_samples = [
            row["sample_id"]
            for row in rows
            if row["strict_visual_match"] != "yes" or row["core_scene_match"] != "yes"
        ]
        summaries.append(
            {
                "run": run,
                "concept_id": concept_id,
                "num_images": n,
                "strict_bad_count": len(strict_bad),
                "core_bad_count": len(core_bad),
                "background_bad_count": len(background_bad),
                "strict_bad_rate": len(strict_bad) / n if n else 0.0,
                "core_bad_rate": len(core_bad) / n if n else 0.0,
                "strict_any_visual_hallucination": int(bool(strict_bad)),
                "core_any_visual_hallucination": int(bool(core_bad)),
                "background_any_visual_issue": int(bool(background_bad)),
                "strict_majority_visual_hallucination": int(len(strict_bad) >= max(1, math.ceil(n / 2))),
                "core_majority_visual_hallucination": int(len(core_bad) >= max(1, math.ceil(n / 2))),
                "problem_sample_ids": ";".join(problem_samples),
                "error_type_counts": ";".join(f"{key}:{value}" for key, value in sorted(type_counts.items())),
            }
        )
    return summaries


def build_joined_datapoints(
    run_datapoints: list[tuple[str, Path]],
    concept_summary: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    visual_by_key = {(row["run"], row["concept_id"]): row for row in concept_summary}
    joined = []
    for run, path in run_datapoints:
        for row in read_csv(path):
            key = (run, row["concept_id"])
            if key not in visual_by_key:
                continue
            visual = visual_by_key[key]
            joined_row = {"run": run, **row}
            for field, value in visual.items():
                if field in {"run", "concept_id"}:
                    continue
                joined_row[field] = value
            joined.append(joined_row)
    return joined


def metric_groups(rows: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    groups: list[tuple[str, list[dict[str, Any]]]] = [("combined/ALL", rows)]
    groups.append(("combined/joint", [row for row in rows if row["slot"] == "joint"]))
    for run in sorted({row["run"] for row in rows}):
        run_rows = [row for row in rows if row["run"] == run]
        groups.append((f"{run}/ALL", run_rows))
        groups.append((f"{run}/joint", [row for row in run_rows if row["slot"] == "joint"]))
    return [(name, group_rows) for name, group_rows in groups if group_rows]


def build_detector_metrics(joined: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for group_name, group_rows in metric_groups(joined):
        independent_concepts = len({(row["run"], row["concept_id"]) for row in group_rows})
        for score_name, score_field in SCORE_SPECS:
            scores = [to_float(row.get(score_field)) for row in group_rows]
            for label_name, label_field in VISUAL_LABEL_SPECS:
                labels = [to_int(row.get(label_field)) for row in group_rows]
                positives = sum(labels)
                accuracies = [1 - label for label in labels]
                auc = auroc(labels, scores)
                ap = average_precision(labels, scores)
                rows.append(
                    {
                        "group": group_name,
                        "uncertainty_measure": score_name,
                        "visual_hallucination_label": label_name,
                        "n_rows": len(group_rows),
                        "n_independent_concepts": independent_concepts,
                        "positives": positives,
                        "positive_rate": positives / len(labels) if labels else 0.0,
                        "mean_score": sum(scores) / len(scores) if scores else 0.0,
                        "mean_positive_score": (
                            sum(score for score, label in zip(scores, labels) if label) / positives
                            if positives
                            else ""
                        ),
                        "mean_negative_score": (
                            sum(score for score, label in zip(scores, labels) if not label)
                            / (len(labels) - positives)
                            if len(labels) > positives
                            else ""
                        ),
                        "AUROC": "" if auc is None else auc,
                        "AUPRC": "" if ap is None else ap,
                        "area_under_thresholded_accuracy": area_under_thresholded_accuracy(accuracies, scores),
                        "accuracy_at_0.8_answer_fraction": accuracy_at_quantile(accuracies, scores, 0.8),
                        "accuracy_at_0.9_answer_fraction": accuracy_at_quantile(accuracies, scores, 0.9),
                        "accuracy_at_0.95_answer_fraction": accuracy_at_quantile(accuracies, scores, 0.95),
                    }
                )
    return rows


def confusion_counts(labels: list[int], predictions: list[int]) -> dict[str, int]:
    tp = sum(1 for y, p in zip(labels, predictions) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(labels, predictions) if y == 0 and p == 1)
    tn = sum(1 for y, p in zip(labels, predictions) if y == 0 and p == 0)
    fn = sum(1 for y, p in zip(labels, predictions) if y == 1 and p == 0)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def safe_div(num: int | float, den: int | float) -> float | str:
    return num / den if den else ""


def build_auto_vs_visual_confusion(joined: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    groups = metric_groups(joined)
    for group_name, group_rows in groups:
        independent_concepts = len({(row["run"], row["concept_id"]) for row in group_rows})
        for visual_name, visual_field in VISUAL_LABEL_SPECS:
            labels = [to_int(row.get(visual_field)) for row in group_rows]
            for auto_name, auto_field in AUTO_LABEL_SPECS:
                predictions = [to_int(row.get(auto_field)) for row in group_rows]
                counts = confusion_counts(labels, predictions)
                tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]
                rows.append(
                    {
                        "group": group_name,
                        "visual_label": visual_name,
                        "auto_label": auto_name,
                        "n_rows": len(group_rows),
                        "n_independent_concepts": independent_concepts,
                        "visual_positives": sum(labels),
                        "auto_positives": sum(predictions),
                        "tp": tp,
                        "fp": fp,
                        "tn": tn,
                        "fn": fn,
                        "precision": safe_div(tp, tp + fp),
                        "recall": safe_div(tp, tp + fn),
                        "specificity": safe_div(tn, tn + fp),
                        "accuracy": safe_div(tp + tn, len(labels)),
                        "balanced_accuracy": (
                            (safe_div(tp, tp + fn) + safe_div(tn, tn + fp)) / 2
                            if (tp + fn) and (tn + fp)
                            else ""
                        ),
                    }
                )
    return rows


def markdown_table(rows: list[dict[str, Any]], fields: list[str], limit: int | None = None) -> list[str]:
    selected = rows[:limit] if limit else rows
    if not selected:
        return ["(none)"]
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in selected:
        values = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                value = fmt(value)
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def build_report(
    out_dir: Path,
    gt_csv: Path,
    run_datapoints: list[tuple[str, Path]],
    concept_summary: list[dict[str, Any]],
    joined: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    confusion: list[dict[str, Any]],
) -> str:
    concept_problem = [
        row
        for row in concept_summary
        if row["strict_any_visual_hallucination"] or row["core_any_visual_hallucination"]
    ]
    run_summary = []
    for run in sorted({row["run"] for row in concept_summary}):
        run_rows = [row for row in concept_summary if row["run"] == run]
        run_summary.append(
            {
                "run": run,
                "concepts": len(run_rows),
                "images": sum(int(row["num_images"]) for row in run_rows),
                "strict_any_concepts": sum(int(row["strict_any_visual_hallucination"]) for row in run_rows),
                "core_any_concepts": sum(int(row["core_any_visual_hallucination"]) for row in run_rows),
                "strict_majority_concepts": sum(int(row["strict_majority_visual_hallucination"]) for row in run_rows),
                "core_majority_concepts": sum(int(row["core_majority_visual_hallucination"]) for row in run_rows),
            }
        )
    run_rows = sorted(run_summary, key=lambda row: row["run"])
    combined_summary = {
        "run": "combined",
        "concepts": sum(row["concepts"] for row in run_rows),
        "images": sum(row["images"] for row in run_rows),
        "strict_any_concepts": sum(row["strict_any_concepts"] for row in run_rows),
        "core_any_concepts": sum(row["core_any_concepts"] for row in run_rows),
        "strict_majority_concepts": sum(row["strict_majority_concepts"] for row in run_rows),
        "core_majority_concepts": sum(row["core_majority_concepts"] for row in run_rows),
    }
    primary_metrics = [
        row
        for row in metrics
        if row["group"].endswith("/joint")
        and row["uncertainty_measure"] in {"semantic_entropy", "normalized_entropy"}
        and row["visual_hallucination_label"]
        in {"strict_any_visual_hallucination", "core_any_visual_hallucination", "background_any_visual_issue"}
    ]
    primary_metrics = sorted(
        primary_metrics,
        key=lambda row: (
            row["group"] != "combined/joint",
            row["group"],
            row["visual_hallucination_label"],
            row["uncertainty_measure"],
        ),
    )
    primary_confusion = [
        row
        for row in confusion
        if row["group"].endswith("/joint")
        and row["visual_label"] in {"strict_any_visual_hallucination", "core_any_visual_hallucination"}
        and row["auto_label"] in {"auto_any_error", "auto_mode_error", "auto_majority_error"}
    ]
    primary_confusion = sorted(
        primary_confusion,
        key=lambda row: (
            row["group"] != "combined/joint",
            row["group"],
            row["visual_label"],
            row["auto_label"],
        ),
    )
    missing_majority = [
        row
        for row in metrics
        if row["group"] == "combined/joint"
        and row["uncertainty_measure"] == "semantic_entropy"
        and row["visual_hallucination_label"].endswith("majority_visual_hallucination")
    ]
    lines = [
        "# T2I True Visual GT Uncertainty Analysis",
        "",
        "Status: ANALYSIS_COMPLETE",
        "",
        "This report joins the Codex first-pass visual GT with the existing T2I semantic-uncertainty datapoints. The primary unit is `slot=joint` at concept level; `ALL` slot-level files are retained only as sensitivity outputs because they duplicate the same visual label across slots.",
        "",
        "## Inputs",
        "",
        f"- Visual GT: `{gt_csv}`",
        *[f"- {run} uncertainty datapoints: `{path}`" for run, path in run_datapoints],
        "",
        "## Output Files",
        "",
        f"- `{out_dir / 'concept_visual_gt_summary.csv'}`",
        f"- `{out_dir / 'true_visual_uncertainty_datapoints.csv'}`",
        f"- `{out_dir / 'true_visual_uncertainty_detector_metrics.csv'}`",
        f"- `{out_dir / 'auto_vs_visual_confusion.csv'}`",
        f"- `{out_dir / 'true_visual_hallucination_uncertainty_report.md'}`",
        "",
        "## Visual GT Aggregation",
        "",
        *markdown_table(
            [combined_summary, *run_rows],
            [
                "run",
                "concepts",
                "images",
                "strict_any_concepts",
                "core_any_concepts",
                "strict_majority_concepts",
                "core_majority_concepts",
            ],
        ),
        "",
        "## Primary Detector Metrics",
        "",
        *markdown_table(
            [
                {
                    **row,
                    "positive_rate": fmt(row["positive_rate"]),
                    "mean_score": fmt(row["mean_score"]),
                    "mean_positive_score": fmt(row["mean_positive_score"]),
                    "mean_negative_score": fmt(row["mean_negative_score"]),
                    "AUROC": fmt(row["AUROC"]) if row["AUROC"] != "" else "",
                    "AUPRC": fmt(row["AUPRC"]) if row["AUPRC"] != "" else "",
                    "area_under_thresholded_accuracy": fmt(row["area_under_thresholded_accuracy"]),
                }
                for row in primary_metrics
            ],
            [
                "group",
                "uncertainty_measure",
                "visual_hallucination_label",
                "n_rows",
                "positives",
                "positive_rate",
                "mean_positive_score",
                "mean_negative_score",
                "AUROC",
                "AUPRC",
                "area_under_thresholded_accuracy",
            ],
        ),
        "",
        "## Auto Readback Labels vs Visual GT",
        "",
        *markdown_table(
            [
                {
                    **row,
                    "precision": fmt(row["precision"]),
                    "recall": fmt(row["recall"]),
                    "specificity": fmt(row["specificity"]),
                    "accuracy": fmt(row["accuracy"]),
                    "balanced_accuracy": fmt(row["balanced_accuracy"]),
                }
                for row in primary_confusion
            ],
            [
                "group",
                "visual_label",
                "auto_label",
                "visual_positives",
                "auto_positives",
                "tp",
                "fp",
                "tn",
                "fn",
                "precision",
                "recall",
                "specificity",
                "accuracy",
                "balanced_accuracy",
            ],
        ),
        "",
        "## Problem Concepts",
        "",
        *markdown_table(
            concept_problem,
            [
                "run",
                "concept_id",
                "num_images",
                "strict_bad_count",
                "core_bad_count",
                "background_bad_count",
                "problem_sample_ids",
                "error_type_counts",
            ],
        ),
        "",
        "## Interpretation",
        "",
        "- Under true visual GT, strict visual hallucination is sparse: 8/50 concepts have at least one strict mismatch, and 5/50 have a core-scene mismatch.",
        "- No concept has majority visual hallucination in this first-pass GT, so majority-label AUROC is undefined; the meaningful labels are `strict_any`, `core_any`, and `background_any`.",
        "- The automatic readback labels substantially over-call errors relative to visual GT. This means prior high automatic T2I error rates should be framed as readback/extractor disagreement unless backed by visual labels.",
        "- Semantic entropy is now evaluated against true visual hallucination rather than automatic slot error. The resulting AUROC/AUPRC values are exploratory because there are only 50 independent concept-level examples and few positives.",
        "",
        "## Claim Boundary",
        "",
        "Use this as Codex first-pass visual GT, not independent human annotation. The primary evidence supports: controlled T2I semantic entropy can be evaluated against true visual hallucination labels, but current Emu3.5 failures in this toy benchmark are rare and mostly isolated/background/support artifacts rather than systematic image hallucination.",
    ]
    if missing_majority:
        lines.extend(
            [
                "",
                "## Majority Label Check",
                "",
                "Combined `joint` majority labels have zero positives:",
                *markdown_table(
                    [
                        {
                            "label": row["visual_hallucination_label"],
                            "positives": row["positives"],
                            "AUROC": row["AUROC"],
                            "AUPRC": row["AUPRC"],
                        }
                        for row in missing_majority
                    ],
                    ["label", "positives", "AUROC", "AUPRC"],
                ),
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    gt_csv = Path(args.gt_csv)
    out_dir = Path(args.out_dir)
    run_datapoints = parse_run_datapoints(args.run_datapoints)

    gt_rows = read_csv(gt_csv)
    concept_summary = build_concept_visual_summary(gt_rows)
    joined = build_joined_datapoints(run_datapoints, concept_summary)
    metrics = build_detector_metrics(joined)
    confusion = build_auto_vs_visual_confusion(joined)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "concept_visual_gt_summary.csv", concept_summary)
    write_csv(out_dir / "true_visual_uncertainty_datapoints.csv", joined)
    write_csv(out_dir / "true_visual_uncertainty_detector_metrics.csv", metrics)
    write_csv(out_dir / "auto_vs_visual_confusion.csv", confusion)
    report = build_report(out_dir, gt_csv, run_datapoints, concept_summary, joined, metrics, confusion)
    (out_dir / "true_visual_hallucination_uncertainty_report.md").write_text(report, encoding="utf-8")
    print(
        f"[INFO] wrote {len(concept_summary)} concept summaries, {len(joined)} joined datapoints, "
        f"{len(metrics)} metrics, {len(confusion)} confusion rows to {out_dir}"
    )


if __name__ == "__main__":
    main()
