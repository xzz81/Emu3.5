#!/usr/bin/env python3
"""Evaluate T2I uncertainty against atom-level visual-check GT.

This is the complex-scene counterpart to the earlier visual-GT analysis. It
expects a per-atom annotation CSV where each generated image has several
visual checks and each check has a pass/fail label. The script aggregates
those checks to image- and concept-level hallucination labels, then joins them
to semantic-uncertainty datapoints.
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

LABEL_SPECS = [
    ("atom_any_image_fail", "atom_any_image_fail"),
    ("atom_majority_image_fail", "atom_majority_image_fail"),
    ("atom_any_check_fail", "atom_any_check_fail"),
    ("atom_check_fail_rate_positive", "atom_check_fail_rate_positive"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--atom-gt-csv",
        required=True,
        help="Per-atom visual-check GT CSV with pass_fail labels.",
    )
    parser.add_argument(
        "--uncertainty-datapoints",
        required=True,
        help="T2I uncertainty datapoints CSV, usually t2i_uncertainty_datapoints.csv.",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--out-dir", required=True)
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
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


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


def normalize_pass_fail(value: str) -> str:
    value = value.strip().lower()
    if value in {"pass", "yes", "1", "true"}:
        return "pass"
    if value in {"fail", "no", "0", "false"}:
        return "fail"
    return ""


def build_image_rows(atom_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in atom_rows:
        label = normalize_pass_fail(row.get("pass_fail", ""))
        if not label:
            continue
        grouped[(row["concept_id"], int(row["sample_id"]))].append(row)

    image_rows = []
    for (concept_id, sample_id), rows in sorted(grouped.items()):
        labels = [normalize_pass_fail(row.get("pass_fail", "")) for row in rows]
        fail_count = sum(label == "fail" for label in labels)
        failure_types = Counter(row.get("failure_type", "") or "unspecified" for row in rows if normalize_pass_fail(row.get("pass_fail", "")) == "fail")
        image_rows.append(
            {
                "concept_id": concept_id,
                "sample_id": sample_id,
                "annotated_checks": len(rows),
                "failed_checks": fail_count,
                "passed_checks": len(rows) - fail_count,
                "check_fail_rate": fail_count / len(rows) if rows else 0.0,
                "image_any_check_fail": int(fail_count > 0),
                "failed_check_ids": ";".join(row["check_id"] for row in rows if normalize_pass_fail(row.get("pass_fail", "")) == "fail"),
                "failure_type_counts": ";".join(f"{key}:{value}" for key, value in sorted(failure_types.items())),
                "image_path": rows[0].get("image_path", ""),
            }
        )
    return image_rows


def build_concept_rows(image_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in image_rows:
        grouped[row["concept_id"]].append(row)

    concept_rows = []
    for concept_id, rows in sorted(grouped.items()):
        annotated_images = len(rows)
        failed_images = sum(to_int(row["image_any_check_fail"]) for row in rows)
        total_checks = sum(to_int(row["annotated_checks"]) for row in rows)
        failed_checks = sum(to_int(row["failed_checks"]) for row in rows)
        concept_rows.append(
            {
                "concept_id": concept_id,
                "annotated_images": annotated_images,
                "failed_images": failed_images,
                "image_fail_rate": failed_images / annotated_images if annotated_images else 0.0,
                "annotated_checks": total_checks,
                "failed_checks": failed_checks,
                "check_fail_rate": failed_checks / total_checks if total_checks else 0.0,
                "atom_any_image_fail": int(failed_images > 0),
                "atom_majority_image_fail": int(failed_images >= max(1, math.ceil(annotated_images / 2))),
                "atom_any_check_fail": int(failed_checks > 0),
                "atom_check_fail_rate_positive": int((failed_checks / total_checks if total_checks else 0.0) > 0),
            }
        )
    return concept_rows


def build_joined_rows(datapoints: list[dict[str, str]], concept_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    label_by_concept = {row["concept_id"]: row for row in concept_rows}
    joined = []
    for row in datapoints:
        labels = label_by_concept.get(row["concept_id"])
        if labels is None:
            continue
        out = dict(row)
        for key, value in labels.items():
            if key != "concept_id":
                out[key] = value
        joined.append(out)
    return joined


def metric_groups(rows: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    groups = [("ALL", rows), ("joint", [row for row in rows if row.get("slot") == "joint"])]
    slots = sorted({row.get("slot", "") for row in rows if row.get("slot") not in {"", "joint"}})
    groups.extend((slot, [row for row in rows if row.get("slot") == slot]) for slot in slots)
    return [(name, group) for name, group in groups if group]


def build_metrics(joined: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for group_name, group_rows in metric_groups(joined):
        independent_concepts = len({row["concept_id"] for row in group_rows})
        for score_name, score_field in SCORE_SPECS:
            scores = [to_float(row.get(score_field)) for row in group_rows]
            for label_name, label_field in LABEL_SPECS:
                labels = [to_int(row.get(label_field)) for row in group_rows]
                positives = sum(labels)
                accuracies = [1 - label for label in labels]
                auc = auroc(labels, scores)
                ap = average_precision(labels, scores)
                rows.append(
                    {
                        "group": group_name,
                        "uncertainty_measure": score_name,
                        "atom_gt_label": label_name,
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
                            sum(score for score, label in zip(scores, labels) if not label) / (len(labels) - positives)
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


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> list[str]:
    if not rows:
        return ["(none)"]
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        vals = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                value = fmt(value)
            vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def build_report(
    atom_gt_csv: Path,
    uncertainty_csv: Path,
    image_rows: list[dict[str, Any]],
    concept_rows: list[dict[str, Any]],
    joined: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    run_id: str | None,
) -> str:
    primary = [
        {
            **row,
            "positive_rate": fmt(row["positive_rate"]),
            "mean_positive_score": fmt(row["mean_positive_score"]),
            "mean_negative_score": fmt(row["mean_negative_score"]),
            "AUROC": fmt(row["AUROC"]) if row["AUROC"] != "" else "",
            "AUPRC": fmt(row["AUPRC"]) if row["AUPRC"] != "" else "",
        }
        for row in metrics
        if row["group"] == "joint"
        and row["uncertainty_measure"] in {"semantic_entropy", "normalized_entropy"}
    ]
    concept_summary = {
        "annotated_concepts": len(concept_rows),
        "annotated_images": len(image_rows),
        "failed_images": sum(to_int(row["image_any_check_fail"]) for row in image_rows),
        "annotated_checks": sum(to_int(row["annotated_checks"]) for row in image_rows),
        "failed_checks": sum(to_int(row["failed_checks"]) for row in image_rows),
        "joined_datapoints": len(joined),
    }
    lines = [
        "# Complex T2I Atom-GT Uncertainty Analysis",
        "",
        "Status: ANALYSIS_COMPLETE" if joined else "Status: WAITING_FOR_UNCERTAINTY_DATAPOINTS",
        "",
        f"- Run id: `{run_id or ''}`",
        f"- Atom GT: `{atom_gt_csv}`",
        f"- Uncertainty datapoints: `{uncertainty_csv}`",
        "",
        "## Annotation Coverage",
        "",
        *markdown_table([concept_summary], list(concept_summary.keys())),
        "",
        "## Primary Joint Metrics",
        "",
        *markdown_table(
            primary,
            [
                "uncertainty_measure",
                "atom_gt_label",
                "n_rows",
                "positives",
                "positive_rate",
                "mean_positive_score",
                "mean_negative_score",
                "AUROC",
                "AUPRC",
            ],
        ),
        "",
        "## Concept Labels",
        "",
        *markdown_table(
            concept_rows,
            [
                "concept_id",
                "annotated_images",
                "failed_images",
                "image_fail_rate",
                "annotated_checks",
                "failed_checks",
                "check_fail_rate",
                "atom_any_image_fail",
                "atom_majority_image_fail",
            ],
        ),
        "",
        "## Notes",
        "",
        "- Metrics are meaningful only after uncertainty datapoints exist for the same concepts.",
        "- Partial GT is allowed for monitoring, but paper claims should use a complete or clearly bounded annotation set.",
        "- The primary detector unit is `slot=joint`; other slots duplicate concept-level visual labels and are sensitivity views.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    atom_gt_csv = Path(args.atom_gt_csv)
    uncertainty_csv = Path(args.uncertainty_datapoints)
    out_dir = Path(args.out_dir)
    atom_rows = read_csv(atom_gt_csv)
    image_rows = build_image_rows(atom_rows)
    concept_rows = build_concept_rows(image_rows)
    datapoints = read_csv(uncertainty_csv) if uncertainty_csv.exists() else []
    joined = build_joined_rows(datapoints, concept_rows) if datapoints else []
    metrics = build_metrics(joined) if joined else []

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "complex_t2i_atom_image_labels.csv", image_rows)
    write_csv(out_dir / "complex_t2i_atom_concept_labels.csv", concept_rows)
    write_csv(out_dir / "complex_t2i_atom_uncertainty_datapoints.csv", joined)
    write_csv(out_dir / "complex_t2i_atom_uncertainty_metrics.csv", metrics)
    report = build_report(atom_gt_csv, uncertainty_csv, image_rows, concept_rows, joined, metrics, args.run_id)
    (out_dir / "complex_t2i_atom_uncertainty_report.md").write_text(report, encoding="utf-8")
    print(
        f"[INFO] image_rows={len(image_rows)} concept_rows={len(concept_rows)} "
        f"joined={len(joined)} metrics={len(metrics)} out={out_dir}"
    )


if __name__ == "__main__":
    main()
