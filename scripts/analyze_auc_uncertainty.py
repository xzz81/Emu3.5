#!/usr/bin/env python3
# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Bootstrap and permutation uncertainty for real UME/error AUC metrics."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Mapping, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_ume_weight_sweep import as_bool, binary_auc, mean, quantile, read_csv, read_jsonl


ScoreFn = Callable[[Mapping[str, Any]], float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--sample-index", default="data/real_ume/real_ume_sample_index.csv")
    parser.add_argument("--labels", default="data/real_ume/manual_sample_judgments.jsonl")
    parser.add_argument("--out-dir", default="outputs/research_logs/auc_uncertainty")
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--permutation-iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260524)
    return parser.parse_args()


def region_metric_functions() -> Dict[str, ScoreFn]:
    return {
        "ume_default": lambda row: float(row.get("ume", 0.0) or 0.0),
        "inverse_ume_default": lambda row: -float(row.get("ume", 0.0) or 0.0),
        "u_cfg": lambda row: float(row.get("u_cfg", 0.0) or 0.0),
        "inverse_u_cfg": lambda row: -float(row.get("u_cfg", 0.0) or 0.0),
        "inverse_best_all_mix": lambda row: -(
            0.5 * float(row.get("u_intra", 0.0) or 0.0)
            + 0.2 * float(row.get("u_cfg", 0.0) or 0.0)
            + 0.3 * float(row.get("u_mod", 0.0) or 0.0)
        ),
        "inverse_best_t2i_mix": lambda row: -(
            0.4 * float(row.get("u_intra", 0.0) or 0.0)
            + 0.5 * float(row.get("u_cfg", 0.0) or 0.0)
            + 0.1 * float(row.get("u_mod", 0.0) or 0.0)
        ),
        "inverse_u_tok": lambda row: -float(row.get("u_tok", 0.0) or 0.0),
    }


def collect_region_rows(trace_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(trace_dir.glob("*_entropy.jsonl")):
        cluster_id = path.name.replace("_entropy.jsonl", "")
        for row in read_jsonl(path):
            if row.get("token_type") != "visual":
                continue
            if "is_error" not in row:
                continue
            rows.append({**row, "cluster_id": f"{row.get('task', 'unknown')}::{cluster_id}"})
    return rows


def label_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (str(row["task"]), str(row["run_id"]), str(row["sample_id"]))


def sample_metric_values(visual: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    def values(component: str) -> List[float]:
        return [float(row.get(component, 0.0) or 0.0) for row in visual]

    ume = values("ume")
    u_cfg = values("u_cfg")
    u_intra = values("u_intra")
    all_mix = [0.1 * intra + 0.9 * cfg for intra, cfg in zip(u_intra, u_cfg)]
    return {
        "mean_ume_default": float(mean(ume)),
        "inverse_mean_ume_default": -float(mean(ume)),
        "p90_ume_default": float(quantile(ume, 0.9)),
        "inverse_p90_ume_default": -float(quantile(ume, 0.9)),
        "mean_u_cfg": float(mean(u_cfg)),
        "inverse_mean_u_cfg": -float(mean(u_cfg)),
        "p90_u_cfg": float(quantile(u_cfg, 0.9)),
        "inverse_p90_u_cfg": -float(quantile(u_cfg, 0.9)),
        "max_u_cfg": max(u_cfg) if u_cfg else 0.0,
        "inverse_max_u_cfg": -(max(u_cfg) if u_cfg else 0.0),
        "inverse_p90_best_all_mix": -float(quantile(all_mix, 0.9)),
    }


def collect_sample_rows(sample_index: Path, labels_path: Path) -> List[Dict[str, Any]]:
    labels = [
        row
        for row in read_jsonl(labels_path)
        if as_bool(row.get("include_in_analysis", False)) is True and row.get("is_error") != ""
    ]
    sample_by_key = {label_key(row): row for row in read_csv(sample_index)}
    rows: List[Dict[str, Any]] = []
    for label in labels:
        sample = sample_by_key.get(label_key(label))
        if sample is None:
            continue
        trace_path = Path(sample["trace_path"])
        if not trace_path.exists():
            continue
        visual = [row for row in read_jsonl(trace_path) if row.get("token_type") == "visual"]
        if not visual:
            continue
        rows.append(
            {
                "task": sample["task"],
                "run_id": sample["run_id"],
                "sample_id": sample["sample_id"],
                "cluster_id": "::".join([sample["task"], sample["run_id"], sample["sample_id"]]),
                "is_error": as_bool(label["is_error"]),
                **sample_metric_values(visual),
            }
        )
    return rows


def percentile(values: Sequence[float], fraction: float) -> float | str:
    if not values:
        return ""
    ordered = sorted(values)
    idx = round((len(ordered) - 1) * fraction)
    return ordered[max(0, min(len(ordered) - 1, idx))]


def compute_auc(scores: Sequence[float], labels: Sequence[int]) -> float | str:
    return binary_auc(scores, labels)


def score_groups(scores: Sequence[float]) -> List[List[int]]:
    ranked = sorted(range(len(scores)), key=lambda idx: scores[idx])
    groups: List[List[int]] = []
    idx = 0
    while idx < len(ranked):
        j = idx + 1
        while j < len(ranked) and scores[ranked[j]] == scores[ranked[idx]]:
            j += 1
        groups.append(ranked[idx:j])
        idx = j
    return groups


def auc_from_score_groups(
    groups: Sequence[Sequence[int]],
    labels: Sequence[int],
    weights: Sequence[float] | None = None,
) -> float | str:
    if weights is None:
        weights = [1.0] * len(labels)
    positives = sum(weight for weight, label in zip(weights, labels) if label == 1)
    negatives = sum(weight for weight, label in zip(weights, labels) if label == 0)
    if positives <= 0 or negatives <= 0:
        return ""

    contribution = 0.0
    negatives_below = 0.0
    for group in groups:
        pos_weight = 0.0
        neg_weight = 0.0
        for idx in group:
            if labels[idx] == 1:
                pos_weight += weights[idx]
            else:
                neg_weight += weights[idx]
        contribution += pos_weight * negatives_below + 0.5 * pos_weight * neg_weight
        negatives_below += neg_weight
    return contribution / (positives * negatives)


def sorted_score_state(scores: Sequence[float]) -> Dict[str, np.ndarray]:
    score_array = np.asarray(scores, dtype=np.float64)
    order = np.argsort(score_array, kind="mergesort")
    sorted_scores = score_array[order]
    if len(sorted_scores) == 0:
        starts = np.asarray([], dtype=np.int64)
    else:
        starts = np.concatenate(([0], np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1)).astype(np.int64)
    return {"order": order.astype(np.int64), "starts": starts}


def auc_from_sorted_state(
    state: Mapping[str, np.ndarray],
    labels: Sequence[int] | np.ndarray,
    weights: Sequence[float] | np.ndarray | None = None,
) -> float | str:
    label_array = np.asarray(labels, dtype=np.float64)
    if weights is None:
        weight_array = np.ones(label_array.shape[0], dtype=np.float64)
    else:
        weight_array = np.asarray(weights, dtype=np.float64)
    pos_total = float(np.sum(weight_array * label_array))
    neg_total = float(np.sum(weight_array * (1.0 - label_array)))
    if pos_total <= 0.0 or neg_total <= 0.0:
        return ""

    order = state["order"]
    starts = state["starts"]
    sorted_weights = weight_array[order]
    sorted_labels = label_array[order]
    pos_by_group = np.add.reduceat(sorted_weights * sorted_labels, starts)
    neg_by_group = np.add.reduceat(sorted_weights * (1.0 - sorted_labels), starts)
    neg_below = np.cumsum(neg_by_group) - neg_by_group
    contribution = float(np.sum(pos_by_group * neg_below + 0.5 * pos_by_group * neg_by_group))
    return contribution / (pos_total * neg_total)


def group_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, List[Mapping[str, Any]]]:
    groups: Dict[str, List[Mapping[str, Any]]] = {"all": list(rows)}
    for row in rows:
        groups.setdefault(str(row.get("task", "unknown")), []).append(row)
    return groups


def cluster_bootstrap_auc(
    rows: Sequence[Mapping[str, Any]],
    score_fn: ScoreFn,
    rng: random.Random,
    iterations: int,
) -> List[float]:
    clusters: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        cluster_id = str(row["cluster_id"])
        clusters.setdefault(cluster_id, []).append(row)
    cluster_ids = sorted(clusters)
    sorted_cluster_to_idx = {cluster_id: idx for idx, cluster_id in enumerate(cluster_ids)}
    cluster_index_array = np.asarray([sorted_cluster_to_idx[str(row["cluster_id"])] for row in rows], dtype=np.int64)
    scores = [score_fn(row) for row in rows]
    labels = np.asarray([1 if as_bool(row.get("is_error", False)) else 0 for row in rows], dtype=np.float64)
    state = sorted_score_state(scores)
    aucs: List[float] = []
    for _ in range(iterations):
        sampled = [rng.randrange(len(cluster_ids)) for _ in cluster_ids]
        cluster_counts = np.bincount(sampled, minlength=len(cluster_ids)).astype(np.float64)
        weights = cluster_counts[cluster_index_array]
        auc = auc_from_sorted_state(state, labels, weights)
        if auc != "":
            aucs.append(float(auc))
    return aucs


def permutation_p_value(
    rows: Sequence[Mapping[str, Any]],
    score_fn: ScoreFn,
    rng: random.Random,
    iterations: int,
    observed_auc: float,
) -> float | str:
    scores = [score_fn(row) for row in rows]
    labels = [1 if as_bool(row.get("is_error", False)) else 0 for row in rows]
    state = sorted_score_state(scores)
    if auc_from_sorted_state(state, labels) == "":
        return ""
    count = 0
    permuted = labels[:]
    for _ in range(iterations):
        rng.shuffle(permuted)
        auc = auc_from_sorted_state(state, permuted)
        if auc != "" and float(auc) >= observed_auc:
            count += 1
    return (count + 1) / (iterations + 1)


def summarize(
    rows: Sequence[Mapping[str, Any]],
    metric_fns: Mapping[str, ScoreFn],
    *,
    level: str,
    rng: random.Random,
    bootstrap_iters: int,
    permutation_iters: int,
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for group, items in sorted(group_rows(rows).items()):
        labels = [1 if as_bool(row.get("is_error", False)) else 0 for row in items]
        for metric, score_fn in metric_fns.items():
            scores = [score_fn(row) for row in items]
            observed = compute_auc(scores, labels)
            boot = cluster_bootstrap_auc(items, score_fn, rng, bootstrap_iters)
            p_value = "" if observed == "" else permutation_p_value(items, score_fn, rng, permutation_iters, float(observed))
            output.append(
                {
                    "level": level,
                    "group": group,
                    "metric": metric,
                    "rows": len(items),
                    "clusters": len({row["cluster_id"] for row in items}),
                    "errors": sum(labels),
                    "error_rate": sum(labels) / len(labels) if labels else "",
                    "mean_score_error": mean([score for score, label in zip(scores, labels) if label == 1]),
                    "mean_score_correct": mean([score for score, label in zip(scores, labels) if label == 0]),
                    "auc": observed,
                    "bootstrap_mean_auc": mean(boot),
                    "bootstrap_ci_low": percentile(boot, 0.025),
                    "bootstrap_ci_high": percentile(boot, 0.975),
                    "permutation_p_auc_ge_observed": p_value,
                    "bootstrap_iters": bootstrap_iters,
                    "permutation_iters": permutation_iters,
                }
            )
    return output


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt(value: Any) -> str:
    if value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# AUC Uncertainty Analysis",
        "",
        "Bootstrap and permutation results computed only from real traced outputs and manual labels.",
        "Region-token bootstrap is clustered by sample trace; permutation p-values shuffle labels within the evaluated group.",
        "",
        "| level | group | metric | rows | errors | auc | 95% bootstrap CI | permutation p | mean_error | mean_correct |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["level"]),
                    str(row["group"]),
                    str(row["metric"]),
                    str(row["rows"]),
                    str(row["errors"]),
                    fmt(row["auc"]),
                    f"[{fmt(row['bootstrap_ci_low'])}, {fmt(row['bootstrap_ci_high'])}]",
                    fmt(row["permutation_p_auc_ge_observed"]),
                    fmt(row["mean_score_error"]),
                    fmt(row["mean_score_correct"]),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    region_rows = collect_region_rows(Path(args.trace_dir))
    sample_rows = collect_sample_rows(Path(args.sample_index), Path(args.labels))
    sample_metric_fns = {
        key: (lambda row, metric=key: float(row[metric]))
        for key in [
            "mean_ume_default",
            "inverse_mean_ume_default",
            "p90_ume_default",
            "inverse_p90_ume_default",
            "mean_u_cfg",
            "inverse_mean_u_cfg",
            "p90_u_cfg",
            "inverse_p90_u_cfg",
            "max_u_cfg",
            "inverse_max_u_cfg",
            "inverse_p90_best_all_mix",
        ]
    }

    rows = []
    rows.extend(
        summarize(
            region_rows,
            region_metric_functions(),
            level="region_token",
            rng=rng,
            bootstrap_iters=args.bootstrap_iters,
            permutation_iters=args.permutation_iters,
        )
    )
    rows.extend(
        summarize(
            sample_rows,
            sample_metric_fns,
            level="sample",
            rng=rng,
            bootstrap_iters=args.bootstrap_iters,
            permutation_iters=args.permutation_iters,
        )
    )
    write_csv(out_dir / "auc_uncertainty.csv", rows)
    write_markdown(out_dir / "auc_uncertainty_report.md", rows)
    print(
        f"[INFO] wrote {len(rows)} uncertainty rows using {len(region_rows)} region-token rows "
        f"and {len(sample_rows)} sample rows to {out_dir}"
    )


if __name__ == "__main__":
    main()
