#!/usr/bin/env python3
# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Build a paper-style phase-1 findings report from real UME result tables."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="../research_logs")
    parser.add_argument("--out", default="../research_logs/phase1_findings_report.md")
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def by_key(rows: Iterable[Mapping[str, str]], *keys: str) -> Dict[tuple[str, ...], Mapping[str, str]]:
    return {tuple(str(row[key]) for key in keys): row for row in rows}


def find_row(rows: Sequence[Mapping[str, str]], **criteria: str) -> Mapping[str, str]:
    for row in rows:
        if all(str(row.get(key, "")) == value for key, value in criteria.items()):
            return row
    raise KeyError(f"missing row for {criteria}")


def as_float(value: Any) -> float | None:
    if value == "" or value is None:
        return None
    return float(value)


def fmt(value: Any, digits: int = 4) -> str:
    if value == "" or value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if isinstance(value, int):
        return str(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.{digits}f}"


def md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]], aligns: Sequence[str] | None = None) -> List[str]:
    if aligns is None:
        aligns = ["---"] * len(headers)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(aligns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def select_uncertainty(rows: Sequence[Mapping[str, str]], keys: Sequence[tuple[str, str, str]]) -> List[Mapping[str, str]]:
    lookup = by_key(rows, "level", "group", "metric")
    return [lookup[key] for key in keys if key in lookup]


def select_robustness(rows: Sequence[Mapping[str, str]], keys: Sequence[tuple[str, str, str, str]]) -> List[Mapping[str, str]]:
    lookup = by_key(rows, "level", "group", "metric", "omit_type")
    return [lookup[key] for key in keys if key in lookup]


def figure_rows(log_dir: Path) -> List[tuple[str, str]]:
    specs = [
        ("phase1_figures/region_auc_summary.svg", "Region-token AUC summary with bootstrap CIs"),
        ("phase1_figures/sample_auc_summary.svg", "Sample-level AUC summary with bootstrap CIs"),
        ("phase1_figures/task_error_ume_bars.svg", "Mean UME in manually labeled error vs correct regions"),
        ("phase1_figures/calibration_bins.svg", "Default UME calibration bins"),
    ]
    return [(rel_path, caption) for rel_path, caption in specs if (log_dir / rel_path).exists()]


def optional_csv(path: Path) -> List[Dict[str, str]]:
    return read_csv(path) if path.exists() else []


def build_report(log_dir: Path) -> str:
    task_coverage = read_csv(log_dir / "corpus_audit" / "task_coverage.csv")
    sample_summary = read_csv(log_dir / "manual_sample_judgment_summary.csv")
    hallucination = read_csv(log_dir / "region_labeled_analysis" / "tables" / "hallucination_diagnostics.csv")
    uncertainty = read_csv(log_dir / "auc_uncertainty" / "auc_uncertainty.csv")
    robustness = read_csv(log_dir / "auc_robustness" / "auc_robustness_summary.csv")
    prompt_cluster_auc = optional_csv(log_dir / "prompt_cluster_effects" / "prompt_cluster_auc_summary.csv")
    prompt_cluster_leave = optional_csv(log_dir / "prompt_cluster_effects" / "prompt_cluster_leave_one_summary.csv")
    token_thresholds = optional_csv(log_dir / "token_extremes" / "token_extreme_thresholds.csv")
    token_sample_summary = optional_csv(log_dir / "token_extremes" / "sample_extreme_summary.csv")
    token_case_sheets = optional_csv(log_dir / "token_case_sheets" / "token_case_sheet_summary.csv")
    error_type_strata = optional_csv(log_dir / "error_type_strata" / "error_type_strata_summary.csv")
    boundary_events = optional_csv(log_dir / "boundary_entropy" / "boundary_event_summary.csv")
    boundary_auc = optional_csv(log_dir / "boundary_entropy" / "boundary_sample_auc.csv")
    reproducibility_checks = optional_csv(log_dir / "reproducibility" / "validation_checks.csv")
    coverage_progress = optional_csv(log_dir / "phase1_coverage" / "phase1_coverage_progress.csv")

    t2i_cov = find_row(task_coverage, task="t2i")
    x2i_cov = find_row(task_coverage, task="x2i")
    all_sample = find_row(sample_summary, group="all")
    t2i_sample = find_row(sample_summary, group="t2i")
    x2i_sample = find_row(sample_summary, group="x2i")
    t2i_region = find_row(hallucination, task="t2i")
    x2i_region = find_row(hallucination, task="x2i")

    uncertainty_rows = select_uncertainty(
        uncertainty,
        [
            ("region_token", "all", "ume_default"),
            ("region_token", "all", "inverse_ume_default"),
            ("region_token", "all", "u_cfg"),
            ("region_token", "all", "inverse_u_tok"),
            ("region_token", "t2i", "inverse_best_t2i_mix"),
            ("region_token", "x2i", "u_cfg"),
            ("sample", "all", "inverse_p90_best_all_mix"),
            ("sample", "t2i", "inverse_p90_u_cfg"),
            ("sample", "x2i", "inverse_mean_ume_default"),
        ],
    )
    robustness_rows = select_robustness(
        robustness,
        [
            ("region_token", "all", "inverse_u_tok", "run_id"),
            ("region_token", "all", "inverse_ume_default", "run_id"),
            ("region_token", "all", "u_cfg", "run_id"),
            ("region_token", "t2i", "inverse_u_tok", "run_id"),
            ("region_token", "x2i", "u_cfg", "run_id"),
            ("sample", "all", "inverse_p90_best_all_mix", "run_id"),
            ("sample", "t2i", "inverse_p90_u_cfg", "run_id"),
            ("sample", "x2i", "inverse_p90_ume_default", "run_id"),
        ],
    )

    lines: List[str] = [
        "# Phase-1 Findings Report: Unified Multimodal Entropy",
        "",
        "This report is generated from local real Emu3.5 trace outputs and manual labels. Synthetic fixtures are used only for code regression tests and are not counted as research evidence.",
        "",
        "## Executive Summary",
        "",
        "- UME tracing is implemented for Emu3.5 image generation and records token-level `u_tok`, `u_intra`, `u_mod`, `u_cfg`, and combined `ume` during real GPU inference.",
        f"- Current real corpus: `{int(t2i_cov['samples'])}` T2I samples and `{int(x2i_cov['samples'])}` X2I samples; `{int(t2i_cov['binary_labels']) + int(x2i_cov['binary_labels'])}` binary manual labels; `{int(t2i_cov['error_labels']) + int(x2i_cov['error_labels'])}` error labels.",
        "- Main finding: default UME is not a positive hallucination detector in this corpus. Region-level errors are often lower-entropy commitments.",
        "- Strongest stable region-level signal overall is inverse token entropy/default UME; strongest X2I-specific region signal is high `u_cfg`.",
        "- Sample-level X2I metrics remain small-N sensitive and should not be used as headline evidence.",
        "",
        "## Corpus",
        "",
    ]
    lines.extend(
        md_table(
            ["task", "runs", "samples", "decoded", "complete images", "binary labels", "errors", "visual tokens"],
            [
                [
                    "t2i",
                    t2i_cov["runs"],
                    t2i_cov["samples"],
                    t2i_cov["decoded"],
                    t2i_cov["complete_images"],
                    t2i_cov["binary_labels"],
                    t2i_cov["error_labels"],
                    t2i_cov["visual_tokens"],
                ],
                [
                    "x2i",
                    x2i_cov["runs"],
                    x2i_cov["samples"],
                    x2i_cov["decoded"],
                    x2i_cov["complete_images"],
                    x2i_cov["binary_labels"],
                    x2i_cov["error_labels"],
                    x2i_cov["visual_tokens"],
                ],
            ],
            ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"],
        )
    )
    lines.extend(
        [
            "",
            "Coverage gap: interleaved/story/howto is not covered by real local results because the main `BAAI/Emu3.5` interleaved model is not cached locally.",
            "",
        ]
    )

    if coverage_progress:
        lines.extend(["## Coverage Against Plan", ""])
        lines.extend(
            md_table(
                ["plan item", "target", "samples", "binary labels", "errors", "completion", "remaining", "blocker"],
                [
                    [
                        row["plan_item"],
                        row["target_samples"],
                        row["samples"],
                        row["binary_labels"],
                        row["errors"],
                        fmt(row["completion_fraction"]),
                        row["remaining_to_full_target"],
                        row["blocker"],
                    ]
                    for row in coverage_progress
                ],
                ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---"],
            )
        )
        lines.extend([""])

    lines.extend(
        [
            "## Sample-Level Labels",
            "",
        ]
    )
    lines.extend(
        md_table(
            [
                "group",
                "samples",
                "errors",
                "error rate",
                "mean quality",
                "AUC visual mean UME",
                "AUC visual p90 UME",
                "AUC visual mean u_cfg",
            ],
            [
                [
                    "all",
                    all_sample["samples"],
                    all_sample["errors"],
                    fmt(all_sample["error_rate"]),
                    fmt(all_sample["mean_quality"]),
                    fmt(all_sample["auc_visual_mean_ume_error"]),
                    fmt(all_sample["auc_visual_p90_ume_error"]),
                    fmt(all_sample["auc_visual_mean_u_cfg_error"]),
                ],
                [
                    "t2i",
                    t2i_sample["samples"],
                    t2i_sample["errors"],
                    fmt(t2i_sample["error_rate"]),
                    fmt(t2i_sample["mean_quality"]),
                    fmt(t2i_sample["auc_visual_mean_ume_error"]),
                    fmt(t2i_sample["auc_visual_p90_ume_error"]),
                    fmt(t2i_sample["auc_visual_mean_u_cfg_error"]),
                ],
                [
                    "x2i",
                    x2i_sample["samples"],
                    x2i_sample["errors"],
                    fmt(x2i_sample["error_rate"]),
                    fmt(x2i_sample["mean_quality"]),
                    fmt(x2i_sample["auc_visual_mean_ume_error"]),
                    fmt(x2i_sample["auc_visual_p90_ume_error"]),
                    fmt(x2i_sample["auc_visual_mean_u_cfg_error"]),
                ],
            ],
            ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"],
        )
    )

    lines.extend(["", "## Region-Level Diagnostics", ""])
    lines.extend(
        md_table(
            [
                "task",
                "labeled tokens",
                "error tokens",
                "error rate",
                "mean UME error",
                "mean UME correct",
                "AUC UME->error",
                "false-conf tokens",
                "false-conf rate",
            ],
            [
                [
                    "t2i",
                    t2i_region["labeled_tokens"],
                    t2i_region["error_tokens"],
                    fmt(t2i_region["error_rate"]),
                    fmt(t2i_region["mean_ume_error"]),
                    fmt(t2i_region["mean_ume_correct"]),
                    fmt(t2i_region["auc_ume_error"]),
                    t2i_region["false_conf_tokens"],
                    fmt(t2i_region["false_conf_rate"]),
                ],
                [
                    "x2i",
                    x2i_region["labeled_tokens"],
                    x2i_region["error_tokens"],
                    fmt(x2i_region["error_rate"]),
                    fmt(x2i_region["mean_ume_error"]),
                    fmt(x2i_region["mean_ume_correct"]),
                    fmt(x2i_region["auc_ume_error"]),
                    x2i_region["false_conf_tokens"],
                    fmt(x2i_region["false_conf_rate"]),
                ],
            ],
            ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"],
        )
    )

    lines.extend(["", "## AUC With Uncertainty", ""])
    lines.extend(
        md_table(
            ["level", "group", "metric", "rows", "errors", "AUC", "95% bootstrap CI", "permutation p"],
            [
                [
                    row["level"],
                    row["group"],
                    row["metric"],
                    row["rows"],
                    row["errors"],
                    fmt(row["auc"]),
                    f"[{fmt(row['bootstrap_ci_low'])}, {fmt(row['bootstrap_ci_high'])}]",
                    fmt(row["permutation_p_auc_ge_observed"]),
                ]
                for row in uncertainty_rows
            ],
            ["---", "---", "---", "---:", "---:", "---:", "---", "---:"],
        )
    )

    lines.extend(["", "## Leave-One Robustness", ""])
    lines.extend(
        md_table(
            ["level", "group", "metric", "baseline AUC", "leave-one min", "leave-one max", "max abs delta", "worst omitted"],
            [
                [
                    row["level"],
                    row["group"],
                    row["metric"],
                    fmt(row["baseline_auc"]),
                    fmt(row["min_auc"]),
                    fmt(row["max_auc"]),
                    fmt(row["max_abs_delta"]),
                    row["worst_omitted_id"],
                ]
                for row in robustness_rows
            ],
            ["---", "---", "---", "---:", "---:", "---:", "---:", "---"],
        )
    )

    if prompt_cluster_auc:
        prompt_lookup = by_key(prompt_cluster_auc, "level", "aggregation", "group", "metric")
        prompt_rows = [
            prompt_lookup[key]
            for key in [
                ("sample", "sample_or_token", "all", "visual_mean_ume"),
                ("sample", "prompt_group", "all", "visual_mean_ume"),
                ("sample", "prompt_group", "t2i", "visual_mean_ume"),
                ("sample", "prompt_group", "x2i", "visual_mean_u_cfg"),
                ("region_token", "sample_or_token", "x2i", "u_cfg"),
                ("region_token", "prompt_group", "x2i", "u_cfg"),
            ]
            if key in prompt_lookup
        ]
        lines.extend(["", "## Prompt-Cluster Sensitivity", ""])
        lines.append(
            "Repeated `_rN` samples are collapsed by prompt group here, so repeated prompts do not receive extra independent-sample weight."
        )
        lines.extend([""])
        lines.extend(
            md_table(
                ["level", "aggregation", "group", "metric", "rows", "prompt groups", "errors", "AUC"],
                [
                    [
                        row["level"],
                        row["aggregation"],
                        row["group"],
                        row["metric"],
                        row["rows"],
                        row["prompt_groups"],
                        row["errors"],
                        fmt(row["auc"]),
                    ]
                    for row in prompt_rows
                ],
                ["---", "---", "---", "---", "---:", "---:", "---:", "---:"],
            )
        )
    if prompt_cluster_leave:
        prompt_leave_lookup = by_key(prompt_cluster_leave, "level", "group", "metric")
        leave_rows = [
            prompt_leave_lookup[key]
            for key in [
                ("sample", "all", "visual_mean_ume"),
                ("sample", "x2i", "visual_mean_u_cfg"),
                ("region_token", "all", "inverse_u_tok"),
                ("region_token", "x2i", "u_cfg"),
            ]
            if key in prompt_leave_lookup
        ]
        lines.extend(["", "Prompt-group leave-one sensitivity:", ""])
        lines.extend(
            md_table(
                ["level", "group", "metric", "baseline AUC", "prompt groups", "max abs delta", "worst omitted"],
                [
                    [
                        row["level"],
                        row["group"],
                        row["metric"],
                        fmt(row["baseline_auc"]),
                        row["prompt_groups"],
                        fmt(row["max_abs_delta"]),
                        row["worst_omitted_prompt_group"],
                    ]
                    for row in leave_rows
                ],
                ["---", "---", "---", "---:", "---:", "---:", "---"],
            )
        )

    if token_thresholds:
        thresholds = token_thresholds[0]
        false_conf_samples = sorted(
            token_sample_summary,
            key=lambda row: (float(row.get("low_ume_error_tokens", 0) or 0), float(row.get("false_conf_rate", 0) or 0)),
            reverse=True,
        )
        high_cfg_samples = sorted(
            token_sample_summary,
            key=lambda row: (float(row.get("high_cfg_error_tokens", 0) or 0), float(row.get("high_cfg_error_rate", 0) or 0)),
            reverse=True,
        )
        lines.extend(["", "## Token Extreme Case Catalog", ""])
        lines.append(
            "Token-level catalogs expose concrete regions behind the aggregate AUCs: high default UME is often not an error, while many manually marked errors are low-entropy commitments."
        )
        lines.extend([""])
        lines.extend(
            md_table(
                [
                    "visual tokens",
                    "error tokens",
                    "UME q25",
                    "UME median",
                    "UME q75",
                    "u_cfg q75",
                    "low-UME error tokens",
                    "high-CFG error tokens",
                ],
                [
                    [
                        thresholds["visual_tokens"],
                        thresholds["error_tokens"],
                        fmt(thresholds["ume_q25"]),
                        fmt(thresholds["ume_median"]),
                        fmt(thresholds["ume_q75"]),
                        fmt(thresholds["u_cfg_q75"]),
                        thresholds["low_ume_error_tokens"],
                        thresholds["high_cfg_error_tokens"],
                    ]
                ],
                ["---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"],
            )
        )
        if token_case_sheets:
            lines.append(
                f"\nRendered case-sheet overlays: `{len(token_case_sheets)}` real decoded images in `research_logs/token_case_sheets/`."
            )
        if token_sample_summary:
            lines.extend(["", "Largest low-UME error samples:", ""])
            lines.extend(
                md_table(
                    ["task", "sample", "error tokens", "low-UME errors", "false-conf rate", "mean error UME"],
                    [
                        [
                            row["task"],
                            row["sample_id"],
                            row["error_tokens"],
                            row["low_ume_error_tokens"],
                            fmt(row["false_conf_rate"]),
                            fmt(row["error_mean_ume"]),
                        ]
                        for row in false_conf_samples[:5]
                    ],
                    ["---", "---", "---:", "---:", "---:", "---:"],
                )
            )
            lines.extend(["", "Largest high-CFG error samples:", ""])
            lines.extend(
                md_table(
                    ["task", "sample", "error tokens", "high-CFG errors", "high-CFG error rate", "mean error u_cfg"],
                    [
                        [
                            row["task"],
                            row["sample_id"],
                            row["error_tokens"],
                            row["high_cfg_error_tokens"],
                            fmt(row["high_cfg_error_rate"]),
                            fmt(row["error_mean_u_cfg"]),
                        ]
                        for row in high_cfg_samples[:5]
                    ],
                    ["---", "---", "---:", "---:", "---:", "---:"],
                )
            )

    if boundary_events:
        event_lookup = by_key(boundary_events, "task", "event")
        selected_events = [
            event_lookup[key]
            for key in [
                ("t2i", "image_token"),
                ("t2i", "row_break"),
                ("t2i", "image_end"),
                ("x2i", "image_token"),
                ("x2i", "row_break"),
                ("x2i", "image_end"),
            ]
            if key in event_lookup
        ]
        selected_auc = [
            row
            for row in boundary_auc
            if (row.get("group"), row.get("metric"))
            in {
                ("all", "first_visual_mean_ume"),
                ("all", "row_head_mean_ume"),
                ("x2i", "row_head_mean_ume"),
                ("x2i", "row_tail_mean_ume"),
                ("t2i", "first_visual_mean_ume"),
                ("t2i", "last_visual_mean_u_cfg"),
            }
        ]
        lines.extend(["", "## Boundary Entropy", ""])
        lines.append(
            "Structure-token `u_mod` is degenerate under the current constrained image decoder, so boundary diagnostics are reported on adjacent visual-token windows."
        )
        lines.extend(["", "Boundary event windows:", ""])
        lines.extend(
            md_table(
                ["task", "event", "events", "mean u_mod", "prev mean UME", "next mean UME", "prev err rate", "next err rate"],
                [
                    [
                        row["task"],
                        row["event"],
                        row["events"],
                        fmt(row["mean_u_mod"]),
                        fmt(row["prev_mean_ume"]),
                        fmt(row["next_mean_ume"]),
                        fmt(row["prev_error_rate"]),
                        fmt(row["next_error_rate"]),
                    ]
                    for row in selected_events
                ],
                ["---", "---", "---:", "---:", "---:", "---:", "---:", "---:"],
            )
        )
        if selected_auc:
            lines.extend(["", "Sample-level boundary-window AUC:", ""])
            lines.extend(
                md_table(
                    ["group", "metric", "samples", "errors", "mean error", "mean correct", "AUC", "inverse AUC"],
                    [
                        [
                            row["group"],
                            row["metric"],
                            row["samples"],
                            row["errors"],
                            fmt(row["mean_error"]),
                            fmt(row["mean_correct"]),
                            fmt(row["auc_metric_error"]),
                            fmt(row["auc_inverse_metric_error"]),
                        ]
                        for row in selected_auc
                    ],
                    ["---", "---", "---:", "---:", "---:", "---:", "---:", "---:"],
                )
            )

    if error_type_strata:
        all_error_types = [row for row in error_type_strata if row.get("label_key") == "error_type" and row.get("group") == "all"]
        largest_error_types = sorted(all_error_types, key=lambda row: int(row.get("positive_tokens", 0) or 0), reverse=True)[:8]
        strongest_inverse = sorted(all_error_types, key=lambda row: float(row.get("auc_inverse_ume_default", 0) or 0), reverse=True)[:5]
        strongest_cfg = sorted(all_error_types, key=lambda row: float(row.get("auc_u_cfg", 0) or 0), reverse=True)[:5]
        lines.extend(["", "## Error-Type Stratification", ""])
        lines.append(
            "Manual error labels are stratified here by type, with each type compared against real non-error visual tokens. This separates false-confidence failures from CFG-disagreement failures."
        )
        lines.extend(["", "Largest error-type strata:", ""])
        lines.extend(
            md_table(
                ["error type", "tokens", "samples", "mean UME", "low-UME rate", "mean u_cfg", "high-CFG rate", "best metric", "best AUC"],
                [
                    [
                        row["label"],
                        row["positive_tokens"],
                        row["positive_samples"],
                        fmt(row["mean_ume"]),
                        fmt(row["low_ume_rate"]),
                        fmt(row["mean_u_cfg"]),
                        fmt(row["high_cfg_rate"]),
                        row["best_metric"],
                        fmt(row["best_auc"]),
                    ]
                    for row in largest_error_types
                ],
                ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---", "---:"],
            )
        )
        lines.extend(["", "Strongest inverse-UME strata:", ""])
        lines.extend(
            md_table(
                ["error type", "tokens", "AUC inverse UME", "low-UME rate", "AUC u_cfg"],
                [
                    [
                        row["label"],
                        row["positive_tokens"],
                        fmt(row["auc_inverse_ume_default"]),
                        fmt(row["low_ume_rate"]),
                        fmt(row["auc_u_cfg"]),
                    ]
                    for row in strongest_inverse
                ],
                ["---", "---:", "---:", "---:", "---:"],
            )
        )
        lines.extend(["", "Strongest u_cfg strata:", ""])
        lines.extend(
            md_table(
                ["error type", "tokens", "AUC u_cfg", "high-CFG rate", "AUC inverse UME"],
                [
                    [
                        row["label"],
                        row["positive_tokens"],
                        fmt(row["auc_u_cfg"]),
                        fmt(row["high_cfg_rate"]),
                        fmt(row["auc_inverse_ume_default"]),
                    ]
                    for row in strongest_cfg
                ],
                ["---", "---:", "---:", "---:", "---:"],
            )
        )

    figures = figure_rows(log_dir)
    if figures:
        lines.extend(["", "## Figures", ""])
        for rel_path, caption in figures:
            lines.extend([f"![{caption}]({rel_path})", "", f"*{caption}.*", ""])

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "1. Default UME is useful as a process trace, but not as a one-dimensional positive hallucination score. In current region labels, high default UME has AUC below random overall.",
            "2. T2I contradiction and instruction-resolution failures are dominated by false-confidence: incorrect visual regions often have lower `u_tok`/`u_intra`/default UME than correct regions.",
            "3. X2I reference-edit failures differ: `u_cfg` is elevated in manually marked X2I error regions and remains stable under leave-one-run and leave-one-sample checks.",
            "4. Sample-level metrics are weaker than region-token metrics, especially for X2I. The report should use sample-level X2I numbers only as exploratory context.",
            "",
        ]
    )

    if reproducibility_checks:
        passed_checks = [row for row in reproducibility_checks if str(row.get("passed", "")).lower() in {"true", "1", "yes"}]
        lines.extend(
            [
                "## Reproducibility",
                "",
                f"- Local reproducibility manifest: `research_logs/reproducibility/reproducibility_manifest.md`.",
                f"- Validation checks passed: `{len(passed_checks)}/{len(reproducibility_checks)}`.",
                "- Artifact SHA-256 hashes are recorded in `research_logs/reproducibility/artifact_hashes.csv`.",
                "",
            ]
        )

    lines.extend(
        [
            "## Limitations",
            "",
            "- No real interleaved/story/howto experiment is available yet because the required main model is missing from local cache.",
            "- Manual region labels are bounding-box approximations on visual-token grids, not pixel-perfect segmentation.",
            f"- X2I has only `{x2i_cov['binary_labels']}` binary sample labels and `{x2i_cov['error_labels']}` error samples, so sample-level X2I statistics remain underpowered.",
            "- Synthetic unit tests validate code paths only; they are excluded from all research counts above.",
            "",
            "## Source Artifacts",
            "",
            "- `research_logs/real_ume_run_index.md`",
            "- `research_logs/manual_sample_judgment_analysis.md`",
            "- `research_logs/region_labeled_analysis/entropy_distribution_report.md`",
            "- `research_logs/ume_weight_sweep_step010/ume_weight_sweep_report.md`",
            "- `research_logs/auc_uncertainty/auc_uncertainty_report.md`",
            "- `research_logs/auc_robustness/auc_robustness_report.md`",
            "- `research_logs/prompt_cluster_effects/prompt_cluster_effects_report.md`",
            "- `research_logs/token_extremes/token_extremes_report.md`",
            "- `research_logs/token_case_sheets/token_case_sheets_report.md`",
            "- `research_logs/error_type_strata/error_type_strata_report.md`",
            "- `research_logs/boundary_entropy/boundary_entropy_report.md`",
            "- `research_logs/corpus_audit/corpus_audit_report.md`",
            "- `research_logs/phase1_coverage/phase1_coverage_plan.md`",
            "- `research_logs/reproducibility/reproducibility_manifest.md`",
            "- `research_logs/phase1_figures/`",
            "- `research_logs/ume_phase1_log.md`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    log_dir = Path(args.log_dir)
    out_path = Path(args.out)
    report = build_report(log_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"[INFO] wrote phase-1 findings report to {out_path}")


if __name__ == "__main__":
    main()
