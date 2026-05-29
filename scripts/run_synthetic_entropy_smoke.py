#!/usr/bin/env python3
# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Create a deterministic synthetic UME trace run without loading model weights."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Dict, List, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.entropy_trace import (  # noqa: E402
    build_trace_records,
    calibration_bin_rows,
    component_correlation_rows,
    hallucination_diagnostic_rows,
    render_summary_report,
    summarize_records,
    thinking_transition_rows,
    write_csv,
    write_html_report,
    write_jsonl,
    write_visual_artifacts,
)


SPECIAL_IDS = {
    "BOS": 0,
    "EOS": 1,
    "BOG": 2,
    "EOG": 3,
    "BOC": 4,
    "EOC": 5,
    "BOI": 6,
    "IMG": 7,
    "EOI": 8,
    "EOL": 9,
    "PAD": 10,
    "BOV": 30,
}

VOCAB_SIZE = 42
VISUAL_START = 30


def score_vec(groups: Dict[str, Sequence[int]], sharp_token: int | None = None) -> torch.Tensor:
    scores = torch.full((VOCAB_SIZE,), -torch.inf)
    for name, token_ids in groups.items():
        base = {
            "structure": 0.9,
            "text": 0.4,
            "thinking": 0.5,
            "visual": 0.2,
        }.get(name, 0.0)
        for offset, token_id in enumerate(token_ids):
            scores[token_id] = base - 0.05 * offset
    if sharp_token is not None:
        scores = torch.full((VOCAB_SIZE,), -torch.inf)
        scores[sharp_token] = 4.0
        for token_id in groups.get("visual", []) + groups.get("text", []):
            if token_id != sharp_token:
                scores[token_id] = -3.0
    return scores


def cfg_trace(length: int, high_steps: Sequence[int] = ()) -> List[Dict[str, float]]:
    high = set(high_steps)
    return [
        {
            "u_cfg": 0.68 if step in high else 0.0,
            "cfg_js": 0.68 if step in high else 0.0,
            "cfg_available": step in high,
        }
        for step in range(length)
    ]


def synthetic_samples():
    text_group = {"text": [11, 12, 13, 14, 15, 16], "structure": [1, 6]}
    thinking_group = {"thinking": [11, 12, 13], "structure": [3, 5]}
    header_group = {"text": [17, 18, 19], "structure": [7, 8]}
    visual_group = {"visual": [30, 31, 32, 33, 34, 35]}
    boundary_group = {"structure": [1, 3, 5, 6, 7, 8, 9], "text": [11, 12], "visual": [30, 31]}

    return [
        {
            "sample_id": "synthetic_t2i_attribute_000",
            "task": "t2i",
            "tokens": [11, 12, 6, 17, 18, 7, 30, 31, 9, 32, 33, 8, 1],
            "scores": [
                score_vec(text_group),
                score_vec(text_group),
                score_vec(boundary_group),
                score_vec(header_group),
                score_vec(header_group),
                score_vec(boundary_group),
                score_vec(visual_group),
                score_vec(visual_group),
                score_vec(boundary_group),
                score_vec(visual_group, sharp_token=32),
                score_vec(visual_group, sharp_token=33),
                score_vec(boundary_group),
                score_vec({"structure": [1]}),
            ],
            "cfg_high_steps": [6, 7],
            "error_steps": {6: "object_hallucination", 7: "attribute_hallucination"},
        },
        {
            "sample_id": "synthetic_story_thinking_001",
            "task": "story",
            "tokens": [2, 11, 12, 3, 4, 13, 14, 5, 6, 17, 18, 7, 30, 31, 32, 33, 8, 1],
            "scores": [
                score_vec(boundary_group),
                score_vec(thinking_group),
                score_vec(thinking_group),
                score_vec(boundary_group),
                score_vec(boundary_group),
                score_vec(thinking_group),
                score_vec(thinking_group),
                score_vec(boundary_group),
                score_vec(boundary_group),
                score_vec(header_group),
                score_vec(header_group),
                score_vec(boundary_group),
                score_vec(visual_group, sharp_token=30),
                score_vec(visual_group, sharp_token=31),
                score_vec(visual_group, sharp_token=32),
                score_vec(visual_group, sharp_token=33),
                score_vec(boundary_group),
                score_vec({"structure": [1]}),
            ],
            "cfg_high_steps": [12],
            "error_steps": {},
        },
        {
            "sample_id": "synthetic_ambiguous_falseconf_002",
            "task": "ambiguous",
            "tokens": [11, 6, 17, 18, 7, 34, 35, 8, 1],
            "scores": [
                score_vec(text_group, sharp_token=11),
                score_vec(boundary_group),
                score_vec(header_group),
                score_vec(header_group),
                score_vec(boundary_group),
                score_vec(visual_group, sharp_token=34),
                score_vec(visual_group, sharp_token=35),
                score_vec(boundary_group),
                score_vec({"structure": [1]}),
            ],
            "cfg_high_steps": [5, 6],
            "error_steps": {5: "false_confidence_prior", 6: "false_confidence_prior"},
        },
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--out-root", default="outputs/ume_trace_runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = args.run_id or f"synthetic_{time.strftime('%Y%m%d_%H%M%S')}"
    out_root = Path(args.out_root) / run_id
    raw_dir = out_root / "raw_generations"
    trace_dir = out_root / "entropy_traces"
    case_dir = out_root / "case_studies"
    tables_dir = out_root / "tables"
    for path in (raw_dir, trace_dir, case_dir, tables_dir, out_root / "figures"):
        path.mkdir(parents=True, exist_ok=True)

    all_records = []
    for sample in synthetic_samples():
        records = build_trace_records(
            sample["scores"],
            sample["tokens"],
            sample_id=sample["sample_id"],
            special_token_ids=SPECIAL_IDS,
            cfg_trace=cfg_trace(len(sample["tokens"]), sample["cfg_high_steps"]),
            visual_token_start=VISUAL_START,
            metadata={"task": sample["task"]},
        )
        for record in records:
            error_type = sample["error_steps"].get(record["step"])
            record["is_error"] = error_type is not None
            record["error_type"] = error_type or ""
        write_jsonl(records, trace_dir / f"{sample['sample_id']}_entropy.jsonl")
        (raw_dir / f"{sample['sample_id']}.txt").write_text(
            " ".join(str(token) for token in sample["tokens"]) + "\n",
            encoding="utf-8",
        )
        (case_dir / f"{sample['sample_id']}_summary.md").write_text(
            render_summary_report(records, f"Synthetic sample {sample['sample_id']}"),
            encoding="utf-8",
        )
        all_records.extend(records)

    write_csv(summarize_records(all_records, "token_type"), tables_dir / "ume_by_token_type.csv")
    write_csv(summarize_records(all_records, "segment"), tables_dir / "ume_by_segment.csv")
    write_csv(summarize_records(all_records, "task"), tables_dir / "ume_by_task.csv")
    write_csv(component_correlation_rows(all_records), tables_dir / "ume_component_correlations.csv")
    write_csv(thinking_transition_rows(all_records), tables_dir / "thinking_transition.csv")
    write_csv(hallucination_diagnostic_rows(all_records), tables_dir / "hallucination_diagnostics.csv")
    write_csv(calibration_bin_rows(all_records), tables_dir / "calibration_bins.csv")
    figures = write_visual_artifacts(all_records, out_root)
    report = render_summary_report(all_records, f"Synthetic UME Smoke Run {run_id}")
    if figures:
        report += "\n\n## Generated figures\n\n" + "\n".join(
            f"- `{Path(path).relative_to(out_root)}`" for path in figures
        )
    (out_root / "entropy_distribution_report.md").write_text(report + "\n", encoding="utf-8")
    write_html_report(all_records, out_root, f"Synthetic UME Smoke Run {run_id}", figures)
    print(f"[INFO] Synthetic UME run saved to {out_root}")


if __name__ == "__main__":
    main()
