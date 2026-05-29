#!/usr/bin/env python3
# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Summarize UME JSONL traces into CSV tables and a markdown report."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.entropy_trace import (  # noqa: E402
    calibration_bin_rows,
    component_correlation_rows,
    hallucination_diagnostic_rows,
    read_jsonl,
    render_summary_report,
    summarize_records,
    thinking_transition_rows,
    write_csv,
    write_html_report,
    write_visual_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", required=True, help="Directory containing *_entropy.jsonl files.")
    parser.add_argument("--out-dir", default=None, help="Directory for summary tables and report.")
    parser.add_argument("--max-traces", default=10, type=int, help="Maximum per-sample trace figures to emit.")
    parser.add_argument("--skip-figures", action="store_true", help="Only write CSV and markdown outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace_dir = Path(args.trace_dir)
    out_dir = Path(args.out_dir) if args.out_dir else trace_dir.parent
    tables_dir = out_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for path in sorted(trace_dir.glob("*_entropy.jsonl")):
        records.extend(read_jsonl(path))

    write_csv(summarize_records(records, "token_type"), tables_dir / "ume_by_token_type.csv")
    write_csv(summarize_records(records, "segment"), tables_dir / "ume_by_segment.csv")
    write_csv(summarize_records(records, "task"), tables_dir / "ume_by_task.csv")
    write_csv(component_correlation_rows(records), tables_dir / "ume_component_correlations.csv")
    write_csv(thinking_transition_rows(records), tables_dir / "thinking_transition.csv")
    write_csv(hallucination_diagnostic_rows(records), tables_dir / "hallucination_diagnostics.csv")
    write_csv(calibration_bin_rows(records), tables_dir / "calibration_bins.csv")
    report = render_summary_report(records, f"UME Trace Summary: {trace_dir}")
    figure_paths = []
    if not args.skip_figures:
        figure_paths = write_visual_artifacts(records, out_dir, max_traces=args.max_traces)
        if figure_paths:
            report += "\n\n## Generated figures\n\n" + "\n".join(
                f"- `{Path(path).relative_to(out_dir)}`" for path in figure_paths
            )
        write_html_report(records, out_dir, f"UME Trace Summary: {trace_dir}", figure_paths)
    (out_dir / "entropy_distribution_report.md").write_text(report + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} token records to {out_dir}")


if __name__ == "__main__":
    main()
