#!/usr/bin/env python3
"""Build a report for Codex visual GT first-pass labels."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


BASE = Path("outputs/semantic_entropy_umm/codex_visual_gt_first_pass")


def main() -> None:
    rows = list(csv.DictReader((BASE / "codex_visual_gt_first_pass.csv").open(encoding="utf-8")))
    summary = list(csv.DictReader((BASE / "codex_visual_gt_first_pass_summary.csv").open(encoding="utf-8")))
    problem = [row for row in rows if row["strict_visual_match"] == "no"]

    lines: list[str] = []
    lines.append("# Codex Visual GT First-Pass Annotation")
    lines.append("")
    lines.append("Status: CODEX_FIRST_PASS_COMPLETE")
    lines.append("")
    lines.append(
        "This is a Codex visual first-pass annotation, not an independent human ground-truth file. "
        "It is intended to unblock a first manual-style check of true image hallucination labels."
    )
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(
        "- Formal run: `outputs/semantic_entropy_umm/semantic_uncertainty_formal_emu35_20260606_095911`, "
        "30 concepts x 10 samples = 300 images."
    )
    lines.append(
        "- Bench20 run: `outputs/semantic_entropy_umm/t2i_bench20_emu35_20260608`, "
        "20 concepts x 5 samples = 100 images."
    )
    lines.append("- Total annotated images: `400`.")
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    lines.append("- `codex_visual_gt_first_pass.csv`: per-image labels.")
    lines.append("- `codex_visual_gt_first_pass_summary.csv`: run-level summary.")
    lines.append("- `codex_visual_gt_sheet_index.csv`: contact sheet index.")
    lines.append("- `contact_sheets/`: concept-level visual sheets used for annotation.")
    lines.append("")
    lines.append("## Label Semantics")
    lines.append("")
    lines.append(
        "- `strict_visual_match=yes`: target objects, colors, relation, and clean white-background/no-extra-object "
        "requirement are visually satisfied."
    )
    lines.append(
        "- `core_scene_match=yes`: target objects, colors, and relation are satisfied even if background or support "
        "artifacts appear."
    )
    lines.append(
        "- `core_scene_match=partial`: target is mostly present but there is color contamination, extra object, "
        "or support object that affects visual faithfulness."
    )
    lines.append(
        "- `true_image_hallucination=yes_background_only`: core scene is correct but the white-background "
        "requirement is violated."
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| run | samples | strict yes | strict no | core yes | core partial |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for item in summary:
        lines.append(
            "| {run} | {samples} | {strict_match_yes} | {strict_match_no} | {core_yes} | {core_partial} |".format(
                **item
            )
        )
    lines.append("")
    lines.append("Overall error-type counts:")
    lines.append("")
    for key, value in Counter(row["error_type"] for row in rows).items():
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    lines.append("## Problem Cases")
    lines.append("")
    lines.append("| run | concept | sample | error_type | core_match | note |")
    lines.append("| --- | --- | ---: | --- | --- | --- |")
    for row in problem:
        note = row["notes"].replace("|", "/")
        lines.append(
            "| {run} | {concept_id} | {sample_id} | {error_type} | {core_scene_match} | {note} |".format(
                note=note, **row
            )
        )
    lines.append("")
    lines.append("## Immediate Interpretation")
    lines.append("")
    lines.append(
        "The first-pass visual labels support the earlier manual audit: most automatic hallucination/readback "
        "errors are not true image hallucinations. The generated images usually contain the requested objects, "
        "colors, and spatial relations. Remaining strict mismatches are rare and mostly background/support "
        "artifacts or isolated color contamination/extra-object cases."
    )
    lines.append("")
    lines.append(
        "For paper claims, this file should be described as `Codex visual first-pass labels`, not as independent "
        "human annotation."
    )
    (BASE / "codex_visual_gt_first_pass_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(BASE / "codex_visual_gt_first_pass_report.md")


if __name__ == "__main__":
    main()
