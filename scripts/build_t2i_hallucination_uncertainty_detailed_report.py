#!/usr/bin/env python3
"""Build the final detailed report for T2I uncertainty/hallucination analysis."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="outputs/semantic_entropy_umm/t2i_hallucination_uncertainty")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fmt(value: Any) -> str:
    if value in (None, ""):
        return "NA"
    return f"{float(value):.4f}"


def pick(metrics: list[dict[str, str]], group: str, label: str, measure: str = "semantic_entropy") -> dict[str, str]:
    for row in metrics:
        if row["group"] == group and row["hallucination_label"] == label and row["uncertainty_measure"] == measure:
            return row
    raise KeyError((group, label, measure))


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    metrics = read_csv(out / "t2i_uncertainty_detector_metrics.csv")
    manual = read_csv(out / "manual_visual_audit_cases.csv")
    concept_manual_path = out / "manual_visual_audit_concepts.csv"
    concept_manual = read_csv(concept_manual_path) if concept_manual_path.exists() else []

    primary = [
        pick(metrics, "ALL", "any_error"),
        pick(metrics, "ALL", "majority_error"),
        pick(metrics, "ALL", "mode_error"),
        pick(metrics, "joint", "mode_error"),
        pick(metrics, "object_1", "mode_error"),
        pick(metrics, "relation", "any_error"),
    ]
    manual_counts = Counter(row["manual_visual_label"] for row in manual)
    auto_hall = [row for row in manual if row["case_bucket"] != "high_entropy_non_hallucination"]
    auto_non = [row for row in manual if row["case_bucket"] == "high_entropy_non_hallucination"]
    auto_hall_support = sum(row["manual_supports_automatic_label"] == "yes_for_hallucination" for row in auto_hall)
    auto_non_support = sum(row["manual_supports_automatic_label"] == "yes_for_non_hallucination" for row in auto_non)
    concept_correct = sum(int(row.get("correct_images", 0)) for row in concept_manual)
    concept_incorrect = sum(int(row.get("incorrect_images", 0)) for row in concept_manual)

    lines = [
        "# T2I Semantic Entropy and Hallucination Relationship Report",
        "",
        "Status: COMPLETE_WITH_MANUAL_VISUAL_AUDIT",
        "",
        "## Executive Conclusion",
        "",
        "Following the jlko/semantic_uncertainty pattern, T2I semantic entropy can be evaluated as an uncertainty score for automatic semantic readback errors. On the 30-concept formal Emu3.5 T2I run, automatic QA/readback labels give strong detector metrics: ALL-mode-error AUROC=0.9520 and joint-mode-error AUROC=0.9080.",
        "",
        "However, manual visual inspection of the 36 selected cases shows that the automatic hallucination labels mostly do not correspond to true image hallucination. The generated images usually match the target scene, while the visual QA/readback stage swaps object_1 with the left object or otherwise misbinds slots. Therefore the current automatic detector is best interpreted as detecting `semantic readback/extractor hallucination`, not direct T2I image hallucination.",
        "",
        "## What Was Matched From jlko/semantic_uncertainty",
        "",
        "| jlko/semantic_uncertainty | T2I adaptation in this run |",
        "| --- | --- |",
        "| Multiple sampled answers for one QA item | 10 generated T2I images/readbacks for one concept-slot item |",
        "| Semantic equivalence classes via entailment | Finite semantic clusters over object/color/relation/background slots |",
        "| `semantic_entropy` / `cluster_assignment_entropy` | `cluster_assignment_entropy` over T2I semantic clusters |",
        "| `validation_is_false` false-answer label | any/majority/mode/first-sample semantic error labels |",
        "| AUROC and selective accuracy | AUROC, area_under_thresholded_accuracy, accuracy_at_0.8/0.9/0.95/1.0 answer fraction |",
        "",
        "## Input and Output Artifacts",
        "",
        "- Remote run: `outputs/semantic_entropy_umm/semantic_uncertainty_formal_emu35_20260606_095911`",
        "- Analysis dir: `outputs/semantic_entropy_umm/t2i_hallucination_uncertainty`",
        "- Datapoints: `t2i_uncertainty_datapoints.csv`",
        "- Detector metrics: `t2i_uncertainty_detector_metrics.csv`",
        "- Visual-audit manifest: `t2i_hallucination_visual_audit_manifest.csv`",
        "- Manual visual labels: `manual_visual_audit_cases.csv`",
        "- Contact sheet: `t2i_hallucination_contact_sheet.png`",
        "- 10-sample concept sheets: `concept_sample_sheets/*.png`",
        "- Concept-level manual visual audit: `manual_visual_audit_concepts.csv`",
        "",
        "## Automatic Detector Metrics",
        "",
        "| group | label | n | positives | positive_rate | AUROC | area_under_thresholded_accuracy | acc@0.8 | acc@0.9 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in primary:
        lines.append(
            f"| {row['group']} | {row['hallucination_label']} | {row['n']} | {row['positives']} | "
            f"{fmt(row['positive_rate'])} | {fmt(row['AUROC'])} | {fmt(row['area_under_thresholded_accuracy'])} | "
            f"{fmt(row['accuracy_at_0.8_answer_fraction'])} | {fmt(row['accuracy_at_0.9_answer_fraction'])} |"
        )
    lines += [
        "",
        "Interpretation: if hallucination is defined by the automatic QA/readback label, semantic entropy is a strong detector. `any_error` is nearly trivial because any nonzero entropy often means at least one of 10 samples varied into an error. The stricter `mode_error` and `majority_error` labels are closer to jlko's false-answer setup and remain high but not perfect.",
        "",
        "## Manual Visual Audit",
        "",
        f"- Audited cases: {len(manual)}",
        f"- Automatic hallucination candidates audited: {len(auto_hall)}",
        f"- Manual support for automatic hallucination candidates: {auto_hall_support}/{len(auto_hall)}",
        f"- Automatic non-hallucination candidates audited: {len(auto_non)}",
        f"- Manual support for non-hallucination candidates: {auto_non_support}/{len(auto_non)}",
        f"- Manual label counts: `{dict(manual_counts)}`",
        f"- Concept-level sheet audit: {concept_correct}/{concept_correct + concept_incorrect if concept_manual else 0} images visually match the target core scene",
        "",
        "The visual audit used both the selected contact sheet and 10-sample concept sheets. The dominant failure mode is not bad image generation: images generally show the requested left/right object layout and colors. The dominant failure mode is semantic readback or slot alignment, especially treating the left object as object_1 even when the prompt defines object_1 as the first named object, which can be on the right.",
        "",
        "### Concept-Level 10-Sample Audit",
        "",
        "| concept | target scene | correct | incorrect | supports true image hallucination | notes |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for row in concept_manual:
        lines.append(
            f"| {row['concept_id']} | {row['target_scene']} | {row['correct_images']} | "
            f"{row['incorrect_images']} | {row['supports_true_t2i_hallucination']} | {row['notes']} |"
        )
    lines += [
        "",
        "### Representative Manual Findings",
        "",
        "| bucket | concept | slot | automatic target -> mode | manual conclusion |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in manual[:18]:
        lines.append(
            f"| {row['case_bucket']} | {row['concept_id']} | {row['slot']} | "
            f"{row['target_value']} -> {row['mode_value']} | {row['manual_visual_label']}: {row['manual_note']} |"
        )
    lines += [
        "",
        "## Main Scientific Finding",
        "",
        "The current T2I semantic entropy is strongly related to hallucination only under the automatic semantic-readback definition. Under manual visual inspection, many high-error / high-entropy and low-entropy hallucination candidates are actually correct images. Thus the relationship discovered so far is:",
        "",
        "> semantic entropy predicts instability or failure in the T2I -> visual-QA/readback semantic measurement pipeline, but true image hallucination requires a manually validated or stronger vision-grounded label.",
        "",
        "This is still useful: it identifies exactly where the unified entropy protocol is vulnerable. It also prevents overclaiming that T2I image hallucination has been solved or directly detected by entropy alone.",
        "",
        "## Recommended Next Experiments",
        "",
        "1. Replace object_1/object_2 naming with position-grounded labels: `left object` and `right object`, then rerun the T2I QA/readback entropy detector. This tests whether current errors are mostly slot-index artifacts.",
        "2. Build a small manual visual ground-truth table for all 300 T2I images over object/color/relation/background, then rerun jlko-style AUROC against true image hallucination labels.",
        "3. Add a second independent extractor or direct image parser. A detector that only works for one VQA readback model is not yet a detector of image hallucination.",
        "4. Split hallucination targets: object identity, color binding, spatial relation, count/object omission, and background/style. Each may have a different entropy relationship.",
        "",
        "## Broader Unified Entropy Research Design",
        "",
        "### I2T/T2I unified entropy and hallucination",
        "",
        "- Use one shared semantic state variable S across routes: objects, colors, relation, background, and optional texture/material/count.",
        "- For each route, estimate H(S | concept, route) from repeated samples.",
        "- Evaluate entropy as a detector of route-specific hallucination labels with the same jlko metrics: AUROC, selective accuracy, and area under thresholded accuracy.",
        "- Compare whether low-entropy high-hallucination appears more in T2I than I2T. This distinguishes stable wrong generation from uncertain recognition.",
        "",
        "### X2X unified entropy",
        "",
        "- Generalize S to an interleaved scene graph for text+image input and text+image output.",
        "- Score consistency across generated caption, generated image, and any follow-up readback: text says red cube, image shows blue sphere, readback says green cone are three route-specific projections of the same latent S.",
        "- Define cross-route hallucination as inconsistency among projections of S, not merely disagreement with one modality.",
        "- Run interventions on high-unified-entropy or low-entropy-high-hallucination cases: attention heads, residual path patches, prompt route changes, and object-binding controls.",
        "",
        "## Claim Boundary",
        "",
        "Allowed: this report adapts semantic_uncertainty-style detector evaluation to T2I semantic readback errors and shows a major gap between automatic readback hallucination and manually inspected image hallucination.",
        "",
        "Not allowed: raw text/image entropy comparability, a unified internal entropy space, or a claim that semantic entropy currently detects true T2I image hallucination without manual or stronger extractor labels.",
    ]
    target = out / "t2i_hallucination_uncertainty_detailed_report.md"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
