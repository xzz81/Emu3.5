#!/usr/bin/env python3
"""Build evidence that the current T2I scene benchmark is too simple.

The report compares the existing Codex visual GT / detector outputs against a
new complex-scene concept manifest. It does not run model inference.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


SLOTS = ["object_1", "color_1", "object_2", "color_2", "relation", "background"]
BASE = Path("outputs/semantic_entropy_umm")
GT_DIR = BASE / "codex_visual_gt_first_pass"
TRUE_VIS_DIR = GT_DIR / "true_visual_hallucination_uncertainty"
OUT_DIR = BASE / "complex_scene_upgrade"
REPORT_PATH = Path("docs/t2i_complex_scene_upgrade_report_2026-06-16.md")
COMPLEX_CONCEPTS = Path("configs/semantic_entropy_t2i_complex_scene_concepts.jsonl")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 4) -> str:
    if value in ("", None):
        return "NA"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def token_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9_]+", text))


def current_concept_rows() -> list[dict[str, Any]]:
    gt_rows = read_csv(GT_DIR / "codex_visual_gt_first_pass.csv")
    by_key: dict[tuple[str, str], dict[str, str]] = {}
    for row in gt_rows:
        key = (row["run"], row["concept_id"])
        by_key.setdefault(key, row)
    visual_summary = {
        (row["run"], row["concept_id"]): row
        for row in read_csv(TRUE_VIS_DIR / "concept_visual_gt_summary.csv")
    }
    rows = []
    for (run, concept_id), row in sorted(by_key.items()):
        summary = visual_summary[(run, concept_id)]
        target = json.loads(row["target_semantics"])
        rows.append(
            {
                "suite": "current",
                "run": run,
                "concept_id": concept_id,
                "prompt_tokens": token_count(row["prompt"]),
                "target_slots": sum(1 for slot in SLOTS if slot in target),
                "target_objects": 2,
                "target_relations": 1,
                "visual_check_count": 6,
                "complexity_tag_count": 1,
                "complexity_tags": "two_object_single_relation",
                "strict_any_visual_hallucination": summary["strict_any_visual_hallucination"],
                "core_any_visual_hallucination": summary["core_any_visual_hallucination"],
                "strict_bad_rate": summary["strict_bad_rate"],
                "core_bad_rate": summary["core_bad_rate"],
            }
        )
    return rows


def complex_concept_rows() -> list[dict[str, Any]]:
    rows = []
    for row in read_jsonl(COMPLEX_CONCEPTS):
        target = row["target_semantics"]
        checks = row.get("visual_checks", [])
        tags = row.get("complexity_tags", [])
        rows.append(
            {
                "suite": "complex_proposed",
                "run": "complex24",
                "concept_id": row["concept_id"],
                "prompt_tokens": token_count(row["prompt"]),
                "target_slots": sum(1 for slot in SLOTS if slot in target),
                "target_objects": 2,
                "target_relations": 1,
                "visual_check_count": len(checks),
                "complexity_tag_count": len(tags),
                "complexity_tags": ";".join(tags),
                "strict_any_visual_hallucination": "",
                "core_any_visual_hallucination": "",
                "strict_bad_rate": "",
                "core_bad_rate": "",
            }
        )
    return rows


def suite_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for suite in sorted({row["suite"] for row in rows}):
        group = [row for row in rows if row["suite"] == suite]
        tag_counter = Counter()
        for row in group:
            for tag in str(row["complexity_tags"]).split(";"):
                if tag:
                    tag_counter[tag] += 1
        out.append(
            {
                "suite": suite,
                "concepts": len(group),
                "mean_prompt_tokens": mean(float(row["prompt_tokens"]) for row in group),
                "mean_visual_check_count": mean(float(row["visual_check_count"]) for row in group),
                "mean_complexity_tag_count": mean(float(row["complexity_tag_count"]) for row in group),
                "unique_complexity_tags": len(tag_counter),
                "top_complexity_tags": ";".join(f"{key}:{value}" for key, value in tag_counter.most_common(12)),
            }
        )
    return out


def visual_gt_summary() -> dict[str, Any]:
    concept_rows = read_csv(TRUE_VIS_DIR / "concept_visual_gt_summary.csv")
    strict_any = sum(int(row["strict_any_visual_hallucination"]) for row in concept_rows)
    core_any = sum(int(row["core_any_visual_hallucination"]) for row in concept_rows)
    strict_majority = sum(int(row["strict_majority_visual_hallucination"]) for row in concept_rows)
    core_majority = sum(int(row["core_majority_visual_hallucination"]) for row in concept_rows)
    image_rows = read_csv(GT_DIR / "codex_visual_gt_first_pass.csv")
    strict_bad_images = sum(1 for row in image_rows if row["strict_visual_match"] != "yes")
    core_bad_images = sum(1 for row in image_rows if row["core_scene_match"] != "yes")
    return {
        "concepts": len(concept_rows),
        "images": len(image_rows),
        "strict_any_concepts": strict_any,
        "core_any_concepts": core_any,
        "strict_majority_concepts": strict_majority,
        "core_majority_concepts": core_majority,
        "strict_bad_images": strict_bad_images,
        "core_bad_images": core_bad_images,
        "strict_bad_image_rate": strict_bad_images / len(image_rows),
        "core_bad_image_rate": core_bad_images / len(image_rows),
    }


def detector_summary() -> list[dict[str, Any]]:
    rows = read_csv(TRUE_VIS_DIR / "true_visual_uncertainty_detector_metrics.csv")
    keep = []
    for row in rows:
        if row["group"] != "combined/joint":
            continue
        if row["uncertainty_measure"] != "semantic_entropy":
            continue
        if row["visual_hallucination_label"] not in {
            "strict_any_visual_hallucination",
            "core_any_visual_hallucination",
            "background_any_visual_issue",
        }:
            continue
        keep.append(row)
    return keep


def duplicate_summary() -> dict[str, Any] | None:
    path = BASE / "image_embedding_duplicate_check" / "generated_embedding_similarity_summary.csv"
    if not path.exists():
        return None
    rows = read_csv(path)
    if not rows:
        return None
    rates = [float(row["high_similarity_pair_rate"]) for row in rows]
    collapse = sum(1 for row in rows if row["fallback_mode_collapse_flag"] == "True")
    return {
        "concepts": len(rows),
        "mean_high_similarity_pair_rate": mean(rates),
        "max_high_similarity_pair_rate": max(rates),
        "fallback_mode_collapse_concepts": collapse,
    }


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                value = fmt(value)
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def build_report(all_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> str:
    visual = visual_gt_summary()
    detector = detector_summary()
    dup = duplicate_summary()
    complex_tags = Counter()
    for row in read_jsonl(COMPLEX_CONCEPTS):
        complex_tags.update(row.get("complexity_tags", []))

    lines = [
        "# T2I Complex Scene Upgrade Report",
        "",
        "Status: READY_TO_RUN_COMPLEX_BENCHMARK",
        "",
        "## Why the current scene is too simple",
        "",
        f"- Current visual GT covers `{visual['concepts']}` concepts and `{visual['images']}` images, but the task schema is always two primitive objects, two colors, one binary spatial relation, and a background slot.",
        f"- True visual errors are sparse: strict bad images `{visual['strict_bad_images']}/{visual['images']}` ({fmt(visual['strict_bad_image_rate'])}), core bad images `{visual['core_bad_images']}/{visual['images']}` ({fmt(visual['core_bad_image_rate'])}).",
        f"- At concept level, strict-any positives are `{visual['strict_any_concepts']}/{visual['concepts']}` and core-any positives are `{visual['core_any_concepts']}/{visual['concepts']}`; both majority labels have `{visual['strict_majority_concepts']}` / `{visual['core_majority_concepts']}` positives.",
        "- This positive rate is too low for a stable hallucination detector evaluation: AUROC is defined, but the confidence boundary is weak because only a few concepts contribute positives.",
        "",
        "## Existing detector signal under true visual GT",
        "",
        *markdown_table(
            [
                {
                    "label": row["visual_hallucination_label"],
                    "positives": row["positives"],
                    "AUROC": fmt(row["AUROC"]) if row["AUROC"] else "",
                    "AUPRC": fmt(row["AUPRC"]) if row["AUPRC"] else "",
                    "mean_positive_entropy": fmt(row["mean_positive_score"]),
                    "mean_negative_entropy": fmt(row["mean_negative_score"]),
                }
                for row in detector
            ],
            ["label", "positives", "AUROC", "AUPRC", "mean_positive_entropy", "mean_negative_entropy"],
        ),
        "",
        "## Simplicity evidence from sample diversity",
        "",
    ]
    if dup:
        lines.extend(
            [
                f"- Formal generated-image duplicate audit covers `{dup['concepts']}` concepts.",
                f"- Mean high-similarity pair rate is `{fmt(dup['mean_high_similarity_pair_rate'])}` and max is `{fmt(dup['max_high_similarity_pair_rate'])}`.",
                f"- Fallback/mode-collapse flag appears in `{dup['fallback_mode_collapse_concepts']}` concepts.",
            ]
        )
    else:
        lines.append("- Duplicate audit file was not found in this checkout.")
    lines.extend(
        [
            "",
            "## Proposed complex benchmark",
            "",
            f"- Concept manifest: `{COMPLEX_CONCEPTS}`",
            "- Run launcher: `scripts/launch_semantic_uncertainty_complex_t2i_emu35.sh`",
            "- It keeps the current formal Emu3.5 T2I baseline: `model/Emu3.5`, `target_height=64`, `target_width=64`, `image_area=1048576`, `t2i_max_new_tokens=5120`, `classifier_free_guidance=2.0`, `image_top_k=5120`, `image_temperature=1.0`.",
            "- The current finite-slot automatic QA remains available for anchor slots, but the richer hallucination target should be the visual GT over `visual_checks` because current automatic slots cannot score count, occlusion, text, material, containment, and negative constraints.",
            "",
            "## Suite-level complexity comparison",
            "",
            *markdown_table(
                summary_rows,
                [
                    "suite",
                    "concepts",
                    "mean_prompt_tokens",
                    "mean_visual_check_count",
                    "mean_complexity_tag_count",
                    "unique_complexity_tags",
                    "top_complexity_tags",
                ],
            ),
            "",
            "## Complex axes covered",
            "",
        ]
    )
    for tag, count in complex_tags.most_common():
        lines.append(f"- `{tag}`: {count}")
    lines.extend(
        [
            "",
            "## Recommended next run",
            "",
            "```bash",
            "RUN_ID=semantic_uncertainty_complex_t2i_emu35_20260616 \\",
            "NUM_CONCEPTS=30 T2I_SAMPLES=10 GEN_WORKERS=2 STRICT_WORKERS=2 QA_WORKERS=2 \\",
            "DEVICE_GROUPS='0,3;5,7' \\",
            "bash scripts/launch_semantic_uncertainty_complex_t2i_emu35.sh",
            "```",
            "",
            "After generation, build contact sheets and annotate all complex `visual_checks`; then rerun true-visual-GT detector metrics with the complex GT. The expected research value is a higher positive rate and richer error axes, which makes the semantic entropy vs hallucination relationship testable beyond the current toy two-object setting.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    current = current_concept_rows()
    complex_rows = complex_concept_rows()
    all_rows = current + complex_rows
    summary = suite_summary(all_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "scene_complexity_comparison.csv", all_rows)
    write_csv(OUT_DIR / "scene_complexity_suite_summary.csv", summary)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_report(all_rows, summary), encoding="utf-8")
    print(REPORT_PATH)
    print(OUT_DIR / "scene_complexity_comparison.csv")
    print(OUT_DIR / "scene_complexity_suite_summary.csv")


if __name__ == "__main__":
    main()
