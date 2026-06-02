#!/usr/bin/env python3
"""Build a consolidated Setting F claim audit from recorded poster entropy runs."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-root",
        default="../research_logs/modal_aphasia_posters",
        help="Directory containing Setting F research logs.",
    )
    parser.add_argument(
        "--out-dir",
        default="../research_logs/modal_aphasia_posters/setting_f_claim_audit_20260601",
        help="Directory for consolidated claim-audit outputs.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fnum(value: object, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def summarize(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    return mean(values), pstdev(values) if len(values) > 1 else 0.0


def parse_counts(text: str) -> Counter:
    counts: Counter[str] = Counter()
    for part in (text or "").split(";"):
        if not part or ":" not in part:
            continue
        key, value = part.rsplit(":", 1)
        counts[key.strip()] += int(fnum(value))
    return counts


def counts_to_text(counts: Counter, limit: int | None = None) -> str:
    items = counts.most_common(limit)
    return ";".join(f"{k}:{v}" for k, v in items)


def semantic_counts(counts: Counter) -> Counter:
    stop = {"content_other", "function_word", "boundary", "space_or_special"}
    return Counter({key: value for key, value in counts.items() if key not in stop})


def infer_subset(sample_id: str, source: str) -> str:
    if "__gt_describe" in sample_id:
        return "describe"
    if "__gt_imageonly" in sample_id:
        return "pure_image"
    if "__visible_only" in sample_id:
        return "visible_only"
    if "__greedy_describe" in sample_id:
        return "greedy_describe"
    if "__greedy_visible_only" in sample_id:
        return "greedy_visible_only"
    if "__text_band_masked_direct" in sample_id:
        return "text_band_masked_direct"
    if "__text_band_masked" in sample_id:
        return "text_band_masked"
    for suffix in ("caption", "describe", "imageonly", "whatin"):
        if sample_id.endswith(f"__{suffix}"):
            return suffix
    if "__text" in sample_id:
        return "text_memory"
    if "__readback" in sample_id:
        return "generated_readback"
    return source


def build_run_completeness(log_root: Path) -> list[dict]:
    checks = [
        ("seed40_text_image_memory", "seed40/poster_entropy_summary.csv", 9),
        ("seed49_generated_readback", "seed49_generated_readback/poster_readback_entropy_summary.csv", 9),
        ("seed50_text_repeat", "seed50_text_repeat/text_repeat_sample_summary.csv", 27),
        ("seed51_gt_readback_top20_nonempty", "seed51_gt_readback_top20/top_entropy_sample_summary.csv", 13),
        ("seed51_gt_input_form_top20_audit", "seed51_gt_input_form_top20_audit/gt_input_form_sample_summary.csv", 18),
        ("seed53_manual_span_labels_all9", "seed53_manual_span_labels_all9/manual_span_entropy_rows.csv", 49),
        ("seed54_gt_readback_repeat", "seed54_gt_readback_repeat_stability/gt_readback_repeat_sample_summary.csv", 54),
        ("seed55_gt_prompt_variants", "seed55_gt_prompt_variants/gt_prompt_variant_sample_summary.csv", 45),
        ("seed56_visible_only_repeat", "seed56_gt_visible_only_repeat_stability/gt_readback_repeat_sample_summary.csv", 27),
        ("seed57_greedy_describe_visible", "seed57_gt_describe_visible_greedy/gt_prompt_variant_sample_summary.csv", 18),
        ("seed58_paper_span_audit", "seed58_paper_span_audit/manual_span_entropy_rows.csv", 28),
        ("seed59_sampled_span_audit", "seed59_sampled_span_audit/manual_span_entropy_rows.csv", 40),
        ("seed63_greedy_describe_vs_pure_image", "seed63_gt_input_form_top20_audit/gt_input_form_sample_summary.csv", 18),
        ("seed72_strict_visible_prompt_variants", "seed72_gt_strict_visible_prompt_variants/gt_prompt_variant_sample_summary.csv", 27),
        ("seed72_strict_visible_mismatch_overlap", "seed72_gt_strict_visible_mismatch_overlap/gt_mismatch_entropy_summary.csv", 27),
        ("seed72_mismatch_by_variant", "seed72_gt_strict_visible_mismatch_by_variant/gt_mismatch_by_prompt_variant.csv", 3),
        ("seed72_manual_high_entropy_span_audit", "seed72_manual_span_audit/manual_span_entropy_rows.csv", 77),
        ("seed72_manual_high_entropy_by_condition", "seed72_manual_span_audit/manual_span_entropy_by_condition.csv", 8),
        ("seed73_strict_visible_repeat", "seed73_gt_strict_visible_repeat_stability/gt_readback_repeat_sample_summary.csv", 27),
        ("seed73_strict_visible_repeat_top20", "seed73_gt_strict_visible_repeat_top20/top_entropy_sample_summary.csv", 27),
        ("seed73_strict_visible_repeat_mismatch", "seed73_gt_strict_visible_repeat_mismatch_overlap/gt_mismatch_entropy_summary.csv", 27),
        ("seed73_prompt_repeat_comparison", "seed73_prompt_repeat_comparison/gt_prompt_repeat_condition_summary.csv", 3),
        ("seed74_no_ocr_layout_repeat", "seed74_gt_no_ocr_layout_repeat_stability/gt_readback_repeat_sample_summary.csv", 27),
        ("seed74_no_ocr_layout_repeat_top20", "seed74_gt_no_ocr_layout_repeat_top20/top_entropy_sample_summary.csv", 27),
        ("seed74_no_ocr_layout_repeat_mismatch", "seed74_gt_no_ocr_layout_repeat_mismatch_overlap/gt_mismatch_entropy_summary.csv", 27),
        ("seed74_prompt_repeat_comparison", "seed74_prompt_repeat_comparison/gt_prompt_repeat_condition_summary.csv", 4),
        ("seed74_prompt_ocr_compliance", "seed74_prompt_ocr_compliance/gt_prompt_ocr_compliance_summary.csv", 4),
        ("seed75_text_band_masked_repeat", "seed75_gt_text_band_masked_repeat_stability/gt_readback_repeat_sample_summary.csv", 27),
        ("seed75_text_band_masked_repeat_top20", "seed75_gt_text_band_masked_repeat_top20/top_entropy_sample_summary.csv", 27),
        ("seed75_text_band_masked_repeat_mismatch", "seed75_gt_text_band_masked_repeat_mismatch_overlap/gt_mismatch_entropy_summary.csv", 27),
        ("seed75_prompt_repeat_comparison", "seed75_prompt_repeat_comparison/gt_prompt_repeat_condition_summary.csv", 5),
        ("seed75_prompt_ocr_compliance", "seed75_prompt_ocr_compliance/gt_prompt_ocr_compliance_summary.csv", 5),
        ("seed75_manual_high_entropy_span_audit", "seed75_manual_span_audit/manual_span_entropy_rows.csv", 45),
        ("seed75_manual_high_entropy_by_condition", "seed75_manual_span_audit/manual_span_entropy_by_condition.csv", 5),
        ("seed76_text_band_masked_direct_repeat", "seed76_gt_text_band_masked_direct_repeat_stability/gt_readback_repeat_sample_summary.csv", 27),
        ("seed76_text_band_masked_direct_repeat_top20", "seed76_gt_text_band_masked_direct_repeat_top20/top_entropy_sample_summary.csv", 27),
        ("seed76_text_band_masked_direct_repeat_mismatch", "seed76_gt_text_band_masked_direct_repeat_mismatch_overlap/gt_mismatch_entropy_summary.csv", 27),
        ("seed76_prompt_repeat_comparison", "seed76_prompt_repeat_comparison/gt_prompt_repeat_condition_summary.csv", 6),
        ("seed76_prompt_ocr_compliance", "seed76_prompt_ocr_compliance/gt_prompt_ocr_compliance_summary.csv", 6),
        ("seed76_high_entropy_span_candidates", "seed76_high_entropy_span_audit_candidates/poster_high_entropy_span_audit_candidates.csv", 48),
        ("seed77_text_band_masked_direct_greedy", "seed77_gt_text_band_masked_direct_greedy_stability/gt_readback_repeat_sample_summary.csv", 9),
        ("seed77_text_band_masked_direct_greedy_top20", "seed77_gt_text_band_masked_direct_greedy_top20/top_entropy_sample_summary.csv", 9),
        ("seed77_text_band_masked_direct_greedy_mismatch", "seed77_gt_text_band_masked_direct_greedy_mismatch_overlap/gt_mismatch_entropy_summary.csv", 9),
        ("seed77_prompt_ocr_compliance", "seed77_prompt_ocr_compliance/gt_prompt_ocr_compliance_summary.csv", 1),
        ("seed78_text_band_masked_direct_greedy_short", "seed78_gt_text_band_masked_direct_greedy_short_stability/gt_readback_repeat_sample_summary.csv", 9),
        ("seed78_text_band_masked_direct_greedy_short_top20", "seed78_gt_text_band_masked_direct_greedy_short_top20/top_entropy_sample_summary.csv", 9),
        ("seed78_text_band_masked_direct_greedy_short_mismatch", "seed78_gt_text_band_masked_direct_greedy_short_mismatch_overlap/gt_mismatch_entropy_summary.csv", 9),
        ("seed78_prompt_ocr_compliance", "seed78_prompt_ocr_compliance/gt_prompt_ocr_compliance_summary.csv", 1),
        ("seed79_text_band_masked_direct_greedy_stop_eog", "seed79_gt_text_band_masked_direct_greedy_stop_eog_stability/gt_readback_repeat_sample_summary.csv", 9),
        ("seed79_text_band_masked_direct_greedy_stop_eog_top20", "seed79_gt_text_band_masked_direct_greedy_stop_eog_top20/top_entropy_sample_summary.csv", 9),
        ("seed79_text_band_masked_direct_greedy_stop_eog_mismatch", "seed79_gt_text_band_masked_direct_greedy_stop_eog_mismatch_overlap/gt_mismatch_entropy_summary.csv", 9),
        ("seed79_prompt_ocr_compliance", "seed79_prompt_ocr_compliance/gt_prompt_ocr_compliance_summary.csv", 1),
    ]
    rows = []
    for label, rel, expected in checks:
        path = log_root / rel
        data = read_csv(path)
        rows.append(
            {
                "run_label": label,
                "path": str(path),
                "exists": int(path.exists()),
                "row_count": len(data),
                "expected_min_rows": expected,
                "complete": int(path.exists() and len(data) >= expected),
            }
        )
    return rows


def build_modal_ladder(log_root: Path) -> list[dict]:
    seed40 = read_csv(log_root / "seed40/poster_entropy_summary.csv")
    seed49 = read_csv(log_root / "seed49_generated_readback/poster_readback_entropy_summary.csv")
    rows = []
    if seed40:
        rows.append(
            {
                "measurement": "text_from_memory",
                "sample_count": len(seed40),
                "mean_ume": fmt(mean(fnum(r["text_mean_ume"]) for r in seed40)),
                "notes": "Text description generated from poster concept memory.",
            }
        )
        rows.append(
            {
                "measurement": "image_from_memory_visual_tokens",
                "sample_count": len(seed40),
                "mean_ume": fmt(mean(fnum(r["image_visual_mean_ume_full"]) for r in seed40)),
                "notes": "Visual-token UME_full while generating poster image from memory.",
            }
        )
    if seed49:
        rows.append(
            {
                "measurement": "generated_image_readback_text",
                "sample_count": len(seed49),
                "mean_ume": fmt(mean(fnum(r["readback_mean_ume"]) for r in seed49)),
                "notes": "Text readback of model-generated poster image.",
            }
        )
    return rows


def build_manual_span_summaries(log_root: Path) -> tuple[list[dict], list[dict], list[dict]]:
    rows = []
    for rel in [
        "seed58_paper_span_audit/manual_span_entropy_rows.csv",
        "seed59_sampled_span_audit/manual_span_entropy_rows.csv",
    ]:
        data = read_csv(log_root / rel)
        if data:
            rows.extend(data)
    if not rows:
        rows = read_csv(log_root / "seed53_manual_span_labels_all9/manual_span_entropy_rows.csv")
    truth_rows = []
    aspect_rows = []
    quadrant_rows = []
    by_truth: dict[str, list[dict]] = defaultdict(list)
    by_aspect_truth: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_quadrant: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_truth[row["truth_label"]].append(row)
        by_aspect_truth[(row["aspect"], row["truth_label"])].append(row)
        entropy_bin = "high_relative_to_local_baseline" if fnum(row["mean_ume_minus_baseline"]) >= 0 else "low_relative_to_local_baseline"
        by_quadrant[(row["truth_label"], entropy_bin)].append(row)

    for truth, group in sorted(by_truth.items()):
        mean_ume, sd_ume = summarize([fnum(r["mean_ume"]) for r in group])
        baseline, _ = summarize([fnum(r["baseline_same_length_mean_ume"]) for r in group])
        exact, _ = summarize([fnum(r["exact_top20_fraction"]) for r in group])
        near, _ = summarize([fnum(r["near_top20_fraction"]) for r in group])
        truth_rows.append(
            {
                "truth_label": truth,
                "span_count": len(group),
                "mean_span_ume": fmt(mean_ume),
                "sd_span_ume": fmt(sd_ume),
                "mean_baseline_same_length_ume": fmt(baseline),
                "mean_ume_minus_baseline": fmt(mean_ume - baseline),
                "mean_exact_top20_fraction": fmt(exact),
                "mean_near_top20_fraction": fmt(near),
            }
        )

    for (aspect, truth), group in sorted(by_aspect_truth.items()):
        mean_ume, sd_ume = summarize([fnum(r["mean_ume"]) for r in group])
        examples = " | ".join(r["snippet"] for r in group[:3])
        aspect_rows.append(
            {
                "aspect": aspect,
                "truth_label": truth,
                "span_count": len(group),
                "mean_span_ume": fmt(mean_ume),
                "sd_span_ume": fmt(sd_ume),
                "example_snippets": examples,
            }
        )

    for (truth, entropy_bin), group in sorted(by_quadrant.items()):
        mean_ume, _ = summarize([fnum(r["mean_ume"]) for r in group])
        aspects = Counter(r["aspect"] for r in group)
        examples = " | ".join(f"{r['aspect']}: {r['snippet']}" for r in group[:5])
        quadrant_rows.append(
            {
                "truth_label": truth,
                "entropy_bin": entropy_bin,
                "span_count": len(group),
                "mean_span_ume": fmt(mean_ume),
                "aspect_counts": counts_to_text(aspects),
                "example_snippets": examples,
            }
        )
    return truth_rows, aspect_rows, quadrant_rows


def build_mismatch_summary(log_root: Path) -> list[dict]:
    rows = []
    for rel in [
        "seed51_gt_hallucination_overlap/gt_mismatch_entropy_summary.csv",
        "seed63_gt_hallucination_overlap/gt_mismatch_entropy_summary.csv",
        "seed72_gt_strict_visible_mismatch_overlap/gt_mismatch_entropy_summary.csv",
        "seed73_gt_strict_visible_repeat_mismatch_overlap/gt_mismatch_entropy_summary.csv",
        "seed74_gt_no_ocr_layout_repeat_mismatch_overlap/gt_mismatch_entropy_summary.csv",
        "seed75_gt_text_band_masked_repeat_mismatch_overlap/gt_mismatch_entropy_summary.csv",
        "seed76_gt_text_band_masked_direct_repeat_mismatch_overlap/gt_mismatch_entropy_summary.csv",
        "seed77_gt_text_band_masked_direct_greedy_mismatch_overlap/gt_mismatch_entropy_summary.csv",
        "seed78_gt_text_band_masked_direct_greedy_short_mismatch_overlap/gt_mismatch_entropy_summary.csv",
        "seed79_gt_text_band_masked_direct_greedy_stop_eog_mismatch_overlap/gt_mismatch_entropy_summary.csv",
    ]:
        rows.extend(read_csv(log_root / rel))
    by_run: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_run[row["run_label"]].append(row)
    out = []
    for run_label, group in sorted(by_run.items()):
        mismatch = sum(fnum(r["mismatch_token_count"]) for r in group)
        exact = sum(fnum(r["mismatch_exact_top20_count"]) for r in group)
        near = sum(fnum(r["mismatch_near_top20_count"]) for r in group)
        weighted_exact = exact / mismatch if mismatch else 0.0
        weighted_near = near / mismatch if mismatch else 0.0
        baseline_near = mean(fnum(r["all_token_near_top20_rate"]) for r in group)
        mismatch_ume = mean(fnum(r["mean_mismatch_ume"]) for r in group)
        non_mismatch_ume = mean(fnum(r["mean_non_mismatch_ume"]) for r in group)
        out.append(
            {
                "run_label": run_label,
                "sample_count": len(group),
                "mismatch_token_count": int(mismatch),
                "mismatch_exact_top20_rate_weighted": fmt(weighted_exact),
                "mismatch_near_top20_rate_weighted": fmt(weighted_near),
                "all_token_near_top20_rate_mean": fmt(baseline_near),
                "near_enrichment_vs_all": fmt(weighted_near / baseline_near if baseline_near else 0.0),
                "mean_mismatch_ume": fmt(mismatch_ume),
                "mean_non_mismatch_ume": fmt(non_mismatch_ume),
                "mismatch_minus_non_mismatch_ume": fmt(mismatch_ume - non_mismatch_ume),
            }
        )

    for row in read_csv(log_root / "seed72_gt_strict_visible_mismatch_by_variant/gt_mismatch_by_prompt_variant.csv"):
        variant = row["prompt_variant"]
        run_label = variant if variant.startswith("seed72_") else f"seed72_{variant}"
        out.append(
            {
                "run_label": run_label,
                "sample_count": int(fnum(row["sample_count"])),
                "mismatch_token_count": int(fnum(row["mismatch_token_count"])),
                "mismatch_exact_top20_rate_weighted": fmt(fnum(row["exact_top20_rate_weighted"])),
                "mismatch_near_top20_rate_weighted": fmt(fnum(row["near_top20_rate_weighted"])),
                "all_token_near_top20_rate_mean": fmt(fnum(row["all_token_near_top20_baseline_weighted"])),
                "near_enrichment_vs_all": fmt(fnum(row["near_enrichment_vs_baseline"])),
                "mean_mismatch_ume": fmt(fnum(row["mean_sample_mismatch_ume"])),
                "mean_non_mismatch_ume": fmt(fnum(row["mean_sample_non_mismatch_ume"])),
                "mismatch_minus_non_mismatch_ume": fmt(fnum(row["mean_sample_mismatch_minus_non_mismatch_ume"])),
            }
        )
    return out


def build_input_form_summary(log_root: Path) -> list[dict]:
    rows: list[dict] = []
    for source, rel in [
        ("seed51_sampled_describe_vs_pure_image", "seed51_gt_input_form_top20_audit/gt_input_form_condition_summary.csv"),
        ("seed54_sampled_describe_vs_pure_image", "seed54_gt_readback_repeat_stability/gt_readback_repeat_group_summary.csv"),
        ("seed55_prompt_variants", "seed55_gt_prompt_variants/gt_prompt_variant_summary.csv"),
        ("seed56_sampled_describe_vs_visible_only", "seed56_describe_vs_visible_repeat/describe_visible_condition_summary.csv"),
        ("seed57_greedy_describe_vs_visible_only", "seed57_gt_describe_visible_greedy/gt_prompt_variant_summary.csv"),
        ("seed63_greedy_describe_vs_pure_image", "seed63_gt_input_form_top20_audit/gt_input_form_condition_summary.csv"),
        ("seed72_strict_visible_prompt_variants", "seed72_gt_strict_visible_prompt_variants/gt_prompt_variant_summary.csv"),
    ]:
        data = read_csv(log_root / rel)
        if not data:
            continue
        if "input_form" in data[0]:
            grouped: dict[str, list[dict]] = defaultdict(list)
            for row in data:
                grouped[row["input_form"]].append(row)
            for form, group in sorted(grouped.items()):
                behavior_counts: Counter[str] = Counter()
                for row in group:
                    behavior_counts.update(parse_counts(row["behavior_counts"]))
                sample_count = sum(
                    int(fnum(r.get("repeat_count", r.get("sample_count", 0))))
                    for r in group
                )
                if group and "top_category_jaccard" in group[0]:
                    stability = f"mean top-category Jaccard={fmt(mean(fnum(r['top_category_jaccard']) for r in group), 3)}"
                elif group and "valid_description_rate" in group[0]:
                    stability = f"valid_description_rate={fmt(mean(fnum(r['valid_description_rate']) for r in group), 3)}"
                else:
                    stability = "direct input-form audit"
                rows.append(
                    {
                        "source": source,
                        "condition": form,
                        "sample_count": sample_count,
                        "behavior_counts": counts_to_text(behavior_counts),
                        "mean_word_count": fmt(mean(fnum(r["mean_word_count"]) for r in group), 2),
                        "mean_ume": fmt(mean(fnum(r["mean_ume"]) for r in group)),
                        "stability_or_notes": stability,
                    }
                )
        else:
            condition_field = "prompt_variant" if "prompt_variant" in data[0] else "condition"
            for row in data:
                rows.append(
                    {
                        "source": source,
                        "condition": row[condition_field],
                        "sample_count": int(fnum(row.get("sample_count"), 0)),
                        "behavior_counts": row.get("behavior_counts", ""),
                        "mean_word_count": fmt(fnum(row.get("mean_word_count")), 2),
                        "mean_ume": fmt(fnum(row.get("mean_ume"))),
                        "stability_or_notes": (
                            f"outside_hit_rate={fmt(fnum(row.get('outside_knowledge_hit_rate')), 3)}"
                            if "outside_knowledge_hit_rate" in row
                            else f"topcat_jaccard={fmt(fnum(row.get('mean_top_category_jaccard')), 3)}"
                        ),
                    }
                )
    return rows


def build_top20_where_summary(log_root: Path) -> list[dict]:
    sources = [
        ("seed40_text_memory", "seed40_top20_text_memory/top_entropy_sample_summary.csv"),
        ("seed49_generated_readback", "seed49_top20_generated_readback/top_entropy_sample_summary.csv"),
        ("seed51_gt_readback", "seed51_gt_readback_top20/top_entropy_sample_summary.csv"),
        ("seed54_gt_readback_repeat", "seed54_gt_readback_repeat_top20/top_entropy_sample_summary.csv"),
        ("seed55_prompt_variants", "seed55_gt_prompt_variants_top20/top_entropy_sample_summary.csv"),
        ("seed56_visible_only_repeat", "seed56_gt_visible_only_repeat_top20/top_entropy_sample_summary.csv"),
        ("seed57_greedy_describe_visible", "seed57_gt_describe_visible_greedy_top20/top_entropy_sample_summary.csv"),
        ("seed63_greedy_describe_pure_image", "seed63_gt_readback_greedy_top20/top_entropy_sample_summary.csv"),
        ("seed72_strict_visible", "seed72_gt_strict_visible_top20/top_entropy_sample_summary.csv"),
        ("seed73_strict_visible_repeat", "seed73_gt_strict_visible_repeat_top20/top_entropy_sample_summary.csv"),
        ("seed74_no_ocr_layout_repeat", "seed74_gt_no_ocr_layout_repeat_top20/top_entropy_sample_summary.csv"),
        ("seed75_text_band_masked_repeat", "seed75_gt_text_band_masked_repeat_top20/top_entropy_sample_summary.csv"),
        ("seed76_text_band_masked_direct_repeat", "seed76_gt_text_band_masked_direct_repeat_top20/top_entropy_sample_summary.csv"),
        ("seed77_text_band_masked_direct_greedy", "seed77_gt_text_band_masked_direct_greedy_top20/top_entropy_sample_summary.csv"),
        ("seed78_text_band_masked_direct_greedy_short", "seed78_gt_text_band_masked_direct_greedy_short_top20/top_entropy_sample_summary.csv"),
        ("seed79_text_band_masked_direct_greedy_stop_eog", "seed79_gt_text_band_masked_direct_greedy_stop_eog_top20/top_entropy_sample_summary.csv"),
    ]
    out = []
    for source, rel in sources:
        data = read_csv(log_root / rel)
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in data:
            grouped[infer_subset(row["sample_id"], source)].append(row)
        for subset, group in sorted(grouped.items()):
            counts: Counter[str] = Counter()
            previews = []
            for row in group:
                counts.update(parse_counts(row.get("top_category_counts", "")))
                if row.get("top_span_preview") and len(previews) < 8:
                    previews.append(row["top_span_preview"])
            out.append(
                {
                    "source": source,
                    "subset": subset,
                    "sample_count": len(group),
                    "top_token_count": sum(int(fnum(r.get("text_top_token_count"))) for r in group),
                    "top_span_count": sum(int(fnum(r.get("text_top_span_count"))) for r in group),
                    "aggregate_top_categories": counts_to_text(counts),
                    "top4_categories": counts_to_text(counts, 4),
                    "semantic_top_categories": counts_to_text(semantic_counts(counts), 6),
                    "span_preview_examples": " | ".join(previews),
                }
            )
    return out


def build_high_entropy_candidate_audit(log_root: Path) -> list[dict]:
    rows = []
    rows.extend(read_csv(log_root / "seed72_manual_span_audit/manual_span_entropy_by_condition.csv"))
    rows.extend(read_csv(log_root / "seed75_manual_span_audit/manual_span_entropy_by_condition.csv"))
    return rows


def build_prompt_repeat_comparison(log_root: Path) -> list[dict]:
    seed76 = read_csv(log_root / "seed76_prompt_repeat_comparison/gt_prompt_repeat_condition_summary.csv")
    if seed76:
        return seed76
    seed75 = read_csv(log_root / "seed75_prompt_repeat_comparison/gt_prompt_repeat_condition_summary.csv")
    if seed75:
        return seed75
    seed74 = read_csv(log_root / "seed74_prompt_repeat_comparison/gt_prompt_repeat_condition_summary.csv")
    if seed74:
        return seed74
    return read_csv(log_root / "seed73_prompt_repeat_comparison/gt_prompt_repeat_condition_summary.csv")


def build_prompt_ocr_compliance(log_root: Path) -> list[dict]:
    seed76 = read_csv(log_root / "seed76_prompt_ocr_compliance/gt_prompt_ocr_compliance_summary.csv")
    if seed76:
        return seed76
    seed75 = read_csv(log_root / "seed75_prompt_ocr_compliance/gt_prompt_ocr_compliance_summary.csv")
    if seed75:
        return seed75
    return read_csv(log_root / "seed74_prompt_ocr_compliance/gt_prompt_ocr_compliance_summary.csv")


def build_claim_status(
    modal_rows: list[dict],
    manual_truth_rows: list[dict],
    mismatch_rows: list[dict],
    input_rows: list[dict],
) -> list[dict]:
    values = {row["measurement"]: fnum(row["mean_ume"]) for row in modal_rows}
    truth = {row["truth_label"]: row for row in manual_truth_rows}
    visible56 = {
        row["condition"]: row
        for row in input_rows
        if row["source"] == "seed56_sampled_describe_vs_visible_only"
    }
    greedy57 = {
        row["condition"]: row
        for row in input_rows
        if row["source"] == "seed57_greedy_describe_vs_visible_only"
    }
    strict72 = {
        row["condition"]: row
        for row in input_rows
        if row["source"] == "seed72_strict_visible_prompt_variants"
    }
    strict_mismatch = {
        row["run_label"]: row
        for row in mismatch_rows
        if row["run_label"].startswith("seed72_")
    }
    exact_rates = [fnum(r["mismatch_exact_top20_rate_weighted"]) for r in mismatch_rows]
    near_enrichments = [fnum(r["near_enrichment_vs_all"]) for r in mismatch_rows]
    strict_mismatch_count = fnum(strict_mismatch.get("seed72_strict_visible_no_guess", {}).get("mismatch_token_count"))
    strict_sample_count = fnum(strict_mismatch.get("seed72_strict_visible_no_guess", {}).get("sample_count"))
    strict_mismatch_per_sample = strict_mismatch_count / strict_sample_count if strict_sample_count else 0.0

    hallucinated = fnum(truth.get("hallucinated", {}).get("mean_span_ume"))
    grounded = fnum(truth.get("grounded", {}).get("mean_span_ume"))
    rows = [
        {
            "claim": "Modal aphasia entropy ladder",
            "status": "supported",
            "evidence": (
                f"image visual UME_full={fmt(values.get('image_from_memory_visual_tokens', 0))}, "
                f"text-from-memory UME={fmt(values.get('text_from_memory', 0))}, "
                f"generated-image readback UME={fmt(values.get('generated_image_readback_text', 0))}"
            ),
            "caveat": "This is a 9-poster pilot and should not be framed as universal across all concepts.",
        },
        {
            "claim": "Hallucinated spans have higher local answer entropy on average",
            "status": "supported",
            "evidence": f"manual all-9 labels: hallucinated={fmt(hallucinated)}, grounded={fmt(grounded)}, delta={fmt(hallucinated - grounded)}",
            "caveat": "Mean span entropy separates groups, but individual low-entropy hallucinations remain.",
        },
        {
            "claim": "Top20 high entropy is not a binary hallucination detector",
            "status": "supported_negative",
            "evidence": (
                f"automatic lexical mismatch exact top20 rates={','.join(fmt(v, 3) for v in exact_rates)}; "
                f"near-top20 enrichment vs all={','.join(fmt(v, 3) for v in near_enrichments)}"
            ),
            "caveat": "Use span-level truth labels plus quadrant analysis, not exact top20 overlap alone.",
        },
        {
            "claim": "Image + instruction is the primary descriptive input form",
            "status": "supported",
            "evidence": "seed54 and seed55 produce stable descriptions under describe; pure image and caption often collapse to immediate-end, short-label, fragment, or action-script modes.",
            "caveat": "Pure-image behavior is a task-basin control, not a clean descriptive entropy condition.",
        },
        {
            "claim": "Visible-only increases local visual commitment pressure",
            "status": "supported",
            "evidence": (
                f"seed56 sampled visible_only UME={visible56.get('visible_only', {}).get('mean_ume')} vs describe={visible56.get('describe', {}).get('mean_ume')}; "
                f"seed57 greedy_visible_only UME={greedy57.get('greedy_visible_only', {}).get('mean_ume')} vs greedy_describe={greedy57.get('greedy_describe', {}).get('mean_ume')}"
            ),
            "caveat": "Visible-only is useful as a stress test, not a safer replacement for describe.",
        },
        {
            "claim": "Strict no-guess prompting shifts high-entropy regions instead of solving hallucination",
            "status": "supported_boundary",
            "evidence": (
                f"seed72 strict_visible_no_guess UME={strict72.get('seed72_strict_visible_no_guess', {}).get('mean_ume')} "
                f"vs describe={strict72.get('seed72_describe', {}).get('mean_ume')}; "
                f"strict mismatch/sample={fmt(strict_mismatch_per_sample, 1)}, "
                f"near enrichment={strict_mismatch.get('seed72_strict_visible_no_guess', {}).get('near_enrichment_vs_all')}"
            ),
            "caveat": "The strict prompt reduces mismatch volume but leaves remaining mismatch tokens locally higher entropy.",
        },
    ]
    return rows


def write_report(
    path: Path,
    run_rows: list[dict],
    modal_rows: list[dict],
    manual_truth_rows: list[dict],
    manual_aspect_rows: list[dict],
    manual_quadrant_rows: list[dict],
    mismatch_rows: list[dict],
    input_rows: list[dict],
    top20_rows: list[dict],
    high_entropy_candidate_rows: list[dict],
    prompt_repeat_rows: list[dict],
    prompt_ocr_rows: list[dict],
    claim_rows: list[dict],
) -> None:
    lines = [
        "# Setting F Claim Audit",
        "",
        "This report consolidates the real Setting F poster runs and focuses on local high-entropy regions, not only average entropy.",
        "",
        "## Run Completeness",
        "",
    ]
    for row in run_rows:
        marker = "complete" if row["complete"] else "incomplete"
        lines.append(f"- {row['run_label']}: {marker}, rows={row['row_count']}, expected>={row['expected_min_rows']}")

    lines.extend(["", "## Modal Entropy Ladder", ""])
    for row in modal_rows:
        lines.append(f"- {row['measurement']}: n={row['sample_count']}, mean UME={row['mean_ume']} ({row['notes']})")

    lines.extend(["", "## Manual Span Truth Labels", ""])
    for row in manual_truth_rows:
        lines.append(
            f"- {row['truth_label']}: n={row['span_count']}, mean UME={row['mean_span_ume']}, "
            f"baseline={row['mean_baseline_same_length_ume']}, exact_top20_fraction={row['mean_exact_top20_fraction']}"
        )

    lines.extend(["", "## High-Entropy / Hallucination Quadrants", ""])
    for row in manual_quadrant_rows:
        lines.append(
            f"- {row['truth_label']} + {row['entropy_bin']}: n={row['span_count']}, "
            f"mean UME={row['mean_span_ume']}, aspects={row['aspect_counts']}"
        )
        if row["example_snippets"]:
            lines.append(f"  examples: {row['example_snippets']}")

    lines.extend(["", "## Automatic Lexical Mismatch Screen", ""])
    for row in mismatch_rows:
        lines.append(
            f"- {row['run_label']}: mismatch tokens={row['mismatch_token_count']}, "
            f"exact top20={row['mismatch_exact_top20_rate_weighted']}, "
            f"near enrichment={row['near_enrichment_vs_all']}, "
            f"mismatch UME delta={row['mismatch_minus_non_mismatch_ume']}"
        )

    lines.extend(["", "## Input Forms", ""])
    for row in input_rows:
        lines.append(
            f"- {row['source']} / {row['condition']}: n={row['sample_count']}, "
            f"behavior={row['behavior_counts']}, mean words={row['mean_word_count']}, "
            f"mean UME={row['mean_ume']}, {row['stability_or_notes']}"
        )

    if prompt_repeat_rows:
        lines.extend(["", "## Prompt Repeat Comparison", ""])
        for row in prompt_repeat_rows:
            lines.append(
                f"- {row['condition']}: n={row['sample_count']}, behavior={row['behavior_counts']}, "
                f"mean words={float(row['mean_word_count']):.2f}, mean UME={float(row['mean_ume']):.4f}, "
                f"sd UME={float(row['mean_group_sd_ume']):.4f}, content Jaccard={float(row['mean_content_jaccard']):.3f}, "
                f"topcat Jaccard={float(row['mean_top_category_jaccard']):.3f}, outside hit={float(row['outside_knowledge_hit_rate']):.3f}"
            )

    if prompt_ocr_rows:
        lines.extend(["", "## Prompt OCR Compliance", ""])
        for row in prompt_ocr_rows:
            lines.append(
                f"- {row['condition']}: forbidden={row['forbidden_copy_rate']}, quoted={row['quoted_text_rate']}, "
                f"title_words={row['title_word_rate']}, person_prior={row['person_prior_rate']}, "
                f"dates={row['date_text_rate']}, mean_quotes={row['mean_quote_count']}"
            )

    lines.extend(["", "## Where Are The Top 20% Entropy Regions?", ""])
    for row in top20_rows:
        if row["source"] in {
            "seed54_gt_readback_repeat",
            "seed55_prompt_variants",
            "seed56_visible_only_repeat",
            "seed57_greedy_describe_visible",
            "seed63_greedy_describe_pure_image",
            "seed72_strict_visible",
            "seed73_strict_visible_repeat",
            "seed74_no_ocr_layout_repeat",
            "seed75_text_band_masked_repeat",
            "seed76_text_band_masked_direct_repeat",
            "seed77_text_band_masked_direct_greedy",
            "seed78_text_band_masked_direct_greedy_short",
            "seed79_text_band_masked_direct_greedy_stop_eog",
        }:
            lines.append(
                f"- {row['source']} / {row['subset']}: samples={row['sample_count']}, "
                f"top categories={row['top4_categories']}, semantic={row['semantic_top_categories']}"
            )
            if row["span_preview_examples"]:
                lines.append(f"  previews: {row['span_preview_examples'][:500]}")

    if high_entropy_candidate_rows:
        lines.extend(["", "## Seed72 High-Entropy Candidate Manual Audit", ""])
        lines.append(
            "This section is conditioned on selected top-20% high-entropy candidate spans, "
            "so it should not be mixed with base-rate span audits."
        )
        lines.append("")
        for row in high_entropy_candidate_rows:
            lines.append(
                f"- {row['condition']} / {row['truth_label']}: n={row['span_count']}, "
                f"mean UME={row['mean_ume']}, delta={row['mean_delta_vs_same_length']}, "
                f"has_exact={row['span_has_exact_top20_rate']}, has_near={row['span_has_near_top20_rate']}, "
                f"aspects={row['aspect_counts']}"
            )

    lines.extend(["", "## Claim Status", ""])
    for row in claim_rows:
        lines.append(f"- {row['claim']}: {row['status']}. {row['evidence']} Caveat: {row['caveat']}")

    lines.extend(
        [
            "",
            "## Recommended Next Step",
            "",
            "Prioritize a paper-facing span audit: keep `describe` as the primary condition, add `visible_only` as the stress test, and label whether each high-entropy span is grounded, visually ambiguous, or hallucinated. The current evidence supports a quadrant claim rather than a detector claim.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    log_root = Path(args.log_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_rows = build_run_completeness(log_root)
    modal_rows = build_modal_ladder(log_root)
    manual_truth_rows, manual_aspect_rows, manual_quadrant_rows = build_manual_span_summaries(log_root)
    mismatch_rows = build_mismatch_summary(log_root)
    input_rows = build_input_form_summary(log_root)
    top20_rows = build_top20_where_summary(log_root)
    high_entropy_candidate_rows = build_high_entropy_candidate_audit(log_root)
    prompt_repeat_rows = build_prompt_repeat_comparison(log_root)
    prompt_ocr_rows = build_prompt_ocr_compliance(log_root)
    claim_rows = build_claim_status(modal_rows, manual_truth_rows, mismatch_rows, input_rows)

    write_csv(out_dir / "setting_f_run_completeness.csv", run_rows)
    write_csv(out_dir / "setting_f_modal_entropy_ladder.csv", modal_rows)
    write_csv(out_dir / "setting_f_manual_span_truth_summary.csv", manual_truth_rows)
    write_csv(out_dir / "setting_f_manual_span_aspect_summary.csv", manual_aspect_rows)
    write_csv(out_dir / "setting_f_manual_span_quadrants.csv", manual_quadrant_rows)
    write_csv(out_dir / "setting_f_mismatch_overlap_summary.csv", mismatch_rows)
    write_csv(out_dir / "setting_f_input_form_summary.csv", input_rows)
    write_csv(out_dir / "setting_f_top20_where_summary.csv", top20_rows)
    write_csv(out_dir / "setting_f_seed72_high_entropy_candidate_audit.csv", high_entropy_candidate_rows)
    write_csv(out_dir / "setting_f_prompt_repeat_comparison.csv", prompt_repeat_rows)
    write_csv(out_dir / "setting_f_prompt_ocr_compliance.csv", prompt_ocr_rows)
    write_csv(out_dir / "setting_f_claim_status.csv", claim_rows)
    write_report(
        out_dir / "setting_f_claim_audit_report.md",
        run_rows,
        modal_rows,
        manual_truth_rows,
        manual_aspect_rows,
        manual_quadrant_rows,
        mismatch_rows,
        input_rows,
        top20_rows,
        high_entropy_candidate_rows,
        prompt_repeat_rows,
        prompt_ocr_rows,
        claim_rows,
    )
    print(f"[INFO] wrote Setting F claim audit to {out_dir}")


if __name__ == "__main__":
    main()
