#!/usr/bin/env python3
"""Build a compact audit table for poster answer high-entropy spans."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


SEMANTIC_CATEGORIES = {
    "proper_name_or_title",
    "spatial_composition",
    "object_or_character",
    "color_attribute",
    "poster_text_or_ocr",
    "hedge_or_uncertainty",
    "action_script",
}

RUN_DIRS = {
    "seed54_sampled_describe": "outputs/emu3p5-main/modal_aphasia_poster_gt_readback_repeat/ume_trace_runs/main_modal_aphasia_posters_gt_readback_repeat_seed54_20260601",
    "seed56_sampled_visible_only": "outputs/emu3p5-main/modal_aphasia_poster_gt_visible_only_repeat/ume_trace_runs/main_modal_aphasia_posters_gt_visible_only_repeat_seed56_20260601",
    "seed57_greedy_describe_visible": "outputs/emu3p5-main/modal_aphasia_poster_gt_describe_visible_greedy/ume_trace_runs/main_modal_aphasia_posters_gt_describe_visible_greedy_seed57_20260601",
    "seed72_strict_visible": "outputs/emu3p5-main/modal_aphasia_poster_gt_strict_visible/ume_trace_runs/main_modal_aphasia_posters_gt_strict_visible_seed72_20260601",
    "seed75_text_band_masked": "outputs/emu3p5-main/modal_aphasia_poster_gt_text_band_masked_repeat/ume_trace_runs/main_modal_aphasia_posters_gt_text_band_masked_repeat_seed75_20260601",
    "seed76_text_band_masked_direct": "outputs/emu3p5-main/modal_aphasia_poster_gt_text_band_masked_direct_repeat/ume_trace_runs/main_modal_aphasia_posters_gt_text_band_masked_direct_repeat_seed76_20260601",
}

TRIVIAL_SPANS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "but",
    "he",
    "her",
    "his",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "they",
    "this",
    "to",
    "was",
    "were",
    "with",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", default="../research_logs/modal_aphasia_posters")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-per-sample", type=int, default=8)
    parser.add_argument(
        "--source-filter",
        default="",
        help="Optional source label to keep, e.g. seed72_strict_visible.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def poster_key(sample_id: str) -> str:
    match = re.match(r"(poster_\d+_[^_]+(?:_[^_]+)*?)(?:_rep\d+)?__", sample_id)
    if match:
        return match.group(1)
    return sample_id.split("__", 1)[0]


def condition(sample_id: str) -> str:
    if "__seed72_strict_visible_no_guess" in sample_id:
        return "strict_visible_no_guess"
    if "__seed72_visible_only" in sample_id:
        return "visible_only"
    if "__seed72_describe" in sample_id:
        return "describe"
    if "__greedy_visible_only" in sample_id:
        return "greedy_visible_only"
    if "__greedy_describe" in sample_id:
        return "greedy_describe"
    if "__visible_only" in sample_id:
        return "visible_only"
    if "__gt_describe" in sample_id:
        return "describe"
    if "__gt_imageonly" in sample_id:
        return "pure_image"
    if "__text_band_masked_direct" in sample_id:
        return "text_band_masked_direct"
    if "__text_band_masked" in sample_id:
        return "text_band_masked"
    return sample_id.rsplit("__", 1)[-1]


def dominant_semantic_category(row: dict) -> str:
    counts = []
    for item in row.get("category_counts", "").split(";"):
        if ":" not in item:
            continue
        key, value = item.rsplit(":", 1)
        if key in SEMANTIC_CATEGORIES:
            counts.append((int(float(value)), key))
    if not counts:
        return ""
    return sorted(counts, reverse=True)[0][1]


def is_trivial_span(text: str) -> bool:
    normalized = re.sub(r"[^A-Za-z0-9]+", " ", text).strip().lower()
    return normalized in TRIVIAL_SPANS


def load_manifest(log_root: Path) -> dict[str, dict]:
    path = log_root / "gt_posters_wikipedia_20260531/manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for row in data:
        slug = Path(row["local_path"]).stem
        out[slug] = row
    return out


def load_trace_tokens(repo_root: Path, source: str, sample_id: str) -> list[dict]:
    run_rel = RUN_DIRS.get(source)
    if not run_rel:
        return []
    path = repo_root / run_rel / "entropy_traces" / f"{sample_id}_entropy.jsonl"
    return [row for row in read_jsonl(path) if row.get("token_type") in {"text", "thinking"}]


def span_context(tokens: list[dict], start_step: int, end_step: int, radius: int = 8) -> str:
    if not tokens:
        return ""
    selected = []
    for row in tokens:
        step = int(row.get("step", -1))
        if start_step - radius <= step <= end_step + radius:
            selected.append(str(row.get("token_text", "")))
    return "".join(selected).strip()


def build_rows(log_root: Path, top_per_sample: int, source_filter: str = "") -> list[dict]:
    manifest = load_manifest(log_root)
    repo_root = log_root.parents[1] / "Emu3.5"
    sources = [
        (
            "seed54_sampled_describe",
            log_root / "seed54_gt_readback_repeat_top20/top_entropy_span_rows.csv",
        ),
        (
            "seed56_sampled_visible_only",
            log_root / "seed56_gt_visible_only_repeat_top20/top_entropy_span_rows.csv",
        ),
        (
            "seed57_greedy_describe_visible",
            log_root / "seed57_gt_describe_visible_greedy_top20/top_entropy_span_rows.csv",
        ),
        (
            "seed72_strict_visible",
            log_root / "seed72_gt_strict_visible_top20/top_entropy_span_rows.csv",
        ),
        (
            "seed75_text_band_masked",
            log_root / "seed75_gt_text_band_masked_repeat_top20/top_entropy_span_rows.csv",
        ),
        (
            "seed76_text_band_masked_direct",
            log_root / "seed76_gt_text_band_masked_direct_repeat_top20/top_entropy_span_rows.csv",
        ),
    ]
    rows: list[dict] = []
    for source, path in sources:
        if source_filter and source != source_filter:
            continue
        by_sample: dict[str, list[dict]] = {}
        for row in read_csv(path):
            semcat = dominant_semantic_category(row)
            if not semcat or is_trivial_span(row.get("span_text", "")):
                continue
            by_sample.setdefault(row["sample_id"], []).append({**row, "semantic_category": semcat})
        for sample_id, group in sorted(by_sample.items()):
            group = sorted(group, key=lambda r: float(r["mean_score"]), reverse=True)[:top_per_sample]
            key = poster_key(sample_id)
            poster = manifest.get(key, {})
            trace_tokens = load_trace_tokens(repo_root, source, sample_id)
            for rank, row in enumerate(group, start=1):
                start_step = int(float(row["start_step"]))
                end_step = int(float(row["end_step"]))
                rows.append(
                    {
                        "source": source,
                        "sample_id": sample_id,
                        "condition": condition(sample_id),
                        "poster_key": key,
                        "poster_name": poster.get("poster_name", ""),
                        "poster_image": poster.get("local_path", ""),
                        "rank_in_sample": rank,
                        "start_step": row["start_step"],
                        "end_step": row["end_step"],
                        "token_count": row["token_count"],
                        "mean_ume": row["mean_score"],
                        "max_ume": row["max_score"],
                        "dominant_category": row["dominant_category"],
                        "semantic_category": row["semantic_category"],
                        "category_counts": row["category_counts"],
                        "span_text": row["span_text"],
                        "context_window": span_context(trace_tokens, start_step, end_step),
                    }
                )
    return rows


def write_report(path: Path, rows: list[dict]) -> None:
    by_condition: dict[str, int] = {}
    by_semcat: dict[str, int] = {}
    for row in rows:
        by_condition[row["condition"]] = by_condition.get(row["condition"], 0) + 1
        by_semcat[row["semantic_category"]] = by_semcat.get(row["semantic_category"], 0) + 1
    lines = [
        "# Poster High-Entropy Span Audit Candidates",
        "",
        "Rows are semantic top-entropy spans extracted from real poster readback answers.",
        "",
        "## Counts",
        "",
    ]
    lines.append("- by condition: " + "; ".join(f"{k}:{v}" for k, v in sorted(by_condition.items())))
    lines.append("- by semantic category: " + "; ".join(f"{k}:{v}" for k, v in sorted(by_semcat.items())))
    lines.extend(["", "## Candidate Preview", ""])
    for row in rows[:80]:
        lines.append(
            f"- {row['source']} / {row['condition']} / {row['poster_key']} "
            f"rank={row['rank_in_sample']} {row['semantic_category']} "
            f"UME={float(row['mean_ume']):.4f}: `{row['span_text']}`"
        )
        if row.get("context_window"):
            lines.append(f"  context: `{row['context_window']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    log_root = Path(args.log_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    rows = build_rows(log_root, args.top_per_sample, args.source_filter)
    write_csv(out_dir / "poster_high_entropy_span_audit_candidates.csv", rows)
    write_report(out_dir / "poster_high_entropy_span_audit_candidates.md", rows)
    print(f"[INFO] wrote {len(rows)} candidate rows to {out_dir}")


if __name__ == "__main__":
    main()
