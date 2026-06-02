#!/usr/bin/env python3
"""Analyze G2U-style spatial prompt strategies from image-read entropy traces."""

from __future__ import annotations

import argparse
import csv
import importlib as imp
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from math import ceil
from statistics import mean


POSITIONS = ("upper-left", "upper-right", "lower-left", "lower-right")
POSITION_VARIANTS = {
    "upper-left": ("upper-left", "upper left", "top-left", "top left"),
    "upper-right": ("upper-right", "upper right", "top-right", "top right"),
    "lower-left": ("lower-left", "lower left", "bottom-left", "bottom left"),
    "lower-right": ("lower-right", "lower right", "bottom-right", "bottom right"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def load_cfg(path: str):
    cfg_name = Path(path).stem
    cfg_package = Path(path).parent.__str__().replace("/", ".")
    return imp.import_module(f".{cfg_name}", package=cfg_package)


def read_jsonl(path: Path) -> list[dict]:
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


def norm_text(text: str) -> str:
    text = re.sub(r"<\|[^>]+?\|>", " ", text)
    text = text.lower()
    return re.sub(r"[^a-z0-9-]+", " ", text)


def text_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9-]+", norm_text(text))


def infer_strategy(sample_id: str, prompt_meta: dict) -> str:
    if prompt_meta.get("strategy"):
        return prompt_meta["strategy"]
    return sample_id.rsplit("__", 1)[-1]


def token_text_and_spans(rows: list[dict]) -> tuple[str, list[dict]]:
    text = ""
    spans = []
    for idx, row in enumerate(rows):
        token = str(row.get("token_text", ""))
        start = len(text)
        text += token
        spans.append({"idx": idx, "start": start, "end": len(text), "row": row})
    return text, spans


def span_ume_for_phrase(rows: list[dict], phrase: str) -> tuple[float, int, int]:
    text, spans = token_text_and_spans(rows)
    start = text.lower().find(phrase.lower())
    if start < 0:
        return 0.0, -1, -1
    end = start + len(phrase)
    selected = [item for item in spans if item["start"] < end and item["end"] > start]
    if not selected:
        return 0.0, -1, -1
    scores = [float(item["row"].get("ume", 0.0) or 0.0) for item in selected]
    steps = [int(item["row"].get("step", item["idx"])) for item in selected]
    return mean(scores), min(steps), max(steps)


def top_entropy_regions(rows: list[dict], fraction: float = 0.2) -> tuple[list[dict], list[int]]:
    if not rows:
        return [], []
    k = max(1, ceil(len(rows) * fraction))
    ranked = sorted(
        enumerate(rows),
        key=lambda item: float(item[1].get("ume", 0.0) or 0.0),
        reverse=True,
    )[:k]
    top_indices = sorted(idx for idx, _ in ranked)
    regions = []
    start = prev = top_indices[0]
    for idx in top_indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        regions.append(summarize_token_region(rows, start, prev))
        start = prev = idx
    regions.append(summarize_token_region(rows, start, prev))
    return regions, top_indices


def summarize_token_region(rows: list[dict], start: int, end: int) -> dict:
    selected = rows[start : end + 1]
    scores = [float(row.get("ume", 0.0) or 0.0) for row in selected]
    text = "".join(str(row.get("token_text", "")) for row in selected)
    text = re.sub(r"\s+", " ", text).strip()
    return {
        "start_step": int(selected[0].get("step", start)),
        "end_step": int(selected[-1].get("step", end)),
        "token_count": len(selected),
        "mean_ume": mean(scores) if scores else 0.0,
        "max_ume": max(scores) if scores else 0.0,
        "text": text[:220],
    }


def format_top_regions(regions: list[dict], limit: int = 6) -> str:
    parts = []
    for region in regions[:limit]:
        text = region["text"] or "<blank>"
        parts.append(
            f"{region['start_step']}-{region['end_step']}:"
            f"{float(region['mean_ume']):.3f}:\"{text}\""
        )
    return " | ".join(parts)


def final_position(text: str) -> str:
    low = text.lower()
    final_matches = re.findall(
        r"position\s*[:\-]?\s*((?:upper|lower|top|bottom)[-\s]+(?:left|right))",
        low,
    )
    if final_matches:
        normalized = normalize_position_phrase(final_matches[-1])
        if normalized:
            return normalized
    matches = [pos for pos, variants in POSITION_VARIANTS.items() if any(v in low for v in variants)]
    return matches[-1] if matches else ""


def normalize_position_phrase(text: str) -> str:
    low = text.lower().strip().replace("_", "-")
    low = re.sub(r"\s+", " ", low)
    for pos, variants in POSITION_VARIANTS.items():
        if low in variants:
            return pos
    return ""


def position_mentioned(text: str, position: str) -> bool:
    low = text.lower()
    return any(variant in low for variant in POSITION_VARIANTS[position])


def behavior(text: str) -> str:
    words = text_words(text)
    if not words:
        return "empty"
    if len(words) <= 4 and any(pos in " ".join(words) for pos in POSITIONS):
        return "short_position"
    if "position" in words:
        return "final_position"
    if any(pos in " ".join(words) for pos in POSITIONS):
        return "mentions_position"
    return "other"


def analyze_sample(run_dir: Path, sample_id: str, meta: dict) -> dict:
    raw_path = run_dir / "raw_generations" / f"{sample_id}.txt"
    trace_path = run_dir / "entropy_traces" / f"{sample_id}_entropy.jsonl"
    text = raw_path.read_text(encoding="utf-8") if raw_path.exists() else ""
    rows = [r for r in read_jsonl(trace_path) if r.get("token_type") in {"text", "thinking"}]
    scores = [float(r.get("ume", 0.0) or 0.0) for r in rows]
    expected = meta.get("expected_position", "")
    predicted = final_position(text)
    wrong_positions = [pos for pos in POSITIONS if pos != expected and position_mentioned(text, pos)]
    phrase_for_span = ""
    for variant in POSITION_VARIANTS.get(predicted or expected, (predicted or expected,)):
        if variant in text.lower():
            phrase_for_span = variant
            break
    answer_ume, answer_start, answer_end = span_ume_for_phrase(rows, phrase_for_span or predicted or expected)
    regions, top_indices = top_entropy_regions(rows)
    top_scores = [float(rows[idx].get("ume", 0.0) or 0.0) for idx in top_indices]
    answer_indices = set(range(answer_start, answer_end + 1)) if answer_start >= 0 and answer_end >= 0 else set()
    answer_top_overlap = len(answer_indices.intersection(top_indices)) / len(answer_indices) if answer_indices else 0.0
    return {
        "sample_id": sample_id,
        "case_key": meta.get("case_key", ""),
        "base_case_key": meta.get("base_case_key", meta.get("case_key", "")),
        "transform": meta.get("transform", ""),
        "source_position": meta.get("source_position", ""),
        "repeat_id": meta.get("repeat_id", ""),
        "strategy": infer_strategy(sample_id, meta),
        "expected_object": meta.get("expected_object", ""),
        "expected_position": expected,
        "predicted_position": predicted,
        "position_hit": int(predicted == expected),
        "wrong_position_mentioned": int(any(wrong_positions)),
        "wrong_positions": ";".join(wrong_positions),
        "behavior": behavior(text),
        "trace_token_count": len(rows),
        "mean_ume": mean(scores) if scores else 0.0,
        "max_ume": max(scores) if scores else 0.0,
        "top20_token_count": len(top_indices),
        "top20_mean_ume": mean(top_scores) if top_scores else 0.0,
        "top20_regions": format_top_regions(regions),
        "answer_span_top20_overlap": answer_top_overlap,
        "answer_span_ume": answer_ume,
        "answer_start_step": answer_start,
        "answer_end_step": answer_end,
        "word_count": len(text_words(text)),
        "raw_text_preview": re.sub(r"\s+", " ", text)[:260],
    }


def aggregate(rows: list[dict], key: str) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    out = []
    for value, group in sorted(grouped.items()):
        hits = [r for r in group if int(r["position_hit"]) == 1]
        misses = [r for r in group if int(r["position_hit"]) == 0]
        behaviors = Counter(r["behavior"] for r in group)
        out.append(
            {
                key: value,
                "sample_count": len(group),
                "position_hits": len(hits),
                "position_accuracy": len(hits) / len(group) if group else 0.0,
                "mean_ume": mean(float(r["mean_ume"]) for r in group),
                "mean_ume_hit": mean(float(r["mean_ume"]) for r in hits) if hits else 0.0,
                "mean_ume_miss": mean(float(r["mean_ume"]) for r in misses) if misses else 0.0,
                "mean_answer_span_ume": mean(float(r["answer_span_ume"]) for r in group),
                "mean_top20_ume": mean(float(r["top20_mean_ume"]) for r in group),
                "mean_answer_top20_overlap": mean(float(r["answer_span_top20_overlap"]) for r in group),
                "wrong_position_mention_rate": mean(int(r["wrong_position_mentioned"]) for r in group),
                "mean_word_count": mean(float(r["word_count"]) for r in group),
                "behavior_counts": ";".join(f"{k}:{v}" for k, v in behaviors.most_common()),
            }
        )
    return out


def write_report(
    path: Path,
    strategy_rows: list[dict],
    case_rows: list[dict],
    sample_rows: list[dict],
    transform_rows: list[dict] | None = None,
) -> None:
    lines = [
        "# G2U Spatial Prompt Strategy Report",
        "",
        "This report compares answer-only, describe-then-answer, and grid-then-answer prompt strategies on the same simple spatial images.",
        "",
        "## Strategy Summary",
        "",
    ]
    for row in strategy_rows:
        lines.append(
            f"- {row['strategy']}: hits={row['position_hits']}/{row['sample_count']} "
            f"acc={float(row['position_accuracy']):.3f}, mean UME={float(row['mean_ume']):.4f}, "
            f"hit UME={float(row['mean_ume_hit']):.4f}, miss UME={float(row['mean_ume_miss']):.4f}, "
            f"answer-span UME={float(row['mean_answer_span_ume']):.4f}, wrong-pos-rate={float(row['wrong_position_mention_rate']):.3f}, "
            f"top20 UME={float(row['mean_top20_ume']):.4f}, answer-top20-overlap={float(row['mean_answer_top20_overlap']):.3f}, "
            f"words={float(row['mean_word_count']):.1f}, behavior={row['behavior_counts']}"
        )
    if transform_rows:
        lines.extend(["", "## Transform Summary", ""])
        for row in transform_rows:
            lines.append(
                f"- {row['transform']}: hits={row['position_hits']}/{row['sample_count']} "
                f"acc={float(row['position_accuracy']):.3f}, mean UME={float(row['mean_ume']):.4f}, "
                f"answer-span UME={float(row['mean_answer_span_ume']):.4f}, "
                f"top20 UME={float(row['mean_top20_ume']):.4f}, "
                f"wrong-pos-rate={float(row['wrong_position_mention_rate']):.3f}, "
                f"behavior={row['behavior_counts']}"
            )
    lines.extend(["", "## Case Summary", ""])
    for row in case_rows:
        lines.append(
            f"- {row['case_key']}: hits={row['position_hits']}/{row['sample_count']} "
            f"acc={float(row['position_accuracy']):.3f}, mean UME={float(row['mean_ume']):.4f}, "
            f"wrong-pos-rate={float(row['wrong_position_mention_rate']):.3f}"
        )
    misses = [row for row in sample_rows if int(row["position_hit"]) == 0]
    if misses:
        lines.extend(["", "## Miss Examples", ""])
        for row in misses:
            lines.append(
                f"- `{row['sample_id']}` expected={row['expected_position']} predicted={row['predicted_position']} "
                f"mean_UME={float(row['mean_ume']):.4f}, top20={row['top20_regions']}: {row['raw_text_preview']}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.cfg)
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    prompt_map = cfg.prompts if isinstance(cfg.prompts, dict) else {}
    rows = []
    for sample_id, meta in sorted(prompt_map.items()):
        trace_path = run_dir / "entropy_traces" / f"{sample_id}_entropy.jsonl"
        if trace_path.exists():
            rows.append(analyze_sample(run_dir, sample_id, meta))
    strategy_rows = aggregate(rows, "strategy")
    case_rows = aggregate(rows, "case_key")
    repeat_rows = aggregate(rows, "repeat_id") if rows and "repeat_id" in rows[0] else []
    transform_rows = aggregate(rows, "transform") if rows and any(row.get("transform") for row in rows) else []
    write_csv(out_dir / "g2u_spatial_sample_summary.csv", rows)
    write_csv(out_dir / "g2u_spatial_strategy_summary.csv", strategy_rows)
    write_csv(out_dir / "g2u_spatial_case_summary.csv", case_rows)
    write_csv(out_dir / "g2u_spatial_repeat_summary.csv", repeat_rows)
    write_csv(out_dir / "g2u_spatial_transform_summary.csv", transform_rows)
    write_report(out_dir / "g2u_spatial_prompt_strategy_report.md", strategy_rows, case_rows, rows, transform_rows)
    print(f"[INFO] wrote G2U spatial summaries for {len(rows)} samples to {out_dir}")


if __name__ == "__main__":
    main()
