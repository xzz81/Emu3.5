#!/usr/bin/env python3
"""Locate top-quantile entropy regions in generated text and image tokens."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path


COLORS = {"red", "blue", "green", "yellow", "orange", "purple", "black", "white", "gray", "grey", "brown", "gold", "silver"}
SPATIAL = {"left", "right", "top", "bottom", "center", "centre", "middle", "foreground", "background", "behind", "above", "below"}
HEDGES = {"appears", "seems", "likely", "might", "possibly", "perhaps", "suggests", "unclear", "looks"}
ACTION = {"finish", "task", "step", "move", "press", "release", "grasp", "lift", "place", "align", "done", "approach"}
OCR_TEXT = {"title", "text", "letters", "font", "tagline", "credits", "readable", "word", "words", "caption"}
POSTER_OBJECTS = {
    "poster",
    "character",
    "characters",
    "figure",
    "figures",
    "face",
    "faces",
    "man",
    "woman",
    "people",
    "helmet",
    "car",
    "city",
    "sky",
    "background",
    "foreground",
}
STOPWORDS = {"a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "with"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--quantile", type=float, default=0.8)
    parser.add_argument("--score-field", choices=["auto", "ume", "ume_full"], default="auto")
    parser.add_argument("--max-spans-per-sample", type=int, default=12)
    return parser.parse_args()


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


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = (len(values) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - idx) + values[hi] * (idx - lo)


def score(row: dict, score_field: str) -> float:
    if score_field == "ume_full":
        return float(row.get("ume_full", row.get("ume", 0.0)) or 0.0)
    if score_field == "ume":
        return float(row.get("ume", 0.0) or 0.0)
    if row.get("token_type") == "visual":
        return float(row.get("ume_full", row.get("ume", 0.0)) or 0.0)
    return float(row.get("ume", 0.0) or 0.0)


def clean_words(text: str) -> list[str]:
    text = re.sub(r"<\|[^>]+?\|>", " ", text.lower())
    return re.findall(r"[a-z0-9]+", text)


def text_category(token: str) -> str:
    words = set(clean_words(token))
    if not words:
        if re.search(r"[.!?,;:\"'()\[\]-]", token):
            return "boundary"
        return "space_or_special"
    if words & ACTION:
        return "action_script"
    if words & HEDGES:
        return "hedge_or_uncertainty"
    if words & OCR_TEXT:
        return "poster_text_or_ocr"
    if words & COLORS:
        return "color_attribute"
    if words & SPATIAL:
        return "spatial_composition"
    if words & POSTER_OBJECTS:
        return "object_or_character"
    if any(w[:1].isupper() for w in re.findall(r"[A-Za-z]+", token)):
        return "proper_name_or_title"
    content = {w for w in words if w not in STOPWORDS}
    return "content_other" if content else "function_word"


def display_text(tokens: list[str]) -> str:
    text = "".join(tokens)
    text = re.sub(r"<\|[^>]+?\|>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:320]


def contiguous_spans(top_rows: list[dict]) -> list[list[dict]]:
    spans = []
    current = []
    prev_step = None
    for row in sorted(top_rows, key=lambda r: int(r["step"])):
        step = int(row["step"])
        if current and prev_step is not None and step != prev_step + 1:
            spans.append(current)
            current = []
        current.append(row)
        prev_step = step
    if current:
        spans.append(current)
    return spans


def image_position(idx: int, count: int) -> dict:
    width = int(round(math.sqrt(count)))
    if width * width != count:
        width = max(1, width)
    x = idx % width
    y = idx // width
    half = width / 2
    quadrant = ("top" if y < half else "bottom") + "_" + ("left" if x < half else "right")
    margin = max(1, int(width * 0.2))
    if x < margin or y < margin or x >= width - margin or y >= width - margin:
        zone = "edge"
    elif abs(x - half) <= margin and abs(y - half) <= margin:
        zone = "center"
    else:
        zone = "midfield"
    return {"visual_index": idx, "x": x, "y": y, "width": width, "quadrant": quadrant, "zone": zone}


def analyze_trace(path: Path, out_dir: Path, quantile: float, score_field: str, max_spans: int):
    sample_id = path.name.replace("_entropy.jsonl", "")
    records = read_jsonl(path)
    text_rows = [r for r in records if r.get("token_type") in {"text", "thinking"}]
    visual_rows = [r for r in records if r.get("token_type") == "visual"]
    outputs = {
        "token_rows": [],
        "span_rows": [],
        "summary_rows": [],
        "visual_rows": [],
        "visual_summary_rows": [],
    }

    if text_rows:
        values = [score(r, score_field) for r in text_rows]
        threshold = percentile(values, quantile)
        top = []
        for r in text_rows:
            s = score(r, score_field)
            if s >= threshold:
                cat = text_category(str(r.get("token_text", "")))
                row = {
                    "sample_id": sample_id,
                    "step": r.get("step"),
                    "token_type": r.get("token_type"),
                    "token_text": r.get("token_text", ""),
                    "score": s,
                    "ume": r.get("ume", ""),
                    "ume_full": r.get("ume_full", ""),
                    "u_tok": r.get("u_tok", ""),
                    "u_intra": r.get("u_intra", ""),
                    "category": cat,
                }
                top.append(row)
                outputs["token_rows"].append(row)
        categories = Counter(r["category"] for r in top)
        spans = contiguous_spans(top)
        span_rows = []
        for idx, span in enumerate(spans):
            cats = Counter(r["category"] for r in span)
            span_rows.append(
                {
                    "sample_id": sample_id,
                    "span_id": idx,
                    "start_step": span[0]["step"],
                    "end_step": span[-1]["step"],
                    "token_count": len(span),
                    "mean_score": sum(float(r["score"]) for r in span) / len(span),
                    "max_score": max(float(r["score"]) for r in span),
                    "dominant_category": cats.most_common(1)[0][0],
                    "category_counts": ";".join(f"{k}:{v}" for k, v in cats.most_common()),
                    "span_text": display_text([str(r["token_text"]) for r in span]),
                }
            )
        span_rows = sorted(span_rows, key=lambda r: (float(r["max_score"]), int(r["token_count"])), reverse=True)
        outputs["span_rows"].extend(span_rows[:max_spans])
        outputs["summary_rows"].append(
            {
                "sample_id": sample_id,
                "text_token_count": len(text_rows),
                "text_top_threshold": threshold,
                "text_top_token_count": len(top),
                "text_top_fraction": len(top) / len(text_rows),
                "text_top_span_count": len(spans),
                "top_category_counts": ";".join(f"{k}:{v}" for k, v in categories.most_common()),
                "top_span_preview": " | ".join(r["span_text"] for r in span_rows[:3]),
            }
        )

    if visual_rows:
        values = [score(r, score_field) for r in visual_rows]
        threshold = percentile(values, quantile)
        top = []
        for idx, r in enumerate(visual_rows):
            s = score(r, score_field)
            if s >= threshold:
                pos = image_position(idx, len(visual_rows))
                row = {
                    "sample_id": sample_id,
                    "step": r.get("step"),
                    "score": s,
                    "ume": r.get("ume", ""),
                    "ume_full": r.get("ume_full", ""),
                    "u_tok": r.get("u_tok", ""),
                    "u_intra": r.get("u_intra", ""),
                    "u_cfg": r.get("u_cfg", ""),
                    **pos,
                }
                top.append(row)
                outputs["visual_rows"].append(row)
        quadrants = Counter(r["quadrant"] for r in top)
        zones = Counter(r["zone"] for r in top)
        outputs["visual_summary_rows"].append(
            {
                "sample_id": sample_id,
                "visual_token_count": len(visual_rows),
                "visual_top_threshold": threshold,
                "visual_top_token_count": len(top),
                "visual_top_fraction": len(top) / len(visual_rows),
                "quadrant_counts": ";".join(f"{k}:{v}" for k, v in quadrants.most_common()),
                "zone_counts": ";".join(f"{k}:{v}" for k, v in zones.most_common()),
                "mean_top_x": sum(float(r["x"]) for r in top) / len(top) if top else "",
                "mean_top_y": sum(float(r["y"]) for r in top) / len(top) if top else "",
            }
        )
    return outputs


def write_report(path: Path, summaries: list[dict], spans: list[dict], visual_summaries: list[dict]) -> None:
    lines = [
        "# Top Entropy Region Report",
        "",
        "This report locates the top-quantile entropy regions inside generated text and visual tokens.",
        "",
    ]
    if summaries:
        lines.extend(["## Text Spans", ""])
        for row in summaries:
            lines.append(f"### {row['sample_id']}")
            lines.append(f"- threshold: {float(row['text_top_threshold']):.4f}")
            lines.append(f"- category counts: {row['top_category_counts']}")
            lines.append(f"- preview: {row['top_span_preview']}")
            lines.append("")
    if visual_summaries:
        lines.extend(["## Visual Regions", ""])
        for row in visual_summaries:
            lines.append(f"### {row['sample_id']}")
            lines.append(f"- threshold: {float(row['visual_top_threshold']):.4f}")
            lines.append(f"- quadrants: {row['quadrant_counts']}")
            lines.append(f"- zones: {row['zone_counts']}")
            lines.append("")
    if spans:
        lines.extend(["## Highest Text Spans", ""])
        for row in sorted(spans, key=lambda r: float(r["max_score"]), reverse=True)[:60]:
            lines.append(
                f"- `{row['sample_id']}` {row['start_step']}-{row['end_step']} "
                f"{row['dominant_category']} score={float(row['max_score']):.4f}: `{row['span_text']}`"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    all_tokens = []
    all_spans = []
    all_summaries = []
    all_visual = []
    all_visual_summaries = []
    for path in sorted((run_dir / "entropy_traces").glob("*_entropy.jsonl")):
        result = analyze_trace(path, out_dir, args.quantile, args.score_field, args.max_spans_per_sample)
        all_tokens.extend(result["token_rows"])
        all_spans.extend(result["span_rows"])
        all_summaries.extend(result["summary_rows"])
        all_visual.extend(result["visual_rows"])
        all_visual_summaries.extend(result["visual_summary_rows"])
    write_csv(out_dir / "top_entropy_token_rows.csv", all_tokens)
    write_csv(out_dir / "top_entropy_span_rows.csv", all_spans)
    write_csv(out_dir / "top_entropy_sample_summary.csv", all_summaries)
    write_csv(out_dir / "top_entropy_visual_token_rows.csv", all_visual)
    write_csv(out_dir / "top_entropy_visual_summary.csv", all_visual_summaries)
    write_report(out_dir / "top_entropy_region_report.md", all_summaries, all_spans, all_visual_summaries)
    print(
        f"[INFO] wrote {len(all_tokens)} text tokens, {len(all_spans)} spans, "
        f"{len(all_visual)} visual tokens to {out_dir}"
    )


if __name__ == "__main__":
    main()
