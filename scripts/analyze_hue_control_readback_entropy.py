#!/usr/bin/env python3
"""Audit GT hue-control image readback answers against top-entropy text regions."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


COLOR_WORDS = {
    "red",
    "blue",
    "green",
    "purple",
    "violet",
    "yellow",
    "cyan",
    "teal",
    "orange",
    "magenta",
    "pink",
    "black",
    "white",
    "gray",
    "grey",
}
SHAPE_WORDS = {
    "circle",
    "circular",
    "square",
    "triangle",
    "triangular",
    "diamond",
    "rhombus",
    "quadrilateral",
    "shape",
    "icon",
}
COLOR_SYNONYMS = {
    "cyan": {"cyan", "teal"},
    "magenta": {"magenta", "pink"},
    "purple": {"purple", "violet"},
}
SHAPE_SYNONYMS = {
    "circle": {"circle", "circular"},
    "triangle": {"triangle", "triangular"},
    "diamond": {"diamond", "rhombus", "quadrilateral"},
}
GENERIC_SHAPES = {"shape", "icon"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--top20-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
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
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def clean_text(text: str) -> str:
    text = re.sub(r"<\|[^>]+?\|>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", clean_text(text).lower())


def fnum(value: object, default: float = 0.0) -> float:
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def input_form(sample_id: str) -> str:
    if sample_id.endswith("__describe"):
        return "image_plus_text"
    if sample_id.endswith("__imageonly"):
        return "pure_image"
    return "unknown"


def image_id(sample_id: str) -> str:
    return sample_id.replace("__describe", "").replace("__imageonly", "")


def behavior(text: str) -> str:
    cleaned = clean_text(text)
    wc = len(words(text))
    if not cleaned:
        return "immediate_end"
    low = cleaned.lower()
    if "to finish the task" in low or re.search(r"\bstep\s+[0-9]+", low) or low.startswith("to create "):
        return "action_script"
    if wc <= 6:
        return "short_or_fragment"
    if wc >= 12:
        return "description"
    return "fragment_or_other"


def trace_rows(run_dir: Path, sample_id: str) -> list[dict]:
    return [r for r in read_jsonl(run_dir / "entropy_traces" / f"{sample_id}_entropy.jsonl") if r.get("token_type") in {"text", "thinking"}]


def top_token_words(top_rows: list[dict]) -> list[str]:
    out: list[str] = []
    for row in top_rows:
        out.extend(words(str(row.get("token_text", ""))))
    return out


def color_terms(color: str) -> set[str]:
    return {color} | COLOR_SYNONYMS.get(color, set())


def shape_terms(shape: str) -> set[str]:
    return {shape} | SHAPE_SYNONYMS.get(shape, set())


def summarize_sample(run_dir: Path, sample_id: str, meta: dict, top_tokens: list[dict], top_spans: list[dict]) -> dict:
    raw_path = run_dir / "raw_generations" / f"{sample_id}.txt"
    raw = raw_path.read_text(encoding="utf-8", errors="replace") if raw_path.exists() else ""
    all_words = set(words(raw))
    top_words = set(top_token_words(top_tokens))
    expected_color = meta["fill_color"]
    expected_shape = meta["shape"]
    allowed_colors = (set(str(meta["allowed_colors"]).split(";")) | color_terms(expected_color)) - {""}
    color_words = all_words & COLOR_WORDS
    hallucinated_colors = sorted((color_words - allowed_colors) - {"grey"})
    expected_shape_terms = shape_terms(expected_shape)
    shape_words = all_words & SHAPE_WORDS
    hallucinated_shapes = sorted(shape_words - expected_shape_terms - GENERIC_SHAPES)
    rows = trace_rows(run_dir, sample_id)
    scores = [fnum(r.get("ume")) for r in rows]
    expected_color_top = int(bool(color_terms(expected_color) & top_words))
    expected_shape_top = int(bool(expected_shape_terms & top_words))
    hallucinated_top = sorted(set(hallucinated_colors) & top_words)
    hallucinated_shape_top = sorted(set(hallucinated_shapes) & top_words)
    color_top = sorted(top_words & COLOR_WORDS)
    shape_top = sorted(top_words & SHAPE_WORDS)
    return {
        "sample_id": sample_id,
        "image_id": image_id(sample_id),
        "pair_id": meta["pair_id"],
        "variant": meta["variant"],
        "input_form": input_form(sample_id),
        "expected_color": expected_color,
        "expected_shape": expected_shape,
        "behavior": behavior(raw),
        "word_count": len(words(raw)),
        "text_token_count": len(rows),
        "mean_ume": mean(scores) if scores else 0.0,
        "max_ume": max(scores) if scores else 0.0,
        "expected_color_mentioned": int(bool(color_terms(expected_color) & all_words)),
        "expected_shape_mentioned": int(bool(expected_shape_terms & all_words)),
        "hallucinated_colors": ";".join(hallucinated_colors),
        "hallucinated_color_count": len(hallucinated_colors),
        "hallucinated_shapes": ";".join(hallucinated_shapes),
        "hallucinated_shape_count": len(hallucinated_shapes),
        "expected_color_in_top20": expected_color_top,
        "expected_shape_in_top20": expected_shape_top,
        "hallucinated_color_in_top20": ";".join(hallucinated_top),
        "hallucinated_color_top20_count": len(hallucinated_top),
        "hallucinated_shape_in_top20": ";".join(hallucinated_shape_top),
        "hallucinated_shape_top20_count": len(hallucinated_shape_top),
        "top20_color_words": ";".join(color_top),
        "top20_shape_words": ";".join(shape_top),
        "top20_span_preview": " | ".join(row.get("span_text", "") for row in top_spans[:3]),
        "raw_text": clean_text(raw).replace("\n", "\\n"),
    }


def aggregate(rows: list[dict], key: str) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    out = []
    for value, group in sorted(grouped.items()):
        behaviors = Counter(row["behavior"] for row in group)
        out.append(
            {
                key: value,
                "sample_count": len(group),
                "description_count": sum(row["behavior"] == "description" for row in group),
                "mean_word_count": mean(float(row["word_count"]) for row in group),
                "mean_ume": mean(float(row["mean_ume"]) for row in group),
                "expected_color_mention_rate": mean(int(row["expected_color_mentioned"]) for row in group),
                "expected_shape_mention_rate": mean(int(row["expected_shape_mentioned"]) for row in group),
                "hallucinated_color_sample_rate": mean(int(row["hallucinated_color_count"]) > 0 for row in group),
                "hallucinated_shape_sample_rate": mean(int(row["hallucinated_shape_count"]) > 0 for row in group),
                "expected_color_top20_rate": mean(int(row["expected_color_in_top20"]) for row in group),
                "expected_shape_top20_rate": mean(int(row["expected_shape_in_top20"]) for row in group),
                "hallucinated_color_top20_rate": mean(int(row["hallucinated_color_top20_count"]) > 0 for row in group),
                "hallucinated_shape_top20_rate": mean(int(row["hallucinated_shape_top20_count"]) > 0 for row in group),
                "behavior_counts": ";".join(f"{k}:{v}" for k, v in behaviors.most_common()),
            }
        )
    return out


def write_report(path: Path, condition_rows: list[dict], sample_rows: list[dict]) -> None:
    lines = [
        "# GT Hue-Control Readback Entropy Audit",
        "",
        "This audit compares matched GT hue-control images under image+instruction and pure-image inputs.",
        "",
        "## Conditions",
        "",
    ]
    for row in condition_rows:
        lines.append(
            f"- {row['input_form']}: n={row['sample_count']}, descriptions={row['description_count']}, "
            f"mean UME={float(row['mean_ume']):.4f}, color mention={float(row['expected_color_mention_rate']):.3f}, "
            f"shape mention={float(row['expected_shape_mention_rate']):.3f}, hallucinated-color sample rate="
            f"{float(row['hallucinated_color_sample_rate']):.3f}, hallucinated-shape sample rate="
            f"{float(row['hallucinated_shape_sample_rate']):.3f}, color-in-top20={float(row['expected_color_top20_rate']):.3f}, "
            f"shape-in-top20={float(row['expected_shape_top20_rate']):.3f}, hallucinated-color-in-top20="
            f"{float(row['hallucinated_color_top20_rate']):.3f}, hallucinated-shape-in-top20="
            f"{float(row['hallucinated_shape_top20_rate']):.3f}, behaviors={row['behavior_counts']}"
        )
    lines.extend(["", "## Samples", ""])
    for row in sample_rows:
        lines.append(
            f"- `{row['sample_id']}` expected={row['expected_color']} {row['expected_shape']}, "
            f"behavior={row['behavior']}, UME={float(row['mean_ume']):.4f}, "
            f"color_ok={row['expected_color_mentioned']}, shape_ok={row['expected_shape_mentioned']}, "
            f"hallucinated_colors={row['hallucinated_colors'] or 'none'}, "
            f"hallucinated_shapes={row['hallucinated_shapes'] or 'none'}"
        )
        lines.append(f"  top20: {row['top20_span_preview'][:260]}")
        lines.append(f"  raw: {row['raw_text'][:260]}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    top20_dir = Path(args.top20_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {row["image_id"]: row for row in json.loads(Path(args.manifest).read_text(encoding="utf-8"))}
    top_tokens_by_sample: dict[str, list[dict]] = defaultdict(list)
    for row in read_csv(top20_dir / "top_entropy_token_rows.csv"):
        top_tokens_by_sample[row["sample_id"]].append(row)
    top_spans_by_sample: dict[str, list[dict]] = defaultdict(list)
    for row in read_csv(top20_dir / "top_entropy_span_rows.csv"):
        top_spans_by_sample[row["sample_id"]].append(row)
    sample_rows = []
    for raw_path in sorted((run_dir / "raw_generations").glob("*.txt")):
        sample_id = raw_path.stem
        meta = manifest.get(image_id(sample_id))
        if not meta:
            continue
        sample_rows.append(
            summarize_sample(run_dir, sample_id, meta, top_tokens_by_sample[sample_id], top_spans_by_sample[sample_id])
        )
    condition_rows = aggregate(sample_rows, "input_form")
    pair_rows = aggregate(sample_rows, "pair_id")
    write_csv(out_dir / "hue_control_readback_sample_summary.csv", sample_rows)
    write_csv(out_dir / "hue_control_readback_condition_summary.csv", condition_rows)
    write_csv(out_dir / "hue_control_readback_pair_summary.csv", pair_rows)
    write_report(out_dir / "hue_control_readback_entropy_report.md", condition_rows, sample_rows)
    print(f"[INFO] wrote hue-control readback audit for {len(sample_rows)} samples to {out_dir}")


if __name__ == "__main__":
    main()
