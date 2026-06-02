#!/usr/bin/env python3
"""Compare real-poster readback input forms and their top-entropy regions."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev


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


def fnum(value: object, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def clean_generated(text: str) -> str:
    text = re.sub(r"<\|[^>]+?\|>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9'-]+", clean_generated(text))


def behavior(text: str) -> str:
    cleaned = clean_generated(text)
    word_count = len(words(text))
    low = cleaned.lower()
    if not cleaned:
        return "immediate_end"
    if (
        "to finish the task" in low
        or re.search(r"\bstep\s+[0-9]+", low)
        or low.startswith("to create ")
        or "design process" in low
    ):
        return "action_script"
    if word_count <= 8 and ("answer is" in low or word_count <= 4):
        return "short_label"
    if word_count >= 30 and any(key in low for key in ("poster", "image", "film", "movie")):
        return "description"
    if word_count >= 30:
        return "accidental_description"
    return "fragment_or_other"


def input_form(sample_id: str) -> str:
    if "__gt_describe" in sample_id or "__greedy_gt_describe" in sample_id:
        return "image_plus_text"
    if "__gt_imageonly" in sample_id or "__greedy_gt_imageonly" in sample_id:
        return "pure_image"
    return "unknown"


def poster_id_from_sample(sample_id: str) -> str:
    match = re.search(r"poster_(\d+)_", sample_id)
    return str(int(match.group(1))) if match else ""


def parse_counts(text: str) -> Counter:
    out: Counter[str] = Counter()
    for part in (text or "").split(";"):
        if not part or ":" not in part:
            continue
        key, value = part.rsplit(":", 1)
        out[key] += int(fnum(value))
    return out


def counts_to_text(counts: Counter) -> str:
    return ";".join(f"{k}:{v}" for k, v in counts.most_common())


def summarize_trace(trace_path: Path) -> tuple[int, float, float]:
    rows = [row for row in read_jsonl(trace_path) if row.get("token_type") in {"text", "thinking"}]
    scores = [fnum(row.get("ume")) for row in rows]
    return len(rows), mean(scores) if scores else 0.0, max(scores) if scores else 0.0


def build_sample_rows(run_dir: Path, top20_dir: Path, manifest_path: Path) -> list[dict]:
    manifest = {str(row["poster_id"]): row for row in json.loads(manifest_path.read_text(encoding="utf-8"))}
    top_summary = {row["sample_id"]: row for row in read_csv(top20_dir / "top_entropy_sample_summary.csv")}
    raw_dir = run_dir / "raw_generations"
    trace_dir = run_dir / "entropy_traces"
    rows = []
    for raw_path in sorted(raw_dir.glob("*.txt")):
        sample_id = raw_path.stem
        poster_id = poster_id_from_sample(sample_id)
        text = raw_path.read_text(encoding="utf-8", errors="replace")
        cleaned = clean_generated(text)
        trace_count, mean_ume, max_ume = summarize_trace(trace_dir / f"{sample_id}_entropy.jsonl")
        top = top_summary.get(sample_id, {})
        row = {
            "sample_id": sample_id,
            "poster_id": poster_id,
            "poster_name": manifest.get(poster_id, {}).get("poster_name", ""),
            "input_form": input_form(sample_id),
            "behavior": behavior(text),
            "valid_description": int(behavior(text) == "description"),
            "word_count": len(words(text)),
            "trace_token_count": trace_count,
            "mean_ume": mean_ume,
            "max_ume": max_ume,
            "top20_threshold": top.get("text_top_threshold", ""),
            "top20_token_count": top.get("text_top_token_count", ""),
            "top20_span_count": top.get("text_top_span_count", ""),
            "top20_category_counts": top.get("top_category_counts", ""),
            "top20_span_preview": top.get("top_span_preview", ""),
            "raw_text_preview": cleaned[:500],
        }
        rows.append(row)
    return rows


def build_condition_rows(sample_rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in sample_rows:
        grouped[row["input_form"]].append(row)
    rows = []
    for form, group in sorted(grouped.items()):
        behaviors = Counter(row["behavior"] for row in group)
        categories: Counter[str] = Counter()
        for row in group:
            categories.update(parse_counts(row["top20_category_counts"]))
        valid = sum(int(row["valid_description"]) for row in group)
        rows.append(
            {
                "input_form": form,
                "sample_count": len(group),
                "valid_description_count": valid,
                "valid_description_rate": valid / len(group) if group else 0.0,
                "behavior_counts": counts_to_text(behaviors),
                "mean_word_count": mean(float(row["word_count"]) for row in group),
                "mean_trace_token_count": mean(float(row["trace_token_count"]) for row in group),
                "mean_ume": mean(float(row["mean_ume"]) for row in group),
                "sd_ume": pstdev(float(row["mean_ume"]) for row in group) if len(group) > 1 else 0.0,
                "aggregate_top20_categories": counts_to_text(categories),
            }
        )
    return rows


def build_pair_rows(sample_rows: list[dict]) -> list[dict]:
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in sample_rows:
        grouped[row["poster_id"]][row["input_form"]] = row
    rows = []
    for poster_id, forms in sorted(grouped.items(), key=lambda item: int(item[0] or 0)):
        describe = forms.get("image_plus_text", {})
        pure = forms.get("pure_image", {})
        rows.append(
            {
                "poster_id": poster_id,
                "poster_name": describe.get("poster_name") or pure.get("poster_name", ""),
                "describe_behavior": describe.get("behavior", ""),
                "pure_image_behavior": pure.get("behavior", ""),
                "describe_mean_ume": describe.get("mean_ume", ""),
                "pure_image_mean_ume": pure.get("mean_ume", ""),
                "describe_top20_preview": describe.get("top20_span_preview", ""),
                "pure_image_top20_preview": pure.get("top20_span_preview", ""),
                "describe_preview": describe.get("raw_text_preview", ""),
                "pure_image_preview": pure.get("raw_text_preview", ""),
            }
        )
    return rows


def write_report(path: Path, condition_rows: list[dict], pair_rows: list[dict]) -> None:
    lines = [
        "# GT Poster Input-Form Top20 Audit",
        "",
        "This report compares the same real poster images under image+instruction and pure-image inputs.",
        "",
        "## Conditions",
        "",
    ]
    for row in condition_rows:
        lines.append(
            f"- {row['input_form']}: n={row['sample_count']}, valid descriptions="
            f"{row['valid_description_count']}/{row['sample_count']} ({float(row['valid_description_rate']):.3f}), "
            f"behavior={row['behavior_counts']}, mean words={float(row['mean_word_count']):.1f}, "
            f"mean UME={float(row['mean_ume']):.4f}"
        )
        if row["aggregate_top20_categories"]:
            lines.append(f"  top20 categories: {row['aggregate_top20_categories']}")
    lines.extend(["", "## Poster Pairs", ""])
    for row in pair_rows:
        lines.append(
            f"- poster {row['poster_id']} {row['poster_name']}: "
            f"describe={row['describe_behavior']} UME={fnum(row['describe_mean_ume']):.4f}; "
            f"pure={row['pure_image_behavior']} UME={fnum(row['pure_image_mean_ume']):.4f}"
        )
        lines.append(f"  describe top20: {row['describe_top20_preview'][:260]}")
        lines.append(f"  pure top20: {row['pure_image_top20_preview'][:260]}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    top20_dir = Path(args.top20_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_rows = build_sample_rows(run_dir, top20_dir, Path(args.manifest))
    condition_rows = build_condition_rows(sample_rows)
    pair_rows = build_pair_rows(sample_rows)

    write_csv(out_dir / "gt_input_form_sample_summary.csv", sample_rows)
    write_csv(out_dir / "gt_input_form_condition_summary.csv", condition_rows)
    write_csv(out_dir / "gt_input_form_pair_summary.csv", pair_rows)
    write_report(out_dir / "gt_input_form_top20_report.md", condition_rows, pair_rows)
    print(f"[INFO] wrote GT input-form top20 audit for {len(sample_rows)} samples to {out_dir}")


if __name__ == "__main__":
    main()
