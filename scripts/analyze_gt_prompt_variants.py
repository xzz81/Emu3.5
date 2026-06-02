#!/usr/bin/env python3
"""Analyze prompt-anchor variants for real-poster GT readback."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


OUTSIDE_KNOWLEDGE = {
    "released",
    "release",
    "director",
    "directed",
    "starring",
    "installment",
    "sequel",
    "trilogy",
    "franchise",
    "original",
    "theatrical",
    "christopher",
    "nolan",
    "zemeckis",
    "spielberg",
    "leonardo",
    "dicaprio",
    "keanu",
    "reeves",
    "michael",
    "fox",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--top20-dir", required=True)
    parser.add_argument("--out-dir", required=True)
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


def words(text: str) -> list[str]:
    text = re.sub(r"<\|[^>]+?\|>", " ", text.lower())
    return re.findall(r"[a-z0-9]+", text)


def poster_id(sample_id: str) -> str:
    match = re.match(r"poster_(\d+)_", sample_id)
    return str(int(match.group(1))) if match else sample_id


def variant(sample_id: str) -> str:
    return sample_id.rsplit("__", 1)[-1]


def classify_behavior(text: str, prompt_variant: str) -> str:
    cleaned = re.sub(r"<\|[^>]+?\|>", " ", text).strip()
    lower = cleaned.lower()
    wc = len(cleaned.split())
    normalized_variant = re.sub(r"^(?:greedy_|seed\d+_)+", "", prompt_variant)
    if not cleaned:
        return "immediate_end"
    if "step 1" in lower or "finish the task" in lower or "to create" in lower or "to generate" in lower:
        return "action_script"
    if wc <= 18 and ("answer is" in lower or wc <= 4):
        return "short_label"
    if wc >= 45 and ("poster" in lower or "image" in lower):
        if normalized_variant in {"describe", "visible_only", "strict_visible_no_guess", "caption", "whatin"}:
            return "description"
        return "accidental_description"
    return "fragment_or_other"


def trace_summary(path: Path) -> dict:
    records = [r for r in read_jsonl(path) if r.get("token_type") in {"text", "thinking"}]
    umes = [float(r.get("ume", 0.0) or 0.0) for r in records]
    return {
        "trace_token_count": len(records),
        "mean_ume": mean(umes) if umes else 0.0,
        "max_ume": max(umes) if umes else 0.0,
    }


def load_top20(top20_dir: Path) -> dict[str, dict]:
    path = top20_dir / "top_entropy_sample_summary.csv"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        return {row["sample_id"]: row for row in csv.DictReader(f)}


def parse_counts(text: str) -> Counter:
    counts: Counter[str] = Counter()
    for item in (text or "").split(";"):
        if ":" not in item:
            continue
        key, value = item.rsplit(":", 1)
        try:
            counts[key] += int(value)
        except ValueError:
            pass
    return counts


def write_report(path: Path, sample_rows: list[dict], variant_rows: list[dict]) -> None:
    lines = [
        "# GT Prompt Variant Report",
        "",
        "This report compares prompt anchors for real-poster GT image readback.",
        "",
        "## Variant Summary",
        "",
    ]
    for row in sorted(variant_rows, key=lambda r: r["prompt_variant"]):
        lines.append(f"### {row['prompt_variant']}")
        lines.append(f"- samples: {row['sample_count']}")
        lines.append(f"- behavior counts: {row['behavior_counts']}")
        lines.append(f"- mean words: {float(row['mean_word_count']):.2f}")
        lines.append(f"- mean trace tokens: {float(row['mean_trace_tokens']):.2f}")
        lines.append(f"- mean UME: {float(row['mean_ume']):.4f}")
        lines.append(f"- outside-knowledge hit rate: {float(row['outside_knowledge_hit_rate']):.3f}")
        lines.append(f"- aggregate top categories: {row['aggregate_top_categories']}")
        lines.append("")

    lines.extend(["## Notable Raw Behaviors", ""])
    for row in sample_rows:
        if row["behavior"] != "description" or int(row["outside_knowledge_hits"]) > 0:
            lines.append(
                f"- `{row['sample_id']}` {row['behavior']} words={row['word_count']} "
                f"outside={row['outside_knowledge_hits']}: {row['raw_text_preview']}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    top20 = load_top20(Path(args.top20_dir))
    sample_rows = []
    by_variant: dict[str, list[dict]] = defaultdict(list)
    for trace_path in sorted((run_dir / "entropy_traces").glob("*_entropy.jsonl")):
        sample_id = trace_path.name.replace("_entropy.jsonl", "")
        raw_path = run_dir / "raw_generations" / f"{sample_id}.txt"
        raw_text = raw_path.read_text(encoding="utf-8") if raw_path.exists() else ""
        prompt_variant = variant(sample_id)
        summary = trace_summary(trace_path)
        ws = set(words(raw_text))
        top = top20.get(sample_id, {})
        row = {
            "sample_id": sample_id,
            "poster_id": poster_id(sample_id),
            "prompt_variant": prompt_variant,
            "behavior": classify_behavior(raw_text, prompt_variant),
            "word_count": len(raw_text.split()),
            **summary,
            "outside_knowledge_hits": len(ws & OUTSIDE_KNOWLEDGE),
            "outside_knowledge_words": ";".join(sorted(ws & OUTSIDE_KNOWLEDGE)),
            "top_category_counts": top.get("top_category_counts", ""),
            "top_span_preview": top.get("top_span_preview", ""),
            "raw_text_preview": raw_text.replace("\n", "\\n")[:500],
        }
        sample_rows.append(row)
        by_variant[prompt_variant].append(row)

    variant_rows = []
    for prompt_variant, rows in sorted(by_variant.items()):
        behaviors = Counter(r["behavior"] for r in rows)
        cats = Counter()
        for row in rows:
            cats.update(parse_counts(row["top_category_counts"]))
        variant_rows.append(
            {
                "prompt_variant": prompt_variant,
                "sample_count": len(rows),
                "behavior_counts": ";".join(f"{k}:{v}" for k, v in behaviors.most_common()),
                "mean_word_count": mean(float(r["word_count"]) for r in rows),
                "mean_trace_tokens": mean(float(r["trace_token_count"]) for r in rows),
                "mean_ume": mean(float(r["mean_ume"]) for r in rows),
                "outside_knowledge_hit_rate": sum(int(r["outside_knowledge_hits"]) > 0 for r in rows) / len(rows),
                "mean_outside_knowledge_hits": mean(float(r["outside_knowledge_hits"]) for r in rows),
                "aggregate_top_categories": ";".join(f"{k}:{v}" for k, v in cats.most_common(10)),
            }
        )

    write_csv(out_dir / "gt_prompt_variant_sample_summary.csv", sample_rows)
    write_csv(out_dir / "gt_prompt_variant_summary.csv", variant_rows)
    write_report(out_dir / "gt_prompt_variant_report.md", sample_rows, variant_rows)
    print(f"[INFO] wrote {len(sample_rows)} samples and {len(variant_rows)} variants to {out_dir}")


if __name__ == "__main__":
    main()
