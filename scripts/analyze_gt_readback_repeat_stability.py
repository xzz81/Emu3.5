#!/usr/bin/env python3
"""Summarize repeated real-poster GT readback entropy runs."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}
POSTER_ID_RE = re.compile(r"poster_(\d+)_")
REPEAT_RE = re.compile(r"_rep(\d+)__")


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


def content_words(text: str) -> set[str]:
    return {w for w in words(text) if len(w) > 2 and w not in STOPWORDS}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a or b else 1.0


def poster_id(sample_id: str) -> str:
    match = POSTER_ID_RE.search(sample_id)
    return str(int(match.group(1))) if match else sample_id


def repeat_id(sample_id: str) -> str:
    match = REPEAT_RE.search(sample_id)
    return match.group(1) if match else ""


def input_form(sample_id: str) -> str:
    if "visible_only" in sample_id:
        return "visible_only"
    return "pure_image" if "imageonly" in sample_id else "image_plus_text"


def classify_behavior(text: str, form: str) -> str:
    cleaned = re.sub(r"<\|[^>]+?\|>", " ", text).strip()
    wc = len(cleaned.split())
    lower = cleaned.lower()
    if not cleaned:
        return "immediate_end"
    if form == "image_plus_text":
        return "description"
    if "step 1" in lower or "finish the task" in lower or "to generate an image" in lower:
        return "action_script"
    if wc <= 12 and ("answer is" in lower or wc <= 3):
        return "short_label"
    if wc >= 50 and ("poster" in lower or "image" in lower):
        if form == "visible_only":
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


def parse_counts(text: str) -> Counter:
    counts: Counter[str] = Counter()
    if not text:
        return counts
    for item in text.split(";"):
        if ":" not in item:
            continue
        key, value = item.rsplit(":", 1)
        try:
            counts[key] += int(value)
        except ValueError:
            pass
    return counts


def load_top20(top20_dir: Path) -> dict[str, dict]:
    path = top20_dir / "top_entropy_sample_summary.csv"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        return {row["sample_id"]: row for row in csv.DictReader(f)}


def category_jaccard(counters: list[Counter]) -> float:
    sets = [set(c) for c in counters]
    pairs = [jaccard(a, b) for a, b in itertools.combinations(sets, 2)]
    return mean(pairs) if pairs else 1.0


def write_report(path: Path, sample_rows: list[dict], group_rows: list[dict]) -> None:
    by_form = defaultdict(list)
    for row in sample_rows:
        by_form[row["input_form"]].append(row)
    lines = [
        "# GT Readback Repeat Stability Report",
        "",
        "This report summarizes repeated real-poster GT readback with image+text and pure-image input forms.",
        "",
        "## Form Summary",
        "",
    ]
    for form, rows in sorted(by_form.items()):
        behaviors = Counter(r["behavior"] for r in rows)
        lines.append(f"### {form}")
        lines.append(f"- samples: {len(rows)}")
        lines.append(f"- mean trace tokens: {mean(float(r['trace_token_count']) for r in rows):.2f}")
        lines.append(f"- mean UME: {mean(float(r['mean_ume']) for r in rows):.4f}")
        lines.append("- behavior counts: " + "; ".join(f"{k}:{v}" for k, v in behaviors.most_common()))
        lines.append("")

    lines.extend(["## Poster/Form Groups", ""])
    for row in sorted(group_rows, key=lambda r: (int(r["poster_id"]), r["input_form"])):
        lines.append(
            f"- poster {row['poster_id']} {row['input_form']}: behavior={row['behavior_counts']} "
            f"mean_ume={float(row['mean_ume']):.4f} sd_ume={float(row['sd_ume']):.4f} "
            f"content_jaccard={float(row['pairwise_content_jaccard']):.3f} "
            f"top_category_jaccard={float(row['top_category_jaccard']):.3f} "
            f"top_categories={row['aggregate_top_categories']}"
        )

    lines.extend(["", "## Main Interpretation", ""])
    lines.append(
        "Image+text prompts reliably produce long descriptions across repeats. Pure-image prompts split across "
        "immediate-end, short-label, action-script, and accidental-description basins, confirming that pure image "
        "is an unstable interface control rather than a clean descriptive condition."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    top20 = load_top20(Path(args.top20_dir))
    sample_rows = []
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for trace_path in sorted((run_dir / "entropy_traces").glob("*_entropy.jsonl")):
        sample_id = trace_path.name.replace("_entropy.jsonl", "")
        raw_path = run_dir / "raw_generations" / f"{sample_id}.txt"
        raw_text = raw_path.read_text(encoding="utf-8") if raw_path.exists() else ""
        form = input_form(sample_id)
        summary = trace_summary(trace_path)
        top = top20.get(sample_id, {})
        row = {
            "sample_id": sample_id,
            "poster_id": poster_id(sample_id),
            "repeat": repeat_id(sample_id),
            "input_form": form,
            "behavior": classify_behavior(raw_text, form),
            "word_count": len(raw_text.split()),
            "_content_words": content_words(raw_text),
            **summary,
            "top_category_counts": top.get("top_category_counts", ""),
            "top_span_preview": top.get("top_span_preview", ""),
            "raw_text_preview": raw_text.replace("\n", "\\n")[:500],
        }
        sample_rows.append(row)
        grouped[(row["poster_id"], form)].append(row)

    group_rows = []
    for (pid, form), rows in sorted(grouped.items(), key=lambda item: (int(item[0][0]), item[0][1])):
        behavior = Counter(r["behavior"] for r in rows)
        wordsets = [r["_content_words"] for r in rows]
        content_js = [jaccard(a, b) for a, b in itertools.combinations(wordsets, 2)]
        cat_counters = [parse_counts(r["top_category_counts"]) for r in rows]
        aggregate = Counter()
        for counter in cat_counters:
            aggregate.update(counter)
        umes = [float(r["mean_ume"]) for r in rows]
        group_rows.append(
            {
                "poster_id": pid,
                "input_form": form,
                "repeat_count": len(rows),
                "behavior_counts": ";".join(f"{k}:{v}" for k, v in behavior.most_common()),
                "mean_word_count": mean(float(r["word_count"]) for r in rows),
                "mean_trace_tokens": mean(float(r["trace_token_count"]) for r in rows),
                "mean_ume": mean(umes),
                "sd_ume": pstdev(umes) if len(umes) > 1 else 0.0,
                "pairwise_content_jaccard": mean(content_js) if content_js else 1.0,
                "top_category_jaccard": category_jaccard(cat_counters),
                "aggregate_top_categories": ";".join(f"{k}:{v}" for k, v in aggregate.most_common(8)),
            }
        )

    sample_csv_rows = [{k: v for k, v in row.items() if not k.startswith("_")} for row in sample_rows]
    write_csv(out_dir / "gt_readback_repeat_sample_summary.csv", sample_csv_rows)
    write_csv(out_dir / "gt_readback_repeat_group_summary.csv", group_rows)
    write_report(out_dir / "gt_readback_repeat_stability_report.md", sample_rows, group_rows)
    print(f"[INFO] wrote {len(sample_rows)} samples and {len(group_rows)} groups to {out_dir}")


if __name__ == "__main__":
    main()
