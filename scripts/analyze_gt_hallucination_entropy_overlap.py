#!/usr/bin/env python3
"""Compare GT-poster mismatches with top-entropy token neighborhoods."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "there",
    "this",
    "to",
    "with",
}

COMMON_POSTER_WORDS = {
    "poster",
    "movie",
    "film",
    "image",
    "features",
    "featuring",
    "depicts",
    "shows",
    "shown",
    "appears",
    "visible",
    "prominently",
    "displayed",
    "background",
    "foreground",
    "center",
    "central",
    "top",
    "bottom",
    "left",
    "right",
    "large",
    "small",
    "text",
    "title",
    "font",
    "letters",
    "color",
    "colors",
    "palette",
    "composition",
    "dark",
    "bright",
    "white",
    "black",
    "blue",
    "red",
    "yellow",
    "green",
    "orange",
    "gray",
    "grey",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-run-dir", required=True)
    parser.add_argument("--candidate-run", action="append", required=True, help="LABEL=RUN_DIR")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--quantile", type=float, default=0.8)
    parser.add_argument("--window", type=int, default=3)
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


def poster_key(sample_id: str) -> str:
    match = re.match(r"(poster_\d+)", sample_id)
    return match.group(1) if match else sample_id


def words(text: str) -> list[str]:
    text = re.sub(r"<\|[^>]+?\|>", " ", text.lower())
    return re.findall(r"[a-z][a-z0-9]+", text)


def content_words(text: str) -> set[str]:
    return {w for w in words(text) if len(w) > 2 and w not in STOPWORDS and w not in COMMON_POSTER_WORDS}


def token_content(row: dict) -> set[str]:
    return content_words(str(row.get("token_text", "")))


def load_reference_vocab(gt_run_dir: Path) -> dict[str, set[str]]:
    refs: dict[str, set[str]] = {}
    raw_dir = gt_run_dir / "raw_generations"
    for path in sorted(raw_dir.glob("*__gt_describe.txt")):
        refs[poster_key(path.stem)] = content_words(path.read_text(encoding="utf-8"))
    return refs


def load_traces(run_dir: Path) -> dict[str, list[dict]]:
    traces = {}
    for path in sorted((run_dir / "entropy_traces").glob("*_entropy.jsonl")):
        sample_id = path.name.replace("_entropy.jsonl", "")
        traces[sample_id] = [
            r for r in read_jsonl(path) if r.get("token_type") in {"text", "thinking"}
        ]
    return traces


def parse_candidate(arg: str) -> tuple[str, Path]:
    if "=" not in arg:
        raise SystemExit(f"--candidate-run must be LABEL=RUN_DIR, got: {arg}")
    label, path = arg.split("=", 1)
    return label, Path(path)


def reconstruct(rows: list[dict]) -> str:
    text = "".join(str(r.get("token_text", "")) for r in rows)
    text = re.sub(r"<\|[^>]+?\|>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def analyze_sample(
    label: str,
    sample_id: str,
    rows: list[dict],
    ref_vocab: set[str],
    quantile: float,
    window: int,
) -> tuple[dict, list[dict], list[dict]]:
    scores = [float(r.get("ume", 0.0) or 0.0) for r in rows]
    threshold = percentile(scores, quantile)
    high_steps = {int(r.get("step", idx)) for idx, r in enumerate(rows) if float(r.get("ume", 0.0) or 0.0) >= threshold}
    all_steps = [int(r.get("step", idx)) for idx, r in enumerate(rows)]
    all_near_top20 = sum(1 for step in all_steps if any(abs(high - step) <= window for high in high_steps))
    all_near_rate = all_near_top20 / len(all_steps) if all_steps else 0.0
    candidate_rows = []
    for idx, row in enumerate(rows):
        step = int(row.get("step", idx))
        row_words = token_content(row)
        mismatch_words = sorted(w for w in row_words if w not in ref_vocab)
        if not mismatch_words:
            continue
        near_steps = [s for s in high_steps if abs(s - step) <= window]
        candidate_rows.append(
            {
                "run_label": label,
                "sample_id": sample_id,
                "poster_key": poster_key(sample_id),
                "step": step,
                "token_text": row.get("token_text", ""),
                "mismatch_words": ";".join(mismatch_words),
                "ume": float(row.get("ume", 0.0) or 0.0),
                "is_top20_entropy": step in high_steps,
                "near_top20_entropy_window": bool(near_steps),
                "nearest_top20_distance": min((abs(s - step) for s in near_steps), default=""),
            }
        )

    span_rows = []
    current = []
    prev = None
    for row in candidate_rows:
        step = int(row["step"])
        if current and prev is not None and step != prev + 1:
            span_rows.append(current)
            current = []
        current.append(row)
        prev = step
    if current:
        span_rows.append(current)

    span_out = []
    rows_by_step = {int(r.get("step", idx)): r for idx, r in enumerate(rows)}
    for span_idx, span in enumerate(span_rows):
        start = int(span[0]["step"])
        end = int(span[-1]["step"])
        context = [rows_by_step[s] for s in range(start - window, end + window + 1) if s in rows_by_step]
        all_words = sorted({w for r in span for w in str(r["mismatch_words"]).split(";") if w})
        exact = sum(1 for r in span if r["is_top20_entropy"])
        near = sum(1 for r in span if r["near_top20_entropy_window"])
        span_out.append(
            {
                "run_label": label,
                "sample_id": sample_id,
                "poster_key": poster_key(sample_id),
                "span_id": span_idx,
                "start_step": start,
                "end_step": end,
                "token_count": len(span),
                "mismatch_words": ";".join(all_words),
                "mean_ume": sum(float(r["ume"]) for r in span) / len(span),
                "exact_top20_token_count": exact,
                "near_top20_token_count": near,
                "near_top20_fraction": near / len(span),
                "span_text": reconstruct([rows_by_step[s] for s in range(start, end + 1) if s in rows_by_step]),
                "context_window_text": reconstruct(context),
            }
        )

    total_mismatch = len(candidate_rows)
    exact = sum(1 for r in candidate_rows if r["is_top20_entropy"])
    near = sum(1 for r in candidate_rows if r["near_top20_entropy_window"])
    exact_rate = exact / total_mismatch if total_mismatch else ""
    near_rate = near / total_mismatch if total_mismatch else ""
    non_mismatch_scores = [
        float(r.get("ume", 0.0) or 0.0)
        for r in rows
        if not (token_content(r) and any(w not in ref_vocab for w in token_content(r)))
    ]
    mismatch_scores = [float(r["ume"]) for r in candidate_rows]
    summary = {
        "run_label": label,
        "sample_id": sample_id,
        "poster_key": poster_key(sample_id),
        "token_count": len(rows),
        "top20_threshold": threshold,
        "mismatch_token_count": total_mismatch,
        "mismatch_exact_top20_count": exact,
        "mismatch_near_top20_count": near,
        "mismatch_exact_top20_rate": exact_rate,
        "mismatch_near_top20_rate": near_rate,
        "all_token_near_top20_rate": all_near_rate,
        "near_top20_enrichment_vs_all": (near_rate / all_near_rate) if total_mismatch and all_near_rate else "",
        "exact_top20_enrichment_vs_expected": (exact_rate / (1.0 - quantile)) if total_mismatch and quantile < 1.0 else "",
        "mean_mismatch_ume": sum(mismatch_scores) / len(mismatch_scores) if mismatch_scores else "",
        "mean_non_mismatch_ume": sum(non_mismatch_scores) / len(non_mismatch_scores) if non_mismatch_scores else "",
        "top_mismatch_words": ";".join(w for w, _ in Counter(w for r in candidate_rows for w in r["mismatch_words"].split(";") if w).most_common(12)),
    }
    return summary, candidate_rows, span_out


def write_report(path: Path, summaries: list[dict], spans: list[dict]) -> None:
    by_label = defaultdict(list)
    for row in summaries:
        by_label[row["run_label"]].append(row)
    lines = [
        "# GT Hallucination / Top-Entropy Overlap",
        "",
        "Mismatch tokens are automatic lexical candidates: content words in a candidate answer that are absent from the real-poster GT readback vocabulary. They are not final human hallucination labels.",
        "",
        "A mismatch token is counted as `near` when it is within +/- window tokens of a top-20% UME token in the same generated answer.",
        "",
    ]
    for label, rows in sorted(by_label.items()):
        total = sum(int(r["mismatch_token_count"]) for r in rows)
        exact = sum(int(r["mismatch_exact_top20_count"]) for r in rows)
        near = sum(int(r["mismatch_near_top20_count"]) for r in rows)
        baseline_num = sum(float(r["all_token_near_top20_rate"]) * int(r["token_count"]) for r in rows)
        baseline_den = sum(int(r["token_count"]) for r in rows)
        baseline = baseline_num / baseline_den if baseline_den else 0.0
        lines.append(f"## {label}")
        lines.append(f"- samples: {len(rows)}")
        lines.append(f"- mismatch candidate tokens: {total}")
        lines.append(f"- exact top20 overlap: {exact}/{total} = {exact / total:.3f}" if total else "- exact top20 overlap: n/a")
        lines.append(f"- +/-window top20 overlap: {near}/{total} = {near / total:.3f}" if total else "- +/-window top20 overlap: n/a")
        lines.append(f"- all-token +/-window baseline: {baseline:.3f}")
        if total and baseline:
            lines.append(f"- near-overlap enrichment vs baseline: {(near / total) / baseline:.3f}x")
        lines.append("")
        for row in rows:
            lines.append(
                f"- `{row['sample_id']}` mismatch={row['mismatch_token_count']} "
                f"near_rate={row['mismatch_near_top20_rate']} "
                f"baseline={float(row['all_token_near_top20_rate']):.3f} "
                f"enrich={row['near_top20_enrichment_vs_all']} words={row['top_mismatch_words']}"
            )
        lines.append("")

    lines.extend(["## High-Entropy Nearby Mismatch Spans", ""])
    ranked = sorted(spans, key=lambda r: (float(r["near_top20_fraction"]), float(r["mean_ume"])), reverse=True)
    for row in ranked[:80]:
        if float(row["near_top20_fraction"]) <= 0:
            continue
        lines.append(
            f"- `{row['run_label']}` `{row['sample_id']}` {row['start_step']}-{row['end_step']} "
            f"near={float(row['near_top20_fraction']):.2f} mean_ume={float(row['mean_ume']):.4f} "
            f"words={row['mismatch_words']}: `{row['context_window_text']}`"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    refs = load_reference_vocab(Path(args.gt_run_dir))
    summary_rows = []
    token_rows = []
    span_rows = []
    for item in args.candidate_run:
        label, run_dir = parse_candidate(item)
        for sample_id, rows in load_traces(run_dir).items():
            key = poster_key(sample_id)
            if key not in refs:
                continue
            summary, tokens, spans = analyze_sample(label, sample_id, rows, refs[key], args.quantile, args.window)
            summary_rows.append(summary)
            token_rows.extend(tokens)
            span_rows.extend(spans)
    write_csv(out_dir / "gt_mismatch_entropy_summary.csv", summary_rows)
    write_csv(out_dir / "gt_mismatch_token_rows.csv", token_rows)
    write_csv(out_dir / "gt_mismatch_span_rows.csv", span_rows)
    write_report(out_dir / "gt_hallucination_entropy_overlap_report.md", summary_rows, span_rows)
    print(f"[INFO] wrote {len(summary_rows)} summaries, {len(token_rows)} mismatch tokens, {len(span_rows)} spans to {out_dir}")


if __name__ == "__main__":
    main()
