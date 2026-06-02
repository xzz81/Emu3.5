#!/usr/bin/env python3
"""Map manually labeled answer spans to token-level entropy traces."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-file", required=True)
    parser.add_argument("--run-map", action="append", required=True, help="LABEL=RUN_DIR")
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


def parse_run_map(items: list[str]) -> dict[str, Path]:
    out = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--run-map must be LABEL=RUN_DIR, got: {item}")
        label, path = item.split("=", 1)
        out[label] = Path(path)
    return out


def load_trace(run_dir: Path, sample_id: str) -> list[dict]:
    path = run_dir / "entropy_traces" / f"{sample_id}_entropy.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    return [r for r in read_jsonl(path) if r.get("token_type") in {"text", "thinking"}]


def token_spans(rows: list[dict]) -> tuple[str, list[dict]]:
    text = ""
    spans = []
    for idx, row in enumerate(rows):
        token = str(row.get("token_text", ""))
        start = len(text)
        text += token
        end = len(text)
        spans.append({"idx": idx, "step": int(row.get("step", idx)), "start": start, "end": end, "row": row})
    return text, spans


def find_span(text: str, snippet: str) -> tuple[int, int]:
    start = text.find(snippet)
    if start >= 0:
        return start, start + len(snippet)
    compact_text = " ".join(text.split())
    compact_snippet = " ".join(snippet.split())
    compact_start = compact_text.find(compact_snippet)
    if compact_start < 0:
        raise ValueError(f"snippet not found: {snippet}")
    raise ValueError(f"snippet found only after whitespace normalization; use exact raw snippet: {snippet}")


def overlap_tokens(spans: list[dict], start: int, end: int) -> list[dict]:
    return [item for item in spans if item["start"] < end and item["end"] > start]


def analyze_label(label: dict, run_maps: dict[str, Path], quantile: float, window: int) -> dict:
    rows = load_trace(run_maps[label["run_label"]], label["sample_id"])
    values = [float(r.get("ume", 0.0) or 0.0) for r in rows]
    threshold = percentile(values, quantile)
    high_steps = {int(r.get("step", idx)) for idx, r in enumerate(rows) if float(r.get("ume", 0.0) or 0.0) >= threshold}
    text, spans = token_spans(rows)
    start, end = find_span(text, label["snippet"])
    selected = overlap_tokens(spans, start, end)
    if not selected:
        raise ValueError(f"no overlapping tokens for {label['label_id']}")
    steps = [item["step"] for item in selected]
    scores = [float(item["row"].get("ume", 0.0) or 0.0) for item in selected]
    exact = sum(1 for step in steps if step in high_steps)
    near = sum(1 for step in steps if any(abs(step - high) <= window for high in high_steps))
    length = len(selected)
    baseline_windows = []
    for offset in range(0, max(0, len(rows) - length + 1)):
        window_rows = rows[offset : offset + length]
        window_steps = [int(r.get("step", offset + idx)) for idx, r in enumerate(window_rows, start=0)]
        window_scores = [float(r.get("ume", 0.0) or 0.0) for r in window_rows]
        baseline_windows.append(
            {
                "mean_ume": sum(window_scores) / len(window_scores),
                "has_exact": any(step in high_steps for step in window_steps),
                "has_near": any(any(abs(step - high) <= window for high in high_steps) for step in window_steps),
            }
        )
    baseline_mean_ume = sum(r["mean_ume"] for r in baseline_windows) / len(baseline_windows)
    baseline_exact_span_rate = sum(1 for r in baseline_windows if r["has_exact"]) / len(baseline_windows)
    baseline_near_span_rate = sum(1 for r in baseline_windows if r["has_near"]) / len(baseline_windows)
    context_start = max(0, start - 80)
    context_end = min(len(text), end + 80)
    return {
        "label_id": label["label_id"],
        "run_label": label["run_label"],
        "sample_id": label["sample_id"],
        "truth_label": label["truth_label"],
        "aspect": label["aspect"],
        "token_count": len(selected),
        "start_step": min(steps),
        "end_step": max(steps),
        "mean_ume": sum(scores) / len(scores),
        "max_ume": max(scores),
        "baseline_same_length_mean_ume": baseline_mean_ume,
        "mean_ume_minus_baseline": (sum(scores) / len(scores)) - baseline_mean_ume,
        "top20_threshold": threshold,
        "exact_top20_token_count": exact,
        "exact_top20_fraction": exact / len(selected),
        "near_top20_token_count": near,
        "near_top20_fraction": near / len(selected),
        "baseline_same_length_exact_span_rate": baseline_exact_span_rate,
        "baseline_same_length_near_span_rate": baseline_near_span_rate,
        "entropy_class": "high" if exact else ("near_high" if near else "not_high"),
        "snippet": label["snippet"],
        "context": text[context_start:context_end].replace("\n", "\\n"),
        "notes": label.get("notes", ""),
    }


def write_report(path: Path, rows: list[dict], quantile: float, window: int) -> None:
    by_truth = defaultdict(list)
    by_aspect = defaultdict(list)
    for row in rows:
        by_truth[row["truth_label"]].append(row)
        by_aspect[row["aspect"]].append(row)

    lines = [
        "# Manual Span Label vs Entropy Report",
        "",
        f"Top entropy threshold: per-sample top {(1.0 - quantile) * 100:.0f}% UME.",
        f"Near-high window: +/-{window} generated tokens.",
        "",
        "## Truth Label Summary",
        "",
    ]
    for truth, group in sorted(by_truth.items()):
        exact = sum(1 for row in group if float(row["exact_top20_fraction"]) > 0)
        near = sum(1 for row in group if float(row["near_top20_fraction"]) > 0)
        mean_ume = sum(float(row["mean_ume"]) for row in group) / len(group)
        mean_delta = sum(float(row["mean_ume_minus_baseline"]) for row in group) / len(group)
        baseline_exact = sum(float(row["baseline_same_length_exact_span_rate"]) for row in group) / len(group)
        lines.append(
            f"- {truth}: n={len(group)}, span_exact_top20={exact}/{len(group)} "
            f"({exact / len(group):.3f}), span_near_top20={near}/{len(group)} "
            f"({near / len(group):.3f}), mean_span_ume={mean_ume:.4f}, "
            f"mean_delta_vs_same_length_windows={mean_delta:+.4f}, "
            f"same_length_exact_baseline={baseline_exact:.3f}"
        )

    lines.extend(["", "## Aspect Summary", ""])
    for aspect, group in sorted(by_aspect.items()):
        hallucinated = sum(1 for row in group if row["truth_label"] == "hallucinated")
        exact = sum(1 for row in group if float(row["exact_top20_fraction"]) > 0)
        mean_ume = sum(float(row["mean_ume"]) for row in group) / len(group)
        lines.append(f"- {aspect}: n={len(group)}, hallucinated={hallucinated}, exact_top20_spans={exact}, mean_ume={mean_ume:.4f}")

    lines.extend(["", "## Labeled Spans", ""])
    for row in sorted(rows, key=lambda r: (r["truth_label"], r["run_label"], r["sample_id"], int(r["start_step"]))):
        lines.append(
            f"- `{row['label_id']}` {row['truth_label']} {row['aspect']} "
            f"{row['start_step']}-{row['end_step']} class={row['entropy_class']} "
            f"mean_ume={float(row['mean_ume']):.4f} delta={float(row['mean_ume_minus_baseline']):+.4f} "
            f"exact={float(row['exact_top20_fraction']):.2f} "
            f"near={float(row['near_top20_fraction']):.2f}: `{row['snippet']}`"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_maps = parse_run_map(args.run_map)
    labels = read_jsonl(Path(args.label_file))
    rows = [analyze_label(label, run_maps, args.quantile, args.window) for label in labels]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "manual_span_entropy_rows.csv", rows)
    write_report(out_dir / "manual_span_entropy_report.md", rows, args.quantile, args.window)
    print(f"[INFO] wrote {len(rows)} manual span entropy rows to {out_dir}")


if __name__ == "__main__":
    main()
