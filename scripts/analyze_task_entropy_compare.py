#!/usr/bin/env python3
"""Summarize task-level entropy comparisons for the synthetic Emu3.5 suite."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", required=True, help="Directory containing image_understanding_worker*.jsonl")
    parser.add_argument("--bench-dir", required=True, help="Directory containing text/image modal-memory rows")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-understanding", type=int, default=3360)
    return parser.parse_args()


def read_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line:
                rows.append(json.loads(line))
    return rows


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)

    def pct(q: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        pos = (len(ordered) - 1) * q
        lo = int(pos)
        hi = min(lo + 1, len(ordered) - 1)
        frac = pos - lo
        return ordered[lo] * (1.0 - frac) + ordered[hi] * frac

    return {
        "n": len(values),
        "mean": mean(values),
        "median": median(values),
        "p10": pct(0.10),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "max": ordered[-1],
    }


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx = mean(xs)
    my = mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = sum((x - mx) ** 2 for x in xs)
    den_y = sum((y - my) ** 2 for y in ys)
    if den_x <= 0 or den_y <= 0:
        return None
    return num / ((den_x * den_y) ** 0.5)


def auc_score(scores: list[float], labels: list[int]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum = 0.0
    rank = 1
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (rank + rank + (j - i) - 1) / 2.0
        rank_sum += avg_rank * sum(label for _, label in pairs[i:j])
        rank += j - i
        i = j
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def entropy_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    vals: list[float] = []
    for row in rows:
        value = safe_float((row.get("entropy_summary") or {}).get(key))
        if value is not None:
            vals.append(value)
    return vals


def first_text_entropy(row: dict[str, Any]) -> float | None:
    path = row.get("entropy_path")
    if not path:
        return None
    trace_path = Path(path)
    if not trace_path.exists():
        return None
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    steps = payload.get("steps") or []
    if not steps:
        return None
    return safe_float(steps[0].get("entropy"))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    task_dir = Path(args.task_dir)
    bench_dir = Path(args.bench_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    understanding = read_jsonl(sorted(task_dir.glob("image_understanding_worker*.jsonl")))
    text_rows = read_jsonl(sorted(bench_dir.glob("text_memory_worker*.jsonl")))
    image_rows = read_jsonl(sorted(bench_dir.glob("image_memory_worker*.jsonl")))

    errors = [0 if bool(row.get("grading_correct")) else 1 for row in understanding]
    first_entropy = entropy_values(understanding, "first_token_entropy")
    mean_entropy = entropy_values(understanding, "mean_entropy")
    max_entropy = entropy_values(understanding, "max_entropy")

    by_attr_rows: list[dict[str, Any]] = []
    for attr in sorted({str(row.get("attribute")) for row in understanding}):
        rows = [row for row in understanding if row.get("attribute") == attr]
        labels = [0 if bool(row.get("grading_correct")) else 1 for row in rows]
        attr_first = entropy_values(rows, "first_token_entropy")
        attr_mean = entropy_values(rows, "mean_entropy")
        by_attr_rows.append(
            {
                "attribute": attr,
                "n": len(rows),
                "correct": sum(1 - label for label in labels),
                "accuracy": (sum(1 - label for label in labels) / len(rows)) if rows else None,
                "first_entropy_mean": stats(attr_first).get("mean"),
                "first_entropy_median": stats(attr_first).get("median"),
                "mean_entropy_mean": stats(attr_mean).get("mean"),
                "error_first_entropy_mean": stats(
                    [
                        safe_float((row.get("entropy_summary") or {}).get("first_token_entropy"))
                        for row in rows
                        if not bool(row.get("grading_correct"))
                    ]
                ).get("mean"),
                "correct_first_entropy_mean": stats(
                    [
                        safe_float((row.get("entropy_summary") or {}).get("first_token_entropy"))
                        for row in rows
                        if bool(row.get("grading_correct"))
                    ]
                ).get("mean"),
                "first_entropy_error_auc": auc_score(attr_first, labels) if len(attr_first) == len(labels) else None,
            }
        )

    split_rows: list[dict[str, Any]] = []
    for split in sorted({str(row.get("split")) for row in understanding}):
        rows = [row for row in understanding if row.get("split") == split]
        labels = [0 if bool(row.get("grading_correct")) else 1 for row in rows]
        split_rows.append(
            {
                "split": split,
                "n": len(rows),
                "correct": sum(1 - label for label in labels),
                "accuracy": (sum(1 - label for label in labels) / len(rows)) if rows else None,
                "first_entropy_mean": stats(entropy_values(rows, "first_token_entropy")).get("mean"),
                "mean_entropy_mean": stats(entropy_values(rows, "mean_entropy")).get("mean"),
            }
        )

    top_entropy = sorted(
        understanding,
        key=lambda row: safe_float((row.get("entropy_summary") or {}).get("first_token_entropy")) or -1.0,
        reverse=True,
    )[:30]
    error_rows = [row for row in understanding if not bool(row.get("grading_correct"))]
    high_conf_errors = sorted(
        error_rows,
        key=lambda row: safe_float((row.get("entropy_summary") or {}).get("first_token_entropy")) or 999.0,
    )[:30]

    def compact(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "sample_id": row.get("sample_id"),
            "split": row.get("split"),
            "attribute": row.get("attribute"),
            "expected_value": row.get("expected_value"),
            "expected_key": row.get("expected_key"),
            "answer_key": row.get("grading_answer_key"),
            "correct": row.get("grading_correct"),
            "completion": row.get("inference_completion"),
            "first_entropy": (row.get("entropy_summary") or {}).get("first_token_entropy"),
            "mean_entropy": (row.get("entropy_summary") or {}).get("mean_entropy"),
        }

    write_csv(out_dir / "attribute_summary.csv", by_attr_rows, list(by_attr_rows[0].keys()) if by_attr_rows else [])
    write_csv(out_dir / "split_summary.csv", split_rows, list(split_rows[0].keys()) if split_rows else [])
    write_csv(out_dir / "top_first_entropy_examples.csv", [compact(row) for row in top_entropy], list(compact(top_entropy[0]).keys()) if top_entropy else [])
    write_csv(out_dir / "low_entropy_error_examples.csv", [compact(row) for row in high_conf_errors], list(compact(high_conf_errors[0]).keys()) if high_conf_errors else [])

    text_first = [value for row in text_rows if (value := first_text_entropy(row)) is not None]
    image_visual = entropy_values(image_rows, "mean_visual_entropy")
    error_counter = Counter(str(row.get("attribute")) for row in error_rows)

    summary = {
        "task_dir": str(task_dir),
        "bench_dir": str(bench_dir),
        "output_dir": str(out_dir),
        "completeness": {
            "understanding_rows": len(understanding),
            "expected_understanding_rows": args.expected_understanding,
            "fraction": len(understanding) / args.expected_understanding if args.expected_understanding else None,
            "text_rows": len(text_rows),
            "image_rows": len(image_rows),
        },
        "image_understanding": {
            "accuracy": (sum(1 - e for e in errors) / len(errors)) if errors else None,
            "correct": sum(1 - e for e in errors),
            "errors": sum(errors),
            "error_by_attribute": dict(error_counter),
            "first_answer_token_entropy": stats(first_entropy),
            "mean_answer_entropy": stats(mean_entropy),
            "max_answer_entropy": stats(max_entropy),
            "first_entropy_error_pearson": pearson(first_entropy, errors) if len(first_entropy) == len(errors) else None,
            "first_entropy_error_auc": auc_score(first_entropy, errors) if len(first_entropy) == len(errors) else None,
            "mean_entropy_error_auc": auc_score(mean_entropy, errors) if len(mean_entropy) == len(errors) else None,
        },
        "text_question_answer": {
            "rows": len(text_rows),
            "first_answer_token_entropy": stats(text_first),
        },
        "image_generation": {
            "rows": len(image_rows),
            "mean_visual_token_entropy": stats(image_visual),
        },
        "by_attribute": by_attr_rows,
        "by_split": split_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Task Entropy Compare Full Report",
        "",
        "## Run Inputs",
        "",
        f"- task_dir: `{task_dir}`",
        f"- bench_dir: `{bench_dir}`",
        f"- understanding rows: {len(understanding)} / {args.expected_understanding}",
        f"- text rows: {len(text_rows)}",
        f"- image rows: {len(image_rows)}",
        "",
        "## Main Results",
        "",
        f"- image-understanding accuracy: {summary['image_understanding']['accuracy']:.4f}"
        if errors
        else "- image-understanding accuracy: n/a",
        f"- understanding first-token entropy mean: {summary['image_understanding']['first_answer_token_entropy'].get('mean'):.4f}",
        f"- understanding mean answer entropy: {summary['image_understanding']['mean_answer_entropy'].get('mean'):.4f}",
        f"- text QA first-token entropy mean: {summary['text_question_answer']['first_answer_token_entropy'].get('mean'):.4f}",
        f"- image-generation mean visual-token entropy: {summary['image_generation']['mean_visual_token_entropy'].get('mean'):.4f}",
        f"- first-token entropy vs error AUC: {summary['image_understanding']['first_entropy_error_auc']:.4f}"
        if summary["image_understanding"]["first_entropy_error_auc"] is not None
        else "- first-token entropy vs error AUC: n/a",
        "",
        "## Attribute Summary",
        "",
        "| attribute | n | accuracy | first entropy mean | error first entropy mean | AUC(error) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in by_attr_rows:
        lines.append(
            "| {attribute} | {n} | {accuracy:.4f} | {first_entropy_mean:.4f} | {error_first_entropy_mean} | {first_entropy_error_auc} |".format(
                attribute=row["attribute"],
                n=row["n"],
                accuracy=row["accuracy"] if row["accuracy"] is not None else float("nan"),
                first_entropy_mean=row["first_entropy_mean"] if row["first_entropy_mean"] is not None else float("nan"),
                error_first_entropy_mean=(
                    f"{row['error_first_entropy_mean']:.4f}" if row["error_first_entropy_mean"] is not None else "n/a"
                ),
                first_entropy_error_auc=(
                    f"{row['first_entropy_error_auc']:.4f}" if row["first_entropy_error_auc"] is not None else "n/a"
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This report only uses rows that were already written by the real Emu3.5 runs.",
            "- High-confidence errors are listed separately because low entropy with wrong answers is a false-confidence signal.",
            "- The synthetic multiple-choice understanding task is intentionally narrow; use it as a controlled calibration check, not as a replacement for open-ended readback entropy.",
            "",
            "## Artifacts",
            "",
            "- `summary.json`",
            "- `attribute_summary.csv`",
            "- `split_summary.csv`",
            "- `top_first_entropy_examples.csv`",
            "- `low_entropy_error_examples.csv`",
        ]
    )
    (out_dir / "task_entropy_compare_full_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
