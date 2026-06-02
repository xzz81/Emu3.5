#!/usr/bin/env python3
"""Analyze shape option-order probes."""

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
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-rows", type=int, default=2520)
    return parser.parse_args()


def read_rows(input_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(input_dir.glob("shape_option_order_worker*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line:
                rows.append(json.loads(line))
    return rows


def entropy(row: dict[str, Any], key: str = "first_token_entropy") -> float | None:
    value = (row.get("entropy_summary") or {}).get(key)
    return float(value) if value is not None else None


def stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "median": None}
    return {"n": len(values), "mean": mean(values), "median": median(values)}


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(input_dir)

    by_variant: list[dict[str, Any]] = []
    for variant in sorted({row["variant"] for row in rows}):
        sub = [row for row in rows if row["variant"] == variant]
        errors = [row for row in sub if not row.get("grading_correct")]
        by_variant.append(
            {
                "variant": variant,
                "n": len(sub),
                "correct": len(sub) - len(errors),
                "accuracy": (len(sub) - len(errors)) / len(sub) if sub else None,
                "mean_first_entropy": stats([v for row in sub if (v := entropy(row)) is not None])["mean"],
                "error_mean_first_entropy": stats([v for row in errors if (v := entropy(row)) is not None])["mean"],
            }
        )

    by_expected_answer: list[dict[str, Any]] = []
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["variant"], row["expected_value"], row.get("answer_value"))].append(row)
    for (variant, expected, answer), sub in sorted(grouped.items()):
        by_expected_answer.append(
            {
                "variant": variant,
                "expected_value": expected,
                "answer_value": answer,
                "n": len(sub),
                "mean_first_entropy": stats([v for row in sub if (v := entropy(row)) is not None])["mean"],
            }
        )

    sample_groups = defaultdict(list)
    for row in rows:
        sample_groups[(row["split"], row["dataset_index"])].append(row)
    consistency_rows: list[dict[str, Any]] = []
    complete_groups = 0
    same_key = 0
    same_value = 0
    all_correct = 0
    all_wrong_same_value = 0
    for (split, idx), sub in sorted(sample_groups.items()):
        if len(sub) != 3:
            continue
        complete_groups += 1
        keys = {row.get("grading_answer_key") for row in sub}
        values = {row.get("answer_value") for row in sub}
        corrects = [bool(row.get("grading_correct")) for row in sub]
        if len(keys) == 1:
            same_key += 1
        if len(values) == 1:
            same_value += 1
        if all(corrects):
            all_correct += 1
        if (not any(corrects)) and len(values) == 1:
            all_wrong_same_value += 1
        consistency_rows.append(
            {
                "split": split,
                "dataset_index": idx,
                "expected_value": sub[0]["expected_value"],
                "same_answer_key_across_variants": int(len(keys) == 1),
                "same_answer_value_across_variants": int(len(values) == 1),
                "num_correct_variants": sum(corrects),
                "answers": "; ".join(
                    f"{row['variant']}:{row.get('grading_answer_key')}={row.get('answer_value')}" for row in sorted(sub, key=lambda item: item["variant"])
                ),
                "first_entropies": "; ".join(
                    f"{row['variant']}:{entropy(row):.4f}" for row in sorted(sub, key=lambda item: item["variant"]) if entropy(row) is not None
                ),
            }
        )

    low_entropy_errors = sorted(
        [row for row in rows if not row.get("grading_correct")],
        key=lambda row: entropy(row) if entropy(row) is not None else 999.0,
    )[:50]
    low_rows = [
        {
            "sample_id": row["sample_id"],
            "split": row["split"],
            "dataset_index": row["dataset_index"],
            "variant": row["variant"],
            "expected_value": row["expected_value"],
            "expected_key": row["expected_key"],
            "answer_key": row.get("grading_answer_key"),
            "answer_value": row.get("answer_value"),
            "first_entropy": entropy(row),
            "completion": row.get("inference_completion"),
        }
        for row in low_entropy_errors
    ]

    write_csv(out_dir / "variant_summary.csv", by_variant, list(by_variant[0].keys()) if by_variant else [])
    write_csv(out_dir / "confusion_by_variant.csv", by_expected_answer, list(by_expected_answer[0].keys()) if by_expected_answer else [])
    write_csv(out_dir / "cross_variant_consistency.csv", consistency_rows, list(consistency_rows[0].keys()) if consistency_rows else [])
    write_csv(out_dir / "low_entropy_errors.csv", low_rows, list(low_rows[0].keys()) if low_rows else [])

    answer_value_counter = Counter((row["expected_value"], row.get("answer_value")) for row in rows)
    key_counter = Counter((row["variant"], row["expected_value"], row.get("grading_answer_key")) for row in rows)
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(out_dir),
        "rows": len(rows),
        "expected_rows": args.expected_rows,
        "unique_sample_ids": len({row["sample_id"] for row in rows}),
        "entropy_files": len(list((input_dir / "entropy").glob("*.json"))),
        "variant_summary": by_variant,
        "complete_image_groups": complete_groups,
        "same_answer_key_groups": same_key,
        "same_answer_value_groups": same_value,
        "all_correct_groups": all_correct,
        "all_wrong_same_value_groups": all_wrong_same_value,
        "answer_value_confusion_top": [
            {"expected_value": key[0], "answer_value": key[1], "n": value}
            for key, value in answer_value_counter.most_common(20)
        ],
        "answer_key_counter_top": [
            {"variant": key[0], "expected_value": key[1], "answer_key": key[2], "n": value}
            for key, value in key_counter.most_common(30)
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Shape Option-Order Probe Report",
        "",
        "## Completeness",
        "",
        f"- rows: {len(rows)} / {args.expected_rows}",
        f"- unique sample_id: {summary['unique_sample_ids']} / {args.expected_rows}",
        f"- entropy files: {summary['entropy_files']} / {args.expected_rows}",
        "",
        "## Variant Summary",
        "",
        "| variant | n | accuracy | mean first entropy | error first entropy |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in by_variant:
        lines.append(
            f"| {row['variant']} | {row['n']} | {row['accuracy']:.4f} | "
            f"{row['mean_first_entropy']:.4f} | {row['error_mean_first_entropy']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Cross-Variant Consistency",
            "",
            f"- complete image groups: {complete_groups}",
            f"- same answer key across variants: {same_key} / {complete_groups}",
            f"- same answer value across variants: {same_value} / {complete_groups}",
            f"- all three variants correct: {all_correct} / {complete_groups}",
            f"- all three variants wrong with the same answer value: {all_wrong_same_value} / {complete_groups}",
            "",
            "## Interpretation",
            "",
            "- If answer values remain stable while answer keys change, the failure is semantic shape confusion rather than fixed letter bias.",
            "- If answer keys remain stable across variants, the failure is option-label or position bias.",
            "- Low-entropy errors are listed separately for false-confidence auditing.",
            "",
            "## Artifacts",
            "",
            "- `summary.json`",
            "- `variant_summary.csv`",
            "- `confusion_by_variant.csv`",
            "- `cross_variant_consistency.csv`",
            "- `low_entropy_errors.csv`",
        ]
    )
    (out_dir / "shape_option_order_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
