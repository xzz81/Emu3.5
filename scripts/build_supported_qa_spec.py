#!/usr/bin/env python3
"""Build a fixed-target QA spec from validation rows supported by model readback."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-match-fraction", type=float, default=0.67)
    parser.add_argument("--max-answer-words", type=int, default=8)
    parser.add_argument("--max-target-minus-generated-nll", type=float, default=None)
    return parser.parse_args()


def answer_word_count(text: str):
    return len(re.findall(r"[A-Za-z0-9]+", text))


def as_bool(text: str):
    return str(text).strip().lower() in {"1", "true", "yes", "y"}


def keep_row(row, args):
    if not as_bool(row.get("qa_supported", "")):
        return False
    if float(row.get("target_match_fraction", 0.0)) < args.min_match_fraction:
        return False
    if answer_word_count(row["target_answer"]) > args.max_answer_words:
        return False
    if args.max_target_minus_generated_nll is not None:
        if float(row["target_minus_generated_nll"]) > args.max_target_minus_generated_nll:
            return False
    return True


def main():
    args = parse_args()
    rows = list(csv.DictReader(Path(args.validation_csv).open(encoding="utf-8")))
    spec = {"__replace_defaults__": True}
    audit_rows = []
    for row in rows:
        keep = keep_row(row, args)
        if keep:
            spec[row["sample_id"]] = {
                "question": row["question"],
                "answer": row["target_answer"],
                "source": "validated_fixed_target",
                "generated_answer": row["generated_answer"],
                "target_match_fraction": float(row["target_match_fraction"]),
                "target_minus_generated_nll": float(row["target_minus_generated_nll"]),
            }
        audit = dict(row)
        audit["keep_supported_target"] = keep
        audit_rows.append(audit)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    audit_path = out_path.with_suffix(".audit.csv")
    if audit_rows:
        with audit_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()))
            writer.writeheader()
            writer.writerows(audit_rows)
    else:
        audit_path.write_text("", encoding="utf-8")

    print(f"[INFO] wrote {len(spec) - 1} supported QA specs to {out_path}")
    print(f"[INFO] wrote audit rows to {audit_path}")


if __name__ == "__main__":
    main()
