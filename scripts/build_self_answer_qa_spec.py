#!/usr/bin/env python3
"""Build a QA spec that uses clean model readback answers as targets."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


INVALID_PATTERNS = (
    "to determine",
    "to finish",
    "can't see",
    "cannot see",
    "<|image start|>",
    "<|image token|>",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-answer-words", type=int, default=8)
    return parser.parse_args()


def clean_answer(text: str):
    text = re.sub(r"<\|extra_60\|>|<\|extra_100\|>|<\|extra_101\|>|<\|extra_204\|>", " ", text)
    text = re.sub(r"<\|[^>]+?\|>", " ", text)
    text = re.sub(r"^\s*answer\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip(" .")
    return text


def is_valid_answer(raw: str, cleaned: str, max_words: int):
    raw_lower = raw.lower()
    cleaned_lower = cleaned.lower()
    if not cleaned:
        return False
    if any(pattern in raw_lower or pattern in cleaned_lower for pattern in INVALID_PATTERNS):
        return False
    words = re.findall(r"[A-Za-z0-9]+", cleaned)
    if len(words) > max_words:
        return False
    return True


def main():
    args = parse_args()
    rows = list(csv.DictReader(Path(args.validation_csv).open(encoding="utf-8")))
    spec = {"__replace_defaults__": True}
    kept_rows = []
    for row in rows:
        cleaned = clean_answer(row["generated_answer"])
        keep = is_valid_answer(row["generated_answer"], cleaned, args.max_answer_words)
        if keep:
            spec[row["sample_id"]] = {
                "question": row["question"],
                "answer": cleaned,
                "original_target_answer": row["target_answer"],
                "source": "model_self_answer",
            }
        out = dict(row)
        out["cleaned_self_answer"] = cleaned
        out["keep_self_answer"] = keep
        kept_rows.append(out)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit_path = out_path.with_suffix(".audit.csv")
    with audit_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(kept_rows[0].keys()))
        writer.writeheader()
        writer.writerows(kept_rows)
    print(f"[INFO] wrote {len(spec) - 1} self-answer QA specs to {out_path}")
    print(f"[INFO] wrote audit rows to {audit_path}")


if __name__ == "__main__":
    main()
