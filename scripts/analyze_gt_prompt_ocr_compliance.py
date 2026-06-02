#!/usr/bin/env python3
"""Check OCR/name/title copying under poster readback prompt conditions."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path
from statistics import mean


TITLE_WORDS = {
    "dark",
    "knight",
    "matrix",
    "inception",
    "star",
    "wars",
    "empire",
    "strikes",
    "back",
    "future",
    "harry",
    "potter",
    "chamber",
    "secrets",
    "lord",
    "rings",
    "fellowship",
    "return",
    "king",
}
PERSON_PRIOR_WORDS = {
    "christian",
    "bale",
    "gary",
    "oldman",
    "maggie",
    "gyllenhaal",
    "aaron",
    "eckhart",
    "keanu",
    "reeves",
    "laurence",
    "fishburne",
    "leonardo",
    "dicaprio",
    "christopher",
    "nolan",
    "mark",
    "hamill",
    "harrison",
    "ford",
    "carrie",
    "fisher",
    "spielberg",
    "zemeckis",
    "michael",
    "fox",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", action="append", required=True, help="NAME=RUN_DIR")
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


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


def parse_condition(item: str) -> tuple[str, Path]:
    if "=" not in item:
        raise SystemExit(f"--condition must be NAME=RUN_DIR, got {item!r}")
    name, path = item.split("=", 1)
    return name, Path(path)


def sample_id_from_path(path: Path) -> str:
    return path.name.removesuffix(".txt")


def analyze_file(condition: str, path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    ws = words(text)
    title_hits = sorted(set(ws) & TITLE_WORDS)
    prior_hits = sorted(set(ws) & PERSON_PRIOR_WORDS)
    quoted = re.findall(r'"([^"]{2,120})"', text)
    date_hits = re.findall(r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}\b|\b(?:19|20)\d{2}\b", text, flags=re.I)
    forbidden = bool(title_hits or prior_hits or quoted or date_hits)
    return {
        "condition": condition,
        "sample_id": sample_id_from_path(path),
        "word_count": len(text.split()),
        "has_forbidden_copy": int(forbidden),
        "has_quoted_text": int(bool(quoted)),
        "quote_count": len(quoted),
        "has_title_words": int(bool(title_hits)),
        "title_words": ";".join(title_hits),
        "has_person_prior_words": int(bool(prior_hits)),
        "person_prior_words": ";".join(prior_hits),
        "has_date_text": int(bool(date_hits)),
        "date_text": ";".join(date_hits[:10]),
        "preview": text.replace("\n", "\\n")[:400],
    }


def keep_sample(condition: str, path: Path) -> bool:
    sample_id = sample_id_from_path(path)
    if condition == "describe":
        return "__gt_describe" in sample_id or "__seed72_describe" in sample_id or "__greedy_describe" in sample_id
    return True


def summarize(condition: str, rows: list[dict]) -> dict:
    n = len(rows)
    title_counts = Counter(w for row in rows for w in row["title_words"].split(";") if w)
    prior_counts = Counter(w for row in rows for w in row["person_prior_words"].split(";") if w)
    return {
        "condition": condition,
        "sample_count": n,
        "mean_word_count": f"{mean(float(r['word_count']) for r in rows):.2f}" if rows else "0.00",
        "forbidden_copy_rate": f"{sum(int(r['has_forbidden_copy']) for r in rows) / n:.3f}" if n else "0.000",
        "quoted_text_rate": f"{sum(int(r['has_quoted_text']) for r in rows) / n:.3f}" if n else "0.000",
        "mean_quote_count": f"{mean(float(r['quote_count']) for r in rows):.3f}" if rows else "0.000",
        "title_word_rate": f"{sum(int(r['has_title_words']) for r in rows) / n:.3f}" if n else "0.000",
        "person_prior_rate": f"{sum(int(r['has_person_prior_words']) for r in rows) / n:.3f}" if n else "0.000",
        "date_text_rate": f"{sum(int(r['has_date_text']) for r in rows) / n:.3f}" if n else "0.000",
        "top_title_words": ";".join(f"{k}:{v}" for k, v in title_counts.most_common(20)),
        "top_person_prior_words": ";".join(f"{k}:{v}" for k, v in prior_counts.most_common(20)),
    }


def write_report(path: Path, summary_rows: list[dict]) -> None:
    lines = ["# GT Prompt OCR Compliance", ""]
    for row in summary_rows:
        lines.append(
            f"- {row['condition']}: forbidden={row['forbidden_copy_rate']}, quoted={row['quoted_text_rate']}, "
            f"title_words={row['title_word_rate']}, person_prior={row['person_prior_rate']}, dates={row['date_text_rate']}, "
            f"mean_quotes={row['mean_quote_count']}"
        )
        lines.append(f"  title words: {row['top_title_words']}")
        lines.append(f"  person/prior words: {row['top_person_prior_words']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    sample_rows = []
    summary_rows = []
    for condition, run_dir in (parse_condition(item) for item in args.condition):
        rows = [
            analyze_file(condition, path)
            for path in sorted((run_dir / "raw_generations").glob("*.txt"))
            if keep_sample(condition, path)
        ]
        sample_rows.extend(rows)
        summary_rows.append(summarize(condition, rows))
    write_csv(out_dir / "gt_prompt_ocr_compliance_samples.csv", sample_rows)
    write_csv(out_dir / "gt_prompt_ocr_compliance_summary.csv", summary_rows)
    write_report(out_dir / "gt_prompt_ocr_compliance_report.md", summary_rows)
    print(f"[INFO] wrote OCR compliance for {len(summary_rows)} conditions to {out_dir}")


if __name__ == "__main__":
    main()
