#!/usr/bin/env python3
"""Summarize repeated text-from-memory poster entropy traces."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
from pathlib import Path
from statistics import mean, pstdev


POSTER_ID_RE = re.compile(r"poster_(\d+)_")
REPEAT_RE = re.compile(r"_rep(\d+)__text$")
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
HEDGES = {"appears", "seems", "likely", "might", "possibly", "perhaps", "suggests", "unclear", "looks"}
VISUAL_TERMS = {
    "poster",
    "title",
    "text",
    "background",
    "character",
    "characters",
    "figure",
    "figures",
    "face",
    "faces",
    "man",
    "woman",
    "people",
    "blue",
    "green",
    "red",
    "black",
    "white",
    "yellow",
    "dark",
    "light",
    "center",
    "bottom",
    "top",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--poster-data",
        default="data/modal_aphasia/posters-1.json",
    )
    parser.add_argument(
        "--seed40-summary-csv",
        default="outputs/research_logs/modal_aphasia_posters/seed40/poster_entropy_summary.csv",
    )
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def words(text: str) -> list[str]:
    text = re.sub(r"<\|[^>]+?\|>", " ", text.lower())
    return re.findall(r"[a-z0-9]+", text)


def content_words(text: str) -> set[str]:
    return {w for w in words(text) if len(w) > 2 and w not in STOPWORDS}


def poster_id_from_sample(sample_id: str) -> str:
    match = POSTER_ID_RE.search(sample_id)
    return str(int(match.group(1))) if match else sample_id


def repeat_id(sample_id: str) -> str:
    match = REPEAT_RE.search(sample_id)
    return match.group(1) if match else ""


def p90(values: list[float]) -> float | str:
    if not values:
        return ""
    values = sorted(values)
    idx = min(len(values) - 1, math.ceil(0.9 * len(values)) - 1)
    return values[idx]


def pearson(xs, ys) -> float | None:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if x not in ("", None) and y not in ("", None)]
    if len(pairs) < 2:
        return None
    xs = [x for x, _ in pairs]
    ys = [y for _, y in pairs]
    mx = mean(xs)
    my = mean(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    denom = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    return sum(x * y for x, y in zip(dx, dy)) / denom if denom > 1e-12 else None


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a or b else 1.0


def load_posters(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))["posters"]
    return {str(int(k)): v for k, v in data.items()}


def load_seed40(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {row["poster_id"]: row for row in csv.DictReader(f)}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_trace(records: list[dict]) -> dict:
    text = [r for r in records if r.get("token_type") in {"text", "thinking"}]
    umes = [float(r.get("ume", 0.0)) for r in text]
    return {
        "text_tokens": len(text),
        "mean_ume": mean(umes) if umes else "",
        "p90_ume": p90(umes),
        "first_token_ume": float(text[0].get("ume", 0.0)) if text else "",
        "max_ume": max(umes) if umes else "",
    }


def top_tokens(records: list[dict], poster: dict, sample_id: str, limit: int = 8) -> list[dict]:
    text = [r for r in records if r.get("token_type") in {"text", "thinking"}]
    text.sort(key=lambda r: float(r.get("ume", 0.0)), reverse=True)
    return [
        {
            "poster_id": poster["poster_id"],
            "poster_name": poster["poster_name"],
            "sample_id": sample_id,
            "step": r.get("step"),
            "token_text": r.get("token_text"),
            "ume": r.get("ume"),
            "u_tok": r.get("u_tok"),
            "u_intra": r.get("u_intra"),
        }
        for r in text[:limit]
    ]


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    posters = load_posters(Path(args.poster_data))
    seed40 = load_seed40(Path(args.seed40_summary_csv))

    sample_rows = []
    high_rows = []
    by_poster: dict[str, list[dict]] = {}
    for trace_path in sorted((run_dir / "entropy_traces").glob("*_entropy.jsonl")):
        records = read_jsonl(trace_path)
        if not records:
            continue
        sample_id = str(records[0].get("sample_id", trace_path.stem.replace("_entropy", "")))
        poster_id = poster_id_from_sample(sample_id)
        poster = posters[poster_id]
        raw_path = run_dir / "raw_generations" / f"{sample_id}.txt"
        raw_text = raw_path.read_text(encoding="utf-8") if raw_path.exists() else ""
        ws = set(words(raw_text))
        summary = summarize_trace(records)
        row = {
            "poster_id": poster_id,
            "poster_name": poster["poster_name"],
            "sample_id": sample_id,
            "repeat": repeat_id(sample_id),
            **summary,
            "hedge_hit": int(bool(ws & HEDGES)),
            "visual_term_hits": len(ws & VISUAL_TERMS),
            "content_word_count": len(content_words(raw_text)),
            "raw_text": raw_text.replace("\n", "\\n")[:800],
        }
        sample_rows.append(row)
        by_poster.setdefault(poster_id, []).append(row)
        high_rows.extend(top_tokens(records, poster, sample_id))

    group_rows = []
    for poster_id, group in sorted(by_poster.items(), key=lambda item: int(item[0])):
        wordsets = [content_words(r["raw_text"]) for r in group]
        jaccards = [jaccard(a, b) for a, b in itertools.combinations(wordsets, 2)]
        mean_umes = [float(r["mean_ume"]) for r in group if r["mean_ume"] != ""]
        seed = seed40.get(poster_id, {})
        group_rows.append(
            {
                "poster_id": poster_id,
                "poster_name": posters[poster_id]["poster_name"],
                "paper_text_correct_pct": seed.get("paper_text_correct_pct", ""),
                "paper_image_correct_pct": seed.get("paper_image_correct_pct", ""),
                "paper_image_minus_text_correct_pct": seed.get("paper_image_minus_text_correct_pct", ""),
                "seed40_text_mean_ume": seed.get("text_mean_ume", ""),
                "repeat_count": len(group),
                "mean_text_ume": mean(mean_umes) if mean_umes else "",
                "sd_text_ume": pstdev(mean_umes) if len(mean_umes) > 1 else 0.0,
                "min_text_ume": min(mean_umes) if mean_umes else "",
                "max_text_ume": max(mean_umes) if mean_umes else "",
                "mean_pairwise_content_jaccard": mean(jaccards) if jaccards else "",
                "min_pairwise_content_jaccard": min(jaccards) if jaccards else "",
                "hedge_hits": sum(int(r["hedge_hit"]) for r in group),
                "mean_visual_term_hits": mean(float(r["visual_term_hits"]) for r in group),
                "mean_content_word_count": mean(float(r["content_word_count"]) for r in group),
            }
        )

    write_csv(out_dir / "text_repeat_sample_summary.csv", sample_rows)
    write_csv(out_dir / "text_repeat_group_summary.csv", group_rows)
    write_csv(out_dir / "high_entropy_text_repeat_tokens.csv", high_rows)

    report = [
        "# Modal Aphasia Text-from-memory Repeat Stability Report",
        "",
        f"run: `{run_dir}`",
        "",
        "## Aggregate",
        "",
    ]
    for field in ("mean_text_ume", "sd_text_ume", "mean_pairwise_content_jaccard", "hedge_hits"):
        vals = [float(r[field]) for r in group_rows if r[field] not in ("", None)]
        report.append(f"- mean {field}: {mean(vals) if vals else None}")
    report.extend(
        [
            "",
            "## Correlations",
            "",
            f"- corr(mean_text_ume, paper_image_minus_text_correct_pct): {pearson([r['mean_text_ume'] for r in group_rows], [r['paper_image_minus_text_correct_pct'] for r in group_rows])}",
            f"- corr(sd_text_ume, paper_image_minus_text_correct_pct): {pearson([r['sd_text_ume'] for r in group_rows], [r['paper_image_minus_text_correct_pct'] for r in group_rows])}",
            f"- corr(mean_pairwise_content_jaccard, paper_image_minus_text_correct_pct): {pearson([r['mean_pairwise_content_jaccard'] for r in group_rows], [r['paper_image_minus_text_correct_pct'] for r in group_rows])}",
            "",
            "## Per Poster",
            "",
        ]
    )
    for r in group_rows:
        report.append(
            f"- {r['poster_name']}: mean UME={float(r['mean_text_ume']):.4f}, "
            f"sd UME={float(r['sd_text_ume']):.4f}, "
            f"content Jaccard={float(r['mean_pairwise_content_jaccard']):.4f}, "
            f"paper gap={float(r['paper_image_minus_text_correct_pct']):.1f}"
        )
    (out_dir / "text_repeat_stability_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"[INFO] wrote {out_dir}")


if __name__ == "__main__":
    main()
