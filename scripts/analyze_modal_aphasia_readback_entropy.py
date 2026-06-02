#!/usr/bin/env python3
"""Summarize generated-poster readback entropy for Setting F."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean


POSTER_ID_RE = re.compile(r"poster_(\d+)_")
ACTION = {"finish", "task", "step", "move", "press", "release", "grasp", "lift", "place", "align", "done"}
HEDGES = {"appears", "seems", "likely", "might", "possibly", "perhaps", "looks", "resembles", "unclear"}
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
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--readback-run-dir", required=True)
    parser.add_argument(
        "--poster-data",
        default="/workspace/home/AAAI 2027/modal-aphasia/misc/real_world_data/posters-1.json",
    )
    parser.add_argument(
        "--seed40-summary-csv",
        default="/workspace/home/AAAI 2027/research_logs/modal_aphasia_posters/seed40/poster_entropy_summary.csv",
    )
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def poster_id_from_sample(sample_id: str) -> str:
    match = POSTER_ID_RE.search(sample_id)
    return str(int(match.group(1))) if match else sample_id


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


def words(text: str) -> list[str]:
    text = re.sub(r"<\|[^>]+?\|>", " ", text.lower())
    return re.findall(r"[a-z0-9]+", text)


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
    u_toks = [float(r.get("u_tok", 0.0)) for r in text]
    return {
        "readback_tokens": len(text),
        "readback_mean_ume": mean(umes) if umes else "",
        "readback_p90_ume": p90(umes),
        "readback_mean_u_tok": mean(u_toks) if u_toks else "",
        "readback_first_token_ume": float(text[0].get("ume", 0.0)) if text else "",
        "readback_max_ume": max(umes) if umes else "",
    }


def top_tokens(records: list[dict], poster: dict, limit: int = 12) -> list[dict]:
    text = [r for r in records if r.get("token_type") in {"text", "thinking"}]
    text.sort(key=lambda r: float(r.get("ume", 0.0)), reverse=True)
    out = []
    for r in text[:limit]:
        out.append(
            {
                "poster_id": poster["poster_id"],
                "poster_name": poster["poster_name"],
                "step": r.get("step"),
                "token_text": r.get("token_text"),
                "ume": r.get("ume"),
                "u_tok": r.get("u_tok"),
                "u_intra": r.get("u_intra"),
            }
        )
    return out


def main() -> None:
    args = parse_args()
    run_dir = Path(args.readback_run_dir)
    out_dir = Path(args.out_dir)
    posters = load_posters(Path(args.poster_data))
    seed40 = load_seed40(Path(args.seed40_summary_csv))

    trace_by_poster = {}
    for path in sorted((run_dir / "entropy_traces").glob("*_entropy.jsonl")):
        records = read_jsonl(path)
        if not records:
            continue
        sample_id = str(records[0].get("sample_id", path.stem.replace("_entropy", "")))
        trace_by_poster[poster_id_from_sample(sample_id)] = records

    rows = []
    high_rows = []
    for poster_id, poster in sorted(posters.items(), key=lambda item: int(item[0])):
        records = trace_by_poster.get(poster_id, [])
        raw_path = next((run_dir / "raw_generations").glob(f"poster_{int(poster_id):02d}_*__readback.txt"), None)
        raw_text = raw_path.read_text(encoding="utf-8") if raw_path else ""
        ws = set(words(raw_text))
        base = seed40.get(poster_id, {})
        summary = summarize_trace(records)
        row = {
            "poster_id": poster_id,
            "poster_name": poster["poster_name"],
            "paper_text_correct_pct": base.get("paper_text_correct_pct", ""),
            "paper_image_correct_pct": base.get("paper_image_correct_pct", ""),
            "paper_image_minus_text_correct_pct": base.get("paper_image_minus_text_correct_pct", ""),
            "text_memory_mean_ume": base.get("text_mean_ume", ""),
            "image_visual_mean_ume_full": base.get("image_visual_mean_ume_full", ""),
            "image_visual_mean_u_cfg": base.get("image_visual_mean_u_cfg", ""),
            **summary,
            "readback_minus_text_memory_ume": (
                float(summary["readback_mean_ume"]) - float(base["text_mean_ume"])
                if summary["readback_mean_ume"] != "" and base.get("text_mean_ume") not in ("", None)
                else ""
            ),
            "readback_minus_image_visual_ume_full": (
                float(summary["readback_mean_ume"]) - float(base["image_visual_mean_ume_full"])
                if summary["readback_mean_ume"] != "" and base.get("image_visual_mean_ume_full") not in ("", None)
                else ""
            ),
            "action_script": int(bool(ws & ACTION)),
            "hedge_hit": int(bool(ws & HEDGES)),
            "visual_term_hits": len(ws & VISUAL_TERMS),
            "raw_text": raw_text.replace("\n", "\\n")[:800],
        }
        rows.append(row)
        if records:
            high_rows.extend(top_tokens(records, poster))

    write_csv(out_dir / "poster_readback_entropy_summary.csv", rows)
    write_csv(out_dir / "high_entropy_readback_tokens.csv", high_rows)

    report = [
        "# Modal Aphasia Generated-poster Readback Entropy Report",
        "",
        f"readback run: `{run_dir}`",
        "",
        "## Aggregate",
        "",
    ]
    numeric_fields = [
        "readback_mean_ume",
        "text_memory_mean_ume",
        "image_visual_mean_ume_full",
        "readback_minus_text_memory_ume",
        "readback_minus_image_visual_ume_full",
    ]
    for field in numeric_fields:
        vals = [float(r[field]) for r in rows if r[field] not in ("", None)]
        report.append(f"- mean {field}: {mean(vals) if vals else None}")
    report.extend(
        [
            f"- action scripts: {sum(int(r['action_script']) for r in rows)}/{len(rows)}",
            f"- hedge hits: {sum(int(r['hedge_hit']) for r in rows)}/{len(rows)}",
            "",
            "## Correlations",
            "",
            f"- corr(readback_mean_ume, paper_image_minus_text_correct_pct): {pearson([r['readback_mean_ume'] for r in rows], [r['paper_image_minus_text_correct_pct'] for r in rows])}",
            f"- corr(readback_mean_ume, text_memory_mean_ume): {pearson([r['readback_mean_ume'] for r in rows], [r['text_memory_mean_ume'] for r in rows])}",
            f"- corr(readback_mean_ume, image_visual_mean_ume_full): {pearson([r['readback_mean_ume'] for r in rows], [r['image_visual_mean_ume_full'] for r in rows])}",
            "",
            "## Per Poster",
            "",
        ]
    )
    for row in rows:
        report.append(
            f"- {row['poster_name']}: readback UME={float(row['readback_mean_ume']):.4f}, "
            f"text-memory UME={float(row['text_memory_mean_ume']):.4f}, "
            f"image UME_full={float(row['image_visual_mean_ume_full']):.4f}, "
            f"action={row['action_script']}, hedges={row['hedge_hit']}"
        )
    (out_dir / "poster_readback_entropy_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"[INFO] wrote {out_dir}")


if __name__ == "__main__":
    main()
