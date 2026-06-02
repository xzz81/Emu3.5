#!/usr/bin/env python3
# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Summarize Modal Aphasia poster text/image entropy traces."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean


POSTER_ID_RE = re.compile(r"poster_(\d+)_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-run-dir", required=True)
    parser.add_argument("--image-run-dir", required=True)
    parser.add_argument(
        "--poster-data",
        default="/workspace/home/AAAI 2027/modal-aphasia/misc/real_world_data/posters-1.json",
    )
    parser.add_argument(
        "--out-dir",
        default="/workspace/home/AAAI 2027/research_logs/modal_aphasia_posters",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def poster_id_from_sample(sample_id: str) -> str:
    match = POSTER_ID_RE.search(sample_id)
    if not match:
        return sample_id
    return str(int(match.group(1)))


def p90(values: list[float]) -> float | None:
    if not values:
        return None
    values = sorted(values)
    idx = min(len(values) - 1, math.ceil(0.9 * len(values)) - 1)
    return values[idx]


def summarize_records(records: list[dict], token_type: str) -> dict:
    rows = [r for r in records if r.get("token_type") == token_type]
    if not rows:
        return {
            "count": 0,
            "mean_ume": "",
            "p90_ume": "",
            "mean_ume_full": "",
            "p90_ume_full": "",
            "mean_u_tok": "",
            "mean_u_intra": "",
            "mean_u_cfg": "",
        }
    umes = [float(r.get("ume", 0.0)) for r in rows]
    ume_full = [float(r.get("ume_full", r.get("ume", 0.0))) for r in rows]
    return {
        "count": len(rows),
        "mean_ume": mean(umes),
        "p90_ume": p90(umes),
        "mean_ume_full": mean(ume_full),
        "p90_ume_full": p90(ume_full),
        "mean_u_tok": mean(float(r.get("u_tok", 0.0)) for r in rows),
        "mean_u_intra": mean(float(r.get("u_intra", 0.0)) for r in rows),
        "mean_u_cfg": mean(float(r.get("u_cfg", 0.0)) for r in rows),
    }


def load_run(run_dir: Path) -> dict[str, list[dict]]:
    trace_dir = run_dir / "entropy_traces"
    grouped: dict[str, list[dict]] = {}
    for path in sorted(trace_dir.glob("*_entropy.jsonl")):
        records = read_jsonl(path)
        if not records:
            continue
        sample_id = str(records[0].get("sample_id", path.stem.replace("_entropy", "")))
        grouped[poster_id_from_sample(sample_id)] = records
    return grouped


def load_posters(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))["posters"]
    return {str(int(k)): v for k, v in data.items()}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def pearson(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys) if x != "" and y != ""]
    if len(pairs) < 2:
        return None
    xs = [float(x) for x, _ in pairs]
    ys = [float(y) for _, y in pairs]
    mx = mean(xs)
    my = mean(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    denom = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    if denom <= 1e-12:
        return None
    return sum(x * y for x, y in zip(dx, dy)) / denom


def top_text_tokens(records: list[dict], poster: dict, limit: int = 12) -> list[dict]:
    rows = [r for r in records if r.get("token_type") == "text"]
    rows.sort(key=lambda r: float(r.get("ume", 0.0)), reverse=True)
    out = []
    for r in rows[:limit]:
        out.append(
            {
                "poster_id": poster["poster_id"],
                "poster_name": poster["poster_name"],
                "step": r.get("step"),
                "token_text": r.get("token_text"),
                "ume": r.get("ume"),
                "u_tok": r.get("u_tok"),
                "u_intra": r.get("u_intra"),
                "u_cfg": r.get("u_cfg"),
            }
        )
    return out


def main() -> None:
    args = parse_args()
    text_run = Path(args.text_run_dir)
    image_run = Path(args.image_run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    posters = load_posters(Path(args.poster_data))
    text_records = load_run(text_run)
    image_records = load_run(image_run)

    rows = []
    high_entropy_text_rows = []
    for poster_id, poster in sorted(posters.items(), key=lambda item: int(item[0])):
        text_summary = summarize_records(text_records.get(poster_id, []), "text")
        image_visual_summary = summarize_records(image_records.get(poster_id, []), "visual")
        image_text_summary = summarize_records(image_records.get(poster_id, []), "text")
        paper_summary = poster.get("summary", {})
        row = {
            "poster_id": poster_id,
            "poster_name": poster["poster_name"],
            "paper_text_correct_pct": paper_summary.get("text_correct_pct", ""),
            "paper_image_correct_pct": paper_summary.get("image_correct_pct", ""),
            "paper_image_minus_text_correct_pct": (
                paper_summary.get("image_correct_pct", 0) - paper_summary.get("text_correct_pct", 0)
                if paper_summary
                else ""
            ),
            "text_tokens": text_summary["count"],
            "text_mean_ume": text_summary["mean_ume"],
            "text_p90_ume": text_summary["p90_ume"],
            "text_mean_u_tok": text_summary["mean_u_tok"],
            "image_visual_tokens": image_visual_summary["count"],
            "image_visual_mean_ume": image_visual_summary["mean_ume"],
            "image_visual_p90_ume": image_visual_summary["p90_ume"],
            "image_visual_mean_ume_full": image_visual_summary["mean_ume_full"],
            "image_visual_p90_ume_full": image_visual_summary["p90_ume_full"],
            "image_visual_mean_u_cfg": image_visual_summary["mean_u_cfg"],
            "image_preamble_text_tokens": image_text_summary["count"],
            "image_preamble_text_mean_ume": image_text_summary["mean_ume"],
        }
        if row["text_mean_ume"] != "" and row["image_visual_mean_ume_full"] != "":
            row["image_visual_minus_text_mean_ume"] = (
                float(row["image_visual_mean_ume_full"]) - float(row["text_mean_ume"])
            )
        else:
            row["image_visual_minus_text_mean_ume"] = ""
        rows.append(row)
        if poster_id in text_records:
            high_entropy_text_rows.extend(top_text_tokens(text_records[poster_id], poster))

    write_csv(out_dir / "poster_entropy_summary.csv", rows)
    write_csv(out_dir / "high_entropy_text_tokens.csv", high_entropy_text_rows)

    text_entropy = [r["text_mean_ume"] for r in rows]
    image_entropy = [r["image_visual_mean_ume_full"] for r in rows]
    paper_gap = [r["paper_image_minus_text_correct_pct"] for r in rows]
    report = [
        "# Modal Aphasia Poster Entropy Report",
        "",
        f"text run: `{text_run}`",
        f"image run: `{image_run}`",
        "",
        "## Correlations",
        "",
        f"- corr(text_mean_ume, paper_image_minus_text_correct_pct): {pearson(text_entropy, paper_gap)}",
        f"- corr(image_visual_mean_ume_full, paper_image_minus_text_correct_pct): {pearson(image_entropy, paper_gap)}",
        f"- corr(text_mean_ume, image_visual_mean_ume_full): {pearson(text_entropy, image_entropy)}",
        "",
        "## Interpretation Checklist",
        "",
        "1. High text entropy on object/attribute words suggests the model is unsure while verbalizing poster memory.",
        "2. Low text entropy with visibly wrong description is the strongest Modal Aphasia false-confidence pattern.",
        "3. Low visual entropy with good image generation suggests image memory is easier to externalize than text memory.",
        "4. High visual `u_cfg` or high visual UME suggests poster generation depends on prompt/guidance pressure or unstable visual details.",
        "",
        "See `poster_entropy_summary.csv` and `high_entropy_text_tokens.csv` for per-poster details.",
    ]
    (out_dir / "poster_entropy_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"[INFO] wrote {out_dir}")


if __name__ == "__main__":
    main()
