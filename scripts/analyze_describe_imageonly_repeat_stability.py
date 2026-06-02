#!/usr/bin/env python3
"""Aggregate repeated describe vs image-only image-read answer entropy runs."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-csv", required=True)
    parser.add_argument("--token-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        yield from csv.DictReader(f)


def write_csv(rows, path: Path):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def f(row, key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else 0.0


def i(row, key: str) -> int:
    value = row.get(key, "")
    return int(value) if value not in ("", None) else 0


def normalize_answer(text: str) -> str:
    text = re.sub(r"<\|[^>]+?\|>", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def compact_text(text: str, limit: int = 180) -> str:
    text = re.sub(r"\s+", " ", text.replace("\\n", " ")).strip()
    return text[:limit]


def mode_summary(rows):
    out = []
    for mode in sorted({r["mode"] for r in rows}):
        group = [r for r in rows if r["mode"] == mode]
        out.append(
            {
                "mode": mode,
                "sample_count": len(group),
                "mean_answer_ume": mean(f(r, "mean_answer_ume") for r in group),
                "mean_first_token_ume": mean(f(r, "first_token_ume") for r in group),
                "mean_text_like_tokens": mean(f(r, "text_like_tokens") for r in group),
                "target_exact_hits": sum(i(r, "target_exact_phrase_hit") for r in group),
                "target_loose_hits": sum(i(r, "target_loose_hit") for r in group),
                "prompt_object_hits": sum(i(r, "prompt_object_hit") for r in group),
                "action_scripts": sum(i(r, "action_script") for r in group),
                "hedge_hits": sum(i(r, "hedge_hit") for r in group),
                "no_answers": sum(i(r, "no_answer") for r in group),
                "mean_target_object_ume": mean(f(r, "target_object_mean_ume") for r in group),
                "mean_prompt_object_ume": mean(f(r, "prompt_object_mean_ume") for r in group),
                "mean_color_ume": mean(f(r, "color_mean_ume") for r in group if i(r, "color_token_count") > 0),
                "mean_spatial_ume": mean(f(r, "spatial_mean_ume") for r in group if i(r, "spatial_token_count") > 0),
            }
        )
    return out


def base_mode_summary(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["base_id"], row["mode"]), []).append(row)
    out = []
    for (base_id, mode), group in sorted(grouped.items()):
        answers = [normalize_answer(r.get("raw_text", "")) for r in group]
        action = [r for r in group if i(r, "action_script")]
        no_answer = [r for r in group if i(r, "no_answer")]
        loose = sum(i(r, "target_loose_hit") for r in group)
        out.append(
            {
                "base_id": base_id,
                "mode": mode,
                "repeat_count": len(group),
                "unique_answers": len(set(answers)),
                "mean_answer_ume": mean(f(r, "mean_answer_ume") for r in group),
                "mean_first_token_ume": mean(f(r, "first_token_ume") for r in group),
                "mean_text_like_tokens": mean(f(r, "text_like_tokens") for r in group),
                "target_loose_hits": loose,
                "target_loose_misses": len(group) - loose,
                "action_scripts": len(action),
                "hedge_hits": sum(i(r, "hedge_hit") for r in group),
                "no_answers": len(no_answer),
                "example_answer": compact_text(group[0].get("raw_text", "")),
            }
        )
    return out


def paired_summary(base_rows):
    by_base = {}
    for row in base_rows:
        by_base.setdefault(row["base_id"], {})[row["mode"]] = row
    out = []
    for base_id, modes in sorted(by_base.items()):
        if "describe" not in modes or "imageonly" not in modes:
            continue
        d = modes["describe"]
        im = modes["imageonly"]
        out.append(
            {
                "base_id": base_id,
                "describe_mean_ume": d["mean_answer_ume"],
                "imageonly_mean_ume": im["mean_answer_ume"],
                "imageonly_minus_describe_mean_ume": float(im["mean_answer_ume"]) - float(d["mean_answer_ume"]),
                "describe_first_token_ume": d["mean_first_token_ume"],
                "imageonly_first_token_ume": im["mean_first_token_ume"],
                "imageonly_minus_describe_first_token_ume": float(im["mean_first_token_ume"]) - float(d["mean_first_token_ume"]),
                "describe_loose_hits": d["target_loose_hits"],
                "imageonly_loose_hits": im["target_loose_hits"],
                "describe_action_scripts": d["action_scripts"],
                "imageonly_action_scripts": im["action_scripts"],
                "describe_no_answers": d["no_answers"],
                "imageonly_no_answers": im["no_answers"],
                "describe_unique_answers": d["unique_answers"],
                "imageonly_unique_answers": im["unique_answers"],
            }
        )
    return out


def category_mode_summary(token_rows):
    grouped = {}
    for row in token_rows:
        grouped.setdefault((row["mode"], row["category"]), []).append(row)
    out = []
    for (mode, category), group in sorted(grouped.items()):
        out.append(
            {
                "mode": mode,
                "category": category,
                "token_count": len(group),
                "mean_ume": mean(f(r, "ume") for r in group),
                "max_ume": max(f(r, "ume") for r in group),
                "mean_u_tok": mean(f(r, "u_tok") for r in group),
            }
        )
    return out


def write_report(path: Path, mode_rows, base_rows, pair_rows, category_rows, sample_rows):
    lines = [
        "# Describe vs Image-only Repeat Stability",
        "",
        "This report aggregates repeated generated-object image-reading runs for two input forms.",
        "",
        "## Mode Summary",
        "",
    ]
    for row in mode_rows:
        lines.append(
            f"- {row['mode']}: n={row['sample_count']}, mean UME={float(row['mean_answer_ume']):.4f}, "
            f"first-token UME={float(row['mean_first_token_ume']):.4f}, loose={row['target_loose_hits']}/{row['sample_count']}, "
            f"actions={row['action_scripts']}/{row['sample_count']}, no_answer={row['no_answers']}/{row['sample_count']}, "
            f"hedges={row['hedge_hits']}/{row['sample_count']}"
        )
    lines.extend(["", "## Category UME", ""])
    for row in category_rows:
        if row["category"] in {"object", "color", "spatial", "hedge", "other_content", "boundary"}:
            lines.append(
                f"- {row['mode']} {row['category']}: tokens={row['token_count']}, "
                f"mean UME={float(row['mean_ume']):.4f}, max UME={float(row['max_ume']):.4f}"
            )
    lines.extend(["", "## Largest Mode Deltas", ""])
    for row in sorted(pair_rows, key=lambda r: float(r["imageonly_minus_describe_mean_ume"]))[:8]:
        lines.append(
            f"- lower image-only UME `{row['base_id']}`: describe={float(row['describe_mean_ume']):.4f}, "
            f"imageonly={float(row['imageonly_mean_ume']):.4f}, delta={float(row['imageonly_minus_describe_mean_ume']):+.4f}, "
            f"loose d/i={row['describe_loose_hits']}/3 vs {row['imageonly_loose_hits']}/3, "
            f"actions d/i={row['describe_action_scripts']} vs {row['imageonly_action_scripts']}"
        )
    for row in sorted(pair_rows, key=lambda r: float(r["imageonly_minus_describe_mean_ume"]), reverse=True)[:5]:
        lines.append(
            f"- higher image-only UME `{row['base_id']}`: describe={float(row['describe_mean_ume']):.4f}, "
            f"imageonly={float(row['imageonly_mean_ume']):.4f}, delta={float(row['imageonly_minus_describe_mean_ume']):+.4f}, "
            f"loose d/i={row['describe_loose_hits']}/3 vs {row['imageonly_loose_hits']}/3, "
            f"actions d/i={row['describe_action_scripts']} vs {row['imageonly_action_scripts']}"
        )
    lines.extend(["", "## Failure Examples", ""])
    failures = [r for r in sample_rows if i(r, "action_script") or i(r, "no_answer") or not i(r, "target_loose_hit")]
    for row in failures[:24]:
        lines.append(
            f"- `{row['sample_id']}` mode={row['mode']} loose={row['target_loose_hit']} "
            f"action={row['action_script']} no_answer={row['no_answer']} UME={float(row['mean_answer_ume']):.4f}: "
            f"`{compact_text(row.get('raw_text', ''))}`"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    sample_rows = list(read_csv(Path(args.sample_csv)))
    token_rows = list(read_csv(Path(args.token_csv)))
    mode_rows = mode_summary(sample_rows)
    base_rows = base_mode_summary(sample_rows)
    pair_rows = paired_summary(base_rows)
    category_rows = category_mode_summary(token_rows)
    write_csv(mode_rows, out_dir / "mode_repeat_summary.csv")
    write_csv(base_rows, out_dir / "base_mode_repeat_summary.csv")
    write_csv(pair_rows, out_dir / "paired_base_repeat_summary.csv")
    write_csv(category_rows, out_dir / "category_mode_repeat_summary.csv")
    write_report(out_dir / "describe_imageonly_repeat_stability_report.md", mode_rows, base_rows, pair_rows, category_rows, sample_rows)
    print(f"[INFO] wrote repeat stability summaries for {len(sample_rows)} samples to {out_dir}")


if __name__ == "__main__":
    main()
