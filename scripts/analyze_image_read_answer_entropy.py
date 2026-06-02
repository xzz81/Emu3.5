#!/usr/bin/env python3
"""Summarize text-token entropy for image-reading A/B prompts."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


COLORS = {"red", "blue", "purple", "yellow", "green", "black", "white", "gray", "grey"}
OBJECTS = {
    "cube",
    "cubes",
    "button",
    "buttons",
    "disk",
    "disc",
    "circle",
    "circular",
    "triangle",
    "triangular",
    "shape",
    "object",
    "form",
}
SPATIAL = {
    "left",
    "right",
    "upper",
    "lower",
    "top",
    "bottom",
    "center",
    "centre",
    "corner",
    "side",
    "foreground",
    "background",
    "surface",
    "near",
    "on",
}
HEDGES = {"appears", "seems", "likely", "might", "possibly", "perhaps", "suggests", "unclear"}
COUNTS = {"one", "single", "two", "three", "four", "1", "2", "3", "4"}
PROMPT_SUFFIXES = {
    "__describe": "describe",
    "__imageonly": "imageonly",
    "__describe_short": "describe_short",
    "__caption": "caption",
    "__whatin": "whatin",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-k", type=int, default=8)
    return parser.parse_args()


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


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def read_text(path: Path):
    return path.read_text(encoding="utf-8")


def clean_token(text: str):
    text = re.sub(r"<\|[^>]+?\|>", " ", text)
    words = re.findall(r"[a-z0-9]+", text.lower())
    return words


def token_category(text: str):
    words = clean_token(text)
    if not words:
        if re.search(r"[.!?,;:]", text):
            return "boundary"
        return "special_or_space"
    wordset = set(words)
    if wordset & COLORS:
        return "color"
    if wordset & OBJECTS:
        return "object"
    if wordset & SPATIAL:
        return "spatial"
    if wordset & HEDGES:
        return "hedge"
    if wordset & COUNTS:
        return "count"
    return "other_content"


def mode_from_sample(sample_id: str):
    if "__objectname" in sample_id:
        return "objectname"
    for suffix, mode in PROMPT_SUFFIXES.items():
        if sample_id.endswith(suffix):
            return mode
    if sample_id.endswith("__locationsentence"):
        return "locationsentence"
    if sample_id.endswith("__labelsentence"):
        return "labelsentence"
    if sample_id.endswith("__forcedposition"):
        return "forcedposition"
    if sample_id.endswith("__spatialdescribe"):
        return "spatialdescribe"
    return "unknown"


def base_id(sample_id: str):
    sample_id = re.sub(r"__objectname(?:__[^_][A-Za-z0-9_-]*)?$", "", sample_id)
    base = (
        sample_id
        .replace("__locationsentence", "")
        .replace("__labelsentence", "")
        .replace("__forcedposition", "")
        .replace("__spatialdescribe", "")
    )
    for suffix in PROMPT_SUFFIXES:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    return re.sub(r"_rep\d+$", "", base)


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def as_float(value):
    if value in ("", None):
        return 0.0
    return float(value)


def summarize_sample(run_dir: Path, trace_path: Path, top_k: int):
    sample_id = trace_path.name.replace("_entropy.jsonl", "")
    rows = list(read_jsonl(trace_path))
    raw_path = run_dir / "raw_generations" / f"{sample_id}.txt"
    raw_text = read_text(raw_path) if raw_path.exists() else ""
    non_structure = [r for r in rows if r.get("token_type") != "structure"]
    text_like = [r for r in rows if r.get("token_type") in {"text", "thinking"}]
    category_rows = []
    token_rows = []
    for r in text_like:
        cat = token_category(str(r.get("token_text", "")))
        token_rows.append(
            {
                "sample_id": sample_id,
                "base_id": base_id(sample_id),
                "mode": mode_from_sample(sample_id),
                "step": r["step"],
                "token_type": r["token_type"],
                "category": cat,
                "token_text": r["token_text"],
                "u_tok": r["u_tok"],
                "u_intra": r["u_intra"],
                "ume": r["ume"],
                "candidate_count": r.get("candidate_count", ""),
            }
        )
    categories = sorted({r["category"] for r in token_rows})
    for cat in categories:
        group = [r for r in token_rows if r["category"] == cat]
        category_rows.append(
            {
                "sample_id": sample_id,
                "base_id": base_id(sample_id),
                "mode": mode_from_sample(sample_id),
                "category": cat,
                "token_count": len(group),
                "mean_u_tok": mean(float(r["u_tok"]) for r in group),
                "mean_ume": mean(float(r["ume"]) for r in group),
                "max_ume": max(float(r["ume"]) for r in group),
            }
        )

    high_rows = sorted(token_rows, key=lambda r: float(r["ume"]), reverse=True)[:top_k]
    first = text_like[0] if text_like else {}
    sample_summary = {
        "sample_id": sample_id,
        "base_id": base_id(sample_id),
        "mode": mode_from_sample(sample_id),
        "total_tokens": len(rows),
        "text_tokens": sum(1 for r in rows if r.get("token_type") == "text"),
        "thinking_tokens": sum(1 for r in rows if r.get("token_type") == "thinking"),
        "structure_tokens": sum(1 for r in rows if r.get("token_type") == "structure"),
        "mean_u_tok_non_structure": mean(float(r["u_tok"]) for r in non_structure),
        "mean_ume_non_structure": mean(float(r["ume"]) for r in non_structure),
        "max_ume_non_structure": max(float(r["ume"]) for r in non_structure) if non_structure else 0.0,
        "first_token": first.get("token_text", ""),
        "first_token_u_tok": first.get("u_tok", ""),
        "first_token_ume": first.get("ume", ""),
        "object_token_count": sum(1 for r in token_rows if r["category"] == "object"),
        "object_mean_ume": mean(float(r["ume"]) for r in token_rows if r["category"] == "object"),
        "color_token_count": sum(1 for r in token_rows if r["category"] == "color"),
        "color_mean_ume": mean(float(r["ume"]) for r in token_rows if r["category"] == "color"),
        "spatial_token_count": sum(1 for r in token_rows if r["category"] == "spatial"),
        "spatial_mean_ume": mean(float(r["ume"]) for r in token_rows if r["category"] == "spatial"),
        "hedge_token_count": sum(1 for r in token_rows if r["category"] == "hedge"),
        "hedge_mean_ume": mean(float(r["ume"]) for r in token_rows if r["category"] == "hedge"),
        "raw_text": raw_text.replace("\n", "\\n"),
    }
    return sample_summary, category_rows, token_rows, high_rows


def write_markdown(path: Path, sample_rows, pair_rows, high_rows):
    lines = [
        "# Image Read Answer Entropy Report",
        "",
        "This report compares two image-reading input forms: image + `Describe this image carefully.` and image-only chat input.",
        "",
        "## Mode Summary",
        "",
    ]
    modes = sorted({r["mode"] for r in sample_rows})
    preferred = [
        m
        for m in (
            "describe",
            "spatialdescribe",
            "objectname",
            "locationsentence",
            "labelsentence",
            "forcedposition",
            "imageonly",
        )
        if m in modes
    ]
    for mode in preferred + [m for m in modes if m not in preferred]:
        group = [r for r in sample_rows if r["mode"] == mode]
        lines.append(f"### {mode}")
        lines.append("")
        lines.append(f"- Samples: {len(group)}")
        lines.append(f"- Mean non-structure UME: {mean(float(r['mean_ume_non_structure']) for r in group):.4f}")
        lines.append(f"- Mean first-token UME: {mean(as_float(r['first_token_ume']) for r in group):.4f}")
        lines.append(f"- Mean object-token UME: {mean(float(r['object_mean_ume']) for r in group if int(r['object_token_count']) > 0):.4f}")
        lines.append(f"- Mean color-token UME: {mean(float(r['color_mean_ume']) for r in group if int(r['color_token_count']) > 0):.4f}")
        lines.append(f"- Mean spatial-token UME: {mean(float(r['spatial_mean_ume']) for r in group if int(r['spatial_token_count']) > 0):.4f}")
        lines.append("")

    lines.extend(["## Paired Describe vs Image-only", ""])
    for row in pair_rows:
        lines.append(
            f"- `{row['base_id']}`: mean UME describe={float(row['describe_mean_ume']):.4f}, "
            f"imageonly={float(row['imageonly_mean_ume']):.4f}, delta={float(row['imageonly_minus_describe_mean_ume']):+.4f}; "
            f"first-token delta={float(row['imageonly_minus_describe_first_ume']):+.4f}"
        )
    lines.extend(["", "## Highest-Entropy Tokens", ""])
    for row in high_rows[:40]:
        lines.append(
            f"- `{row['sample_id']}` step {row['step']} {row['category']}: "
            f"`{str(row['token_text']).replace('`', '')}` UME={float(row['ume']):.4f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    sample_rows = []
    category_rows = []
    token_rows = []
    high_rows = []
    for trace_path in sorted((run_dir / "entropy_traces").glob("*_entropy.jsonl")):
        sample, categories, tokens, high = summarize_sample(run_dir, trace_path, args.top_k)
        sample_rows.append(sample)
        category_rows.extend(categories)
        token_rows.extend(tokens)
        high_rows.extend(high)

    by_pair = {}
    for row in sample_rows:
        by_pair.setdefault(row["base_id"], {})[row["mode"]] = row
    pair_rows = []
    for bid, pair in sorted(by_pair.items()):
        if "describe" not in pair or "imageonly" not in pair:
            continue
        d = pair["describe"]
        i = pair["imageonly"]
        pair_rows.append(
            {
                "base_id": bid,
                "describe_mean_ume": d["mean_ume_non_structure"],
                "imageonly_mean_ume": i["mean_ume_non_structure"],
                "imageonly_minus_describe_mean_ume": float(i["mean_ume_non_structure"]) - float(d["mean_ume_non_structure"]),
                "describe_first_ume": d["first_token_ume"],
                "imageonly_first_ume": i["first_token_ume"],
                "imageonly_minus_describe_first_ume": as_float(i["first_token_ume"]) - as_float(d["first_token_ume"]),
                "describe_text_tokens": d["text_tokens"],
                "imageonly_text_tokens": i["text_tokens"],
                "describe_thinking_tokens": d["thinking_tokens"],
                "imageonly_thinking_tokens": i["thinking_tokens"],
            }
        )

    high_rows = sorted(high_rows, key=lambda r: float(r["ume"]), reverse=True)
    write_csv(sample_rows, out_dir / "sample_answer_entropy_summary.csv")
    write_csv(category_rows, out_dir / "category_answer_entropy_summary.csv")
    write_csv(token_rows, out_dir / "token_answer_entropy_rows.csv")
    write_csv(high_rows, out_dir / "high_entropy_answer_tokens.csv")
    write_csv(pair_rows, out_dir / "paired_mode_entropy_shift.csv")
    write_markdown(out_dir / "image_read_answer_entropy_report.md", sample_rows, pair_rows, high_rows)
    print(f"[INFO] wrote {len(sample_rows)} sample rows, {len(token_rows)} token rows to {out_dir}")


if __name__ == "__main__":
    main()
