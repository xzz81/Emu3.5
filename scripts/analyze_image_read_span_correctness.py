#!/usr/bin/env python3
"""Token/span-proxy correctness analysis for image-read answer entropy runs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


CHROMATIC_COLORS = {"red", "blue", "purple", "yellow", "green", "orange"}
NEUTRAL_COLORS = {"black", "white", "gray", "grey"}
COLORS = CHROMATIC_COLORS | NEUTRAL_COLORS
NOUN_ALIASES = {
    "circle": {"circle", "circular", "dot", "disc", "disk", "coin", "target", "button"},
    "square": {"square", "box", "block", "rectangle", "rectangular"},
    "triangle": {"triangle", "triangular", "pyramid"},
    "cube": {"cube", "box", "block", "polyhedron", "shape", "object"},
    "button": {"button", "circle", "circular", "dot", "target"},
    "disk": {"disk", "disc", "circle", "circular", "coin", "token"},
}
VISUAL_NOUNS = set().union(*NOUN_ALIASES.values()) | {
    "diamond",
    "pyramid",
    "shape",
    "object",
    "form",
    "mark",
}
HEDGES = {"appears", "seems", "likely", "might", "possibly", "perhaps", "suggests", "unclear", "resembles"}
ACTION_WORDS = {
    "finish",
    "task",
    "step",
    "approach",
    "move",
    "push",
    "press",
    "release",
    "align",
    "grasp",
    "lift",
    "carry",
    "retract",
    "current",
    "done",
}
POSITION_PARTS = {
    "upper-left": {"upper", "top", "left"},
    "upper-right": {"upper", "top", "right"},
    "lower-left": {"lower", "bottom", "left"},
    "lower-right": {"lower", "bottom", "right"},
}
OPPOSITE_PARTS = {
    "upper-left": {"lower", "bottom", "right"},
    "upper-right": {"lower", "bottom", "left"},
    "lower-left": {"upper", "top", "right"},
    "lower-right": {"upper", "top", "left"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


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


def words(text: str) -> list[str]:
    text = re.sub(r"<\|[^>]+?\|>", " ", text)
    return re.findall(r"[a-z0-9]+", text.lower())


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def mode_from_sample(sample_id: str) -> str:
    if sample_id.endswith("__locationsentence"):
        return "locationsentence"
    if sample_id.endswith("__labelsentence"):
        return "labelsentence"
    if sample_id.endswith("__forcedposition"):
        return "forcedposition"
    if sample_id.endswith("__spatialdescribe"):
        return "spatialdescribe"
    if sample_id.endswith("__describe"):
        return "describe"
    if sample_id.endswith("__imageonly"):
        return "imageonly"
    return "unknown"


def base_id(sample_id: str) -> str:
    return (
        sample_id.replace("__locationsentence", "")
        .replace("__labelsentence", "")
        .replace("__forcedposition", "")
        .replace("__spatialdescribe", "")
        .replace("__describe", "")
        .replace("__imageonly", "")
    )


def sample_suffix(sample_id: str, metadata: dict[str, dict[str, str]]) -> str:
    base = base_id(sample_id)
    matches = [suffix for suffix in metadata if base.endswith(suffix)]
    if not matches:
        return ""
    return max(matches, key=len)


def object_parts(object_phrase: str) -> tuple[str, str]:
    parts = object_phrase.lower().split()
    return parts[0], parts[-1]


def classify_token(token_text: str, expected_object: str, expected_position: str) -> str:
    ws = set(words(token_text))
    if not ws:
        if re.search(r"[.!?,;:]", token_text):
            return "boundary"
        return "space_or_special"

    expected_color, expected_noun = object_parts(expected_object)
    loose_nouns = NOUN_ALIASES.get(expected_noun, {expected_noun})
    position_parts = POSITION_PARTS.get(expected_position, set())
    opposite_parts = OPPOSITE_PARTS.get(expected_position, set())

    if ws & NEUTRAL_COLORS:
        return "neutral_color"
    if ws & (CHROMATIC_COLORS - {expected_color}):
        return "wrong_color"
    if expected_color in ws:
        return "correct_color"
    if expected_noun in ws:
        return "exact_object_noun"
    if ws & loose_nouns:
        return "loose_object_noun"
    if ws & (VISUAL_NOUNS - loose_nouns - {expected_noun}):
        return "wrong_or_alternate_object_noun"
    if ws & opposite_parts:
        return "wrong_position_component"
    if ws & position_parts:
        return "correct_position_component"
    if ws & HEDGES:
        return "hedge"
    if ws & ACTION_WORDS:
        return "action_boilerplate"
    return "other"


def sample_hits(raw_text: str, expected_object: str, expected_position: str) -> dict[str, int]:
    ws = set(words(raw_text))
    expected_color, expected_noun = object_parts(expected_object)
    loose_nouns = NOUN_ALIASES.get(expected_noun, {expected_noun})
    pos_parts = POSITION_PARTS.get(expected_position, set())
    opposite_parts = OPPOSITE_PARTS.get(expected_position, set())
    no_answer = int(not ws)
    return {
        "color_hit": int(expected_color in ws),
        "wrong_color_hit": int(bool(ws & (CHROMATIC_COLORS - {expected_color}))),
        "neutral_color_hit": int(bool(ws & NEUTRAL_COLORS)),
        "exact_noun_hit": int(expected_noun in ws),
        "loose_noun_hit": int(bool(ws & loose_nouns)),
        "wrong_object_hit": int(bool(ws & (VISUAL_NOUNS - loose_nouns - {expected_noun}))),
        "position_hit": int(bool(pos_parts) and len(pos_parts & ws) >= 2),
        "wrong_position_hit": int(bool(ws & opposite_parts)),
        "action_script": int(bool(ws & ACTION_WORDS)),
        "hedge_hit": int(bool(ws & HEDGES)),
        "no_answer": no_answer,
    }


def aggregate(rows, keys: list[str]):
    grouped = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        grouped.setdefault(key, []).append(row)
    out = []
    for key, group in sorted(grouped.items()):
        item = {k: v for k, v in zip(keys, key)}
        item.update(
            {
                "token_count": len(group),
                "mean_ume": mean(float(r["ume"]) for r in group),
                "max_ume": max(float(r["ume"]) for r in group),
                "mean_u_tok": mean(float(r["u_tok"]) for r in group),
            }
        )
        out.append(item)
    return out


def aggregate_sample_modes(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["mode"], []).append(row)
    out = []
    for mode, group in sorted(grouped.items()):
        out.append(
            {
                "mode": mode,
                "sample_count": len(group),
                "mean_ume": mean(float(r["mean_ume"]) for r in group),
                "max_ume": max(float(r["max_ume"]) for r in group),
                "color_hits": sum(int(r["color_hit"]) for r in group),
                "loose_noun_hits": sum(int(r["loose_noun_hit"]) for r in group),
                "exact_noun_hits": sum(int(r["exact_noun_hit"]) for r in group),
                "position_hits": sum(int(r["position_hit"]) for r in group),
                "action_scripts": sum(int(r["action_script"]) for r in group),
                "no_answers": sum(int(r["no_answer"]) for r in group),
            }
        )
    return out


def paired_sample_rows(rows):
    by_base = {}
    for row in rows:
        by_base.setdefault(row["base_id"], {})[row["mode"]] = row
    modes = sorted({row["mode"] for row in rows})
    preferred = [
        m
        for m in ("describe", "spatialdescribe", "locationsentence", "labelsentence", "forcedposition", "imageonly")
        if m in modes
    ]
    modes = preferred + [m for m in modes if m not in preferred]
    out = []
    for base, group in sorted(by_base.items()):
        first = next(iter(group.values()))
        item = {
            "base_id": base,
            "expected_object": first["expected_object"],
            "expected_position": first["expected_position"],
        }
        for mode in modes:
            row = group.get(mode)
            prefix = f"{mode}_"
            if row is None:
                item.update(
                    {
                        prefix + "mean_ume": "",
                        prefix + "color_hit": "",
                        prefix + "exact_noun_hit": "",
                        prefix + "loose_noun_hit": "",
                        prefix + "position_hit": "",
                        prefix + "wrong_position_hit": "",
                        prefix + "action_script": "",
                        prefix + "no_answer": "",
                    }
                )
            else:
                item.update(
                    {
                        prefix + "mean_ume": row["mean_ume"],
                        prefix + "color_hit": row["color_hit"],
                        prefix + "exact_noun_hit": row["exact_noun_hit"],
                        prefix + "loose_noun_hit": row["loose_noun_hit"],
                        prefix + "position_hit": row["position_hit"],
                        prefix + "wrong_position_hit": row["wrong_position_hit"],
                        prefix + "action_script": row["action_script"],
                        prefix + "no_answer": row["no_answer"],
                    }
                )
        out.append(item)
    return out


def write_report(path: Path, sample_rows, token_label_rows, label_summary, mode_label_summary):
    lines = [
        "# Image-read Span Correctness Entropy Report",
        "",
        "This is a token-level span proxy: generated tokens are labeled by expected color, object noun, position component, action-script boilerplate, hedge, boundary, or other.",
        "",
        "## Sample Behavior",
        "",
    ]
    modes = sorted({r["mode"] for r in sample_rows})
    preferred = [
        m
        for m in ("describe", "spatialdescribe", "locationsentence", "labelsentence", "forcedposition", "imageonly")
        if m in modes
    ]
    for mode in preferred + [m for m in modes if m not in preferred]:
        group = [r for r in sample_rows if r["mode"] == mode]
        lines.append(f"### {mode}")
        lines.append("")
        lines.append(f"- Samples: {len(group)}")
        for field in (
            "color_hit",
            "wrong_color_hit",
            "neutral_color_hit",
            "exact_noun_hit",
            "loose_noun_hit",
            "wrong_object_hit",
            "position_hit",
            "wrong_position_hit",
            "action_script",
            "hedge_hit",
            "no_answer",
        ):
            lines.append(f"- {field}: {sum(int(r[field]) for r in group)}/{len(group)}")
        lines.append(f"- Mean answer UME: {mean(float(r['mean_ume']) for r in group):.4f}")
        lines.append("")

    lines.extend(["## Label Summary", ""])
    for row in label_summary:
        lines.append(f"- {row['span_label']}: n={row['token_count']}, mean UME={float(row['mean_ume']):.4f}, max UME={float(row['max_ume']):.4f}")

    lines.extend(["", "## Mode x Label Summary", ""])
    for row in mode_label_summary:
        lines.append(
            f"- {row['mode']} / {row['span_label']}: n={row['token_count']}, "
            f"mean UME={float(row['mean_ume']):.4f}"
        )

    high = sorted(token_label_rows, key=lambda r: float(r["ume"]), reverse=True)[:40]
    lines.extend(["", "## Highest-Entropy Labeled Tokens", ""])
    for row in high:
        token = str(row["token_text"]).replace("`", "")
        lines.append(
            f"- `{row['sample_id']}` step {row['step']} {row['span_label']}: `{token}` UME={float(row['ume']):.4f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    metadata = {row["sample_suffix"]: row for row in read_csv(Path(args.metadata_csv))}

    sample_rows = []
    token_label_rows = []
    for trace_path in sorted((run_dir / "entropy_traces").glob("*_entropy.jsonl")):
        sample_id = trace_path.name.replace("_entropy.jsonl", "")
        suffix = sample_suffix(sample_id, metadata)
        if not suffix:
            continue
        meta = metadata[suffix]
        expected_object = meta["object"]
        expected_position = meta["target_position"]
        raw_path = run_dir / "raw_generations" / f"{sample_id}.txt"
        raw_text = raw_path.read_text(encoding="utf-8") if raw_path.exists() else ""
        rows = list(read_jsonl(trace_path))
        text_like = [r for r in rows if r.get("token_type") in {"text", "thinking"}]
        labeled = []
        for r in text_like:
            label = classify_token(str(r.get("token_text", "")), expected_object, expected_position)
            row = {
                "sample_id": sample_id,
                "base_id": base_id(sample_id),
                "mode": mode_from_sample(sample_id),
                "sample_suffix": suffix,
                "expected_object": expected_object,
                "expected_position": expected_position,
                "step": r["step"],
                "token_type": r["token_type"],
                "token_text": r["token_text"],
                "span_label": label,
                "u_tok": r["u_tok"],
                "ume": r["ume"],
            }
            labeled.append(row)
        token_label_rows.extend(labeled)
        hits = sample_hits(raw_text, expected_object, expected_position)
        sample_rows.append(
            {
                "sample_id": sample_id,
                "base_id": base_id(sample_id),
                "mode": mode_from_sample(sample_id),
                "sample_suffix": suffix,
                "expected_object": expected_object,
                "expected_position": expected_position,
                "text_like_tokens": len(text_like),
                "mean_ume": mean(float(r["ume"]) for r in text_like),
                "max_ume": max((float(r["ume"]) for r in text_like), default=0.0),
                **hits,
                "raw_text": raw_text.replace("\n", "\\n"),
            }
        )

    label_summary = aggregate(token_label_rows, ["span_label"])
    mode_label_summary = aggregate(token_label_rows, ["mode", "span_label"])
    sample_mode_summary = aggregate_sample_modes(sample_rows)
    write_csv(sample_rows, out_dir / "sample_span_correctness_summary.csv")
    write_csv(token_label_rows, out_dir / "token_span_correctness_rows.csv")
    write_csv(label_summary, out_dir / "span_label_entropy_summary.csv")
    write_csv(mode_label_summary, out_dir / "mode_span_label_entropy_summary.csv")
    write_csv(sample_mode_summary, out_dir / "sample_mode_entropy_summary.csv")
    write_csv(paired_sample_rows(sample_rows), out_dir / "paired_sample_mode_correctness.csv")
    write_report(out_dir / "image_read_span_correctness_report.md", sample_rows, token_label_rows, label_summary, mode_label_summary)
    print(f"[INFO] wrote {len(sample_rows)} sample rows and {len(token_label_rows)} token rows to {out_dir}")


if __name__ == "__main__":
    main()
