#!/usr/bin/env python3
"""Link generated-object visual entropy to image-read answer entropy."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


COLORS = {"red", "blue", "green", "yellow", "orange", "purple", "black", "white", "brown", "silver"}
SPATIAL = {"left", "right", "upper", "lower", "top", "bottom", "center", "centre", "middle", "corner", "table"}
HEDGES = {"appears", "seems", "likely", "might", "possibly", "perhaps", "looks", "resembles", "unclear"}
ACTION = {"finish", "task", "step", "move", "press", "release", "grasp", "lift", "place", "align", "done"}
GENERIC_OBJECTS = {"object", "item", "thing", "shape", "image", "photo", "picture"}
TARGET_ALIASES = {
    "alarm clock": {"alarm", "clock", "alarm clock"},
    "potted cactus": {"cactus", "plant", "potted cactus"},
    "red toy car": {"car", "toy car", "red toy car"},
    "soccer ball": {"ball", "soccer", "soccer ball"},
    "spoon": {"spoon", "teaspoon"},
    "teddy bear": {"bear", "teddy", "teddy bear"},
    "yellow rubber duck": {"duck", "rubber duck", "yellow rubber duck"},
    "bowl": {"bowl", "cup"},
}
PROMPT_SUFFIXES = {
    "__describe": "describe",
    "__imageonly": "imageonly",
    "__describe_short": "describe_short",
    "__caption": "caption",
    "__whatin": "whatin",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--qa-spec-json", required=True)
    parser.add_argument("--generation-trace-root", required=True)
    parser.add_argument("--uas-summary-csv")
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def write_csv(rows, path: Path) -> None:
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
    text = re.sub(r"<\|[^>]+?\|>", " ", text.lower())
    return re.findall(r"[a-z0-9]+", text)


def phrases(text: str) -> str:
    text = re.sub(r"<\|[^>]+?\|>", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def percentile(values, q: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    idx = (len(values) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - idx) + values[hi] * (idx - lo)


def pearson(xs, ys) -> float:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if x not in ("", None) and y not in ("", None)]
    if len(pairs) < 2:
        return 0.0
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mx = mean(xs)
    my = mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den else 0.0


def mode_from_sample(sample_id: str) -> str:
    if "__objectname" in sample_id:
        return "objectname"
    for suffix, mode in PROMPT_SUFFIXES.items():
        if sample_id.endswith(suffix):
            return mode
    return "unknown"


def base_id(sample_id: str) -> str:
    base = re.sub(r"__objectname(?:__[^_][A-Za-z0-9_-]*)?$", "", sample_id)
    for suffix in PROMPT_SUFFIXES:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    return re.sub(r"_rep\d+$", "", base)


def repeat_index(sample_id: str) -> str:
    match = re.search(r"_rep(\d+)(?:__|$)", sample_id)
    return match.group(1) if match else ""


def prompt_variant(sample_id: str) -> str:
    match = re.search(r"__objectname__(.+)$", sample_id)
    if match:
        return match.group(1)
    mode = mode_from_sample(sample_id)
    return mode if mode != "unknown" else "default"


def target_aliases(target: str) -> set[str]:
    target = target.lower()
    aliases = set(TARGET_ALIASES.get(target, set()))
    aliases.add(target)
    aliases.update(w for w in target.split() if w not in COLORS and w not in {"rubber", "toy", "potted"})
    return {a for a in aliases if a}


def prompt_object_phrase(sample_id: str) -> str:
    rest = re.sub(r"^t2i_object32[a-z]?_", "", sample_id)
    return rest.replace("_", " ")


def phrase_hit(text: str, aliases: set[str]) -> int:
    clean = phrases(text)
    wordset = set(words(text))
    for alias in aliases:
        if " " in alias and alias in clean:
            return 1
        if " " not in alias and alias in wordset:
            return 1
    return 0


def exact_target_hit(text: str, target: str) -> int:
    clean = phrases(text)
    if " " in target:
        return int(target.lower() in clean)
    return int(target.lower() in set(words(text)))


def token_category(token_text: str, target: str, prompt_object: str) -> str:
    clean = phrases(token_text)
    ws = set(words(token_text))
    if not ws:
        if re.search(r"[.!?,;:]", token_text):
            return "boundary"
        return "space_or_special"
    if phrase_hit(clean, target_aliases(target)):
        return "target_object"
    prompt_terms = {w for w in prompt_object.split() if w not in COLORS}
    if ws & prompt_terms:
        return "prompt_object"
    if ws & COLORS:
        return "color"
    if ws & SPATIAL:
        return "spatial"
    if ws & HEDGES:
        return "hedge"
    if ws & ACTION:
        return "action"
    if ws & GENERIC_OBJECTS:
        return "generic_object"
    return "other"


def generation_stats(generation_root: Path, sample_id: str) -> dict[str, float | str]:
    matches = sorted(generation_root.glob(f"*/entropy_traces/{sample_id}_entropy.jsonl"))
    if not matches:
        return {
            "generation_trace": "",
            "gen_visual_tokens": 0,
            "gen_visual_mean_ume": 0.0,
            "gen_visual_p90_ume": 0.0,
            "gen_visual_max_ume": 0.0,
            "gen_visual_mean_topk_prob_mass": 0.0,
        }
    path = matches[0]
    visual = [r for r in read_jsonl(path) if r.get("token_type") == "visual"]
    umes = [float(r.get("ume", 0.0)) for r in visual]
    masses = [float(r["visual_topk_prob_mass"]) for r in visual if r.get("visual_topk_prob_mass") not in (None, "")]
    return {
        "generation_trace": str(path),
        "gen_visual_tokens": len(visual),
        "gen_visual_mean_ume": mean(umes),
        "gen_visual_p90_ume": percentile(umes, 0.9),
        "gen_visual_max_ume": max(umes) if umes else 0.0,
        "gen_visual_mean_topk_prob_mass": mean(masses),
    }


def load_uas(path: str | None) -> dict[str, dict[str, str]]:
    if not path:
        return {}
    out = {}
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["sample_id"]] = row
    return out


def summarize_read_sample(run_dir: Path, trace_path: Path, qa_spec: dict, gen_root: Path, uas: dict) -> tuple[dict, list[dict]]:
    sample_id = trace_path.name.replace("_entropy.jsonl", "")
    bid = base_id(sample_id)
    mode = mode_from_sample(sample_id)
    target = qa_spec[bid]["answer"]
    prompt_object = prompt_object_phrase(bid)
    raw_path = run_dir / "raw_generations" / f"{sample_id}.txt"
    raw_text = raw_path.read_text(encoding="utf-8") if raw_path.exists() else ""
    text_like = [r for r in read_jsonl(trace_path) if r.get("token_type") in {"text", "thinking"}]
    clean = phrases(raw_text)
    wordset = set(words(raw_text))
    prompt_terms = {w for w in prompt_object.split() if w not in COLORS}
    target_loose_hit = phrase_hit(raw_text, target_aliases(target))
    prompt_object_hit = int(bool(wordset & prompt_terms))

    token_rows = []
    for r in text_like:
        cat = token_category(str(r.get("token_text", "")), target, prompt_object)
        if mode == "objectname":
            token_words = words(str(r.get("token_text", "")))
            if token_words and target_loose_hit:
                cat = "target_object"
            elif token_words and prompt_object_hit:
                cat = "prompt_object"
        token_rows.append(
            {
                "sample_id": sample_id,
                "base_id": bid,
                "mode": mode,
                "step": r["step"],
                "category": cat,
                "token_text": r.get("token_text", ""),
                "u_tok": r.get("u_tok", 0.0),
                "ume": r.get("ume", 0.0),
            }
        )

    gen = generation_stats(gen_root, bid)
    uas_row = uas.get(bid, {})
    summary = {
        "sample_id": sample_id,
        "base_id": bid,
        "mode": mode,
        "prompt_variant": prompt_variant(sample_id),
        "repeat": repeat_index(sample_id),
        "target_object": target,
        "prompt_object": prompt_object,
        "text_like_tokens": len(text_like),
        "mean_answer_ume": mean(float(r["ume"]) for r in text_like),
        "max_answer_ume": max((float(r["ume"]) for r in text_like), default=0.0),
        "first_token_ume": float(text_like[0]["ume"]) if text_like else 0.0,
        "target_exact_phrase_hit": exact_target_hit(raw_text, target),
        "target_loose_hit": target_loose_hit,
        "prompt_object_hit": prompt_object_hit,
        "action_script": int(bool(wordset & ACTION)),
        "hedge_hit": int(bool(wordset & HEDGES)),
        "no_answer": int(not wordset),
        "target_object_token_count": sum(1 for r in token_rows if r["category"] == "target_object"),
        "target_object_mean_ume": mean(float(r["ume"]) for r in token_rows if r["category"] == "target_object"),
        "prompt_object_token_count": sum(1 for r in token_rows if r["category"] == "prompt_object"),
        "prompt_object_mean_ume": mean(float(r["ume"]) for r in token_rows if r["category"] == "prompt_object"),
        "color_token_count": sum(1 for r in token_rows if r["category"] == "color"),
        "color_mean_ume": mean(float(r["ume"]) for r in token_rows if r["category"] == "color"),
        "spatial_token_count": sum(1 for r in token_rows if r["category"] == "spatial"),
        "spatial_mean_ume": mean(float(r["ume"]) for r in token_rows if r["category"] == "spatial"),
        "raw_text": raw_text.replace("\n", "\\n"),
        **gen,
        "uas_8x8_ume_delta_nll_pearson": uas_row.get("ume_delta_nll_pearson", ""),
        "uas_8x8_ume_above_shuffle_p95": int(
            bool(uas_row)
            and float(uas_row.get("ume_delta_nll_pearson", 0.0))
            > float(uas_row.get("ume_delta_nll_shuffle_p95_pearson", 0.0))
        ),
    }
    return summary, token_rows


def aggregate_modes(rows: list[dict]) -> list[dict]:
    out = []
    for mode in sorted({r["mode"] for r in rows}):
        group = [r for r in rows if r["mode"] == mode]
        out.append(
            {
                "mode": mode,
                "sample_count": len(group),
                "mean_answer_ume": mean(float(r["mean_answer_ume"]) for r in group),
                "mean_first_token_ume": mean(float(r["first_token_ume"]) for r in group),
                "target_exact_hits": sum(int(r["target_exact_phrase_hit"]) for r in group),
                "target_loose_hits": sum(int(r["target_loose_hit"]) for r in group),
                "prompt_object_hits": sum(int(r["prompt_object_hit"]) for r in group),
                "action_scripts": sum(int(r["action_script"]) for r in group),
                "hedge_hits": sum(int(r["hedge_hit"]) for r in group),
                "no_answers": sum(int(r["no_answer"]) for r in group),
                "mean_target_object_ume": mean(
                    float(r["target_object_mean_ume"]) for r in group if int(r["target_object_token_count"]) > 0
                ),
                "mean_prompt_object_ume": mean(
                    float(r["prompt_object_mean_ume"]) for r in group if int(r["prompt_object_token_count"]) > 0
                ),
                "mean_color_ume": mean(float(r["color_mean_ume"]) for r in group if int(r["color_token_count"]) > 0),
                "mean_spatial_ume": mean(float(r["spatial_mean_ume"]) for r in group if int(r["spatial_token_count"]) > 0),
            }
        )
    return out


def aggregate_variants(rows: list[dict]) -> list[dict]:
    out = []
    keys = sorted({(r["mode"], r.get("prompt_variant", "default")) for r in rows})
    for mode, variant in keys:
        group = [r for r in rows if r["mode"] == mode and r.get("prompt_variant", "default") == variant]
        out.append(
            {
                "mode": mode,
                "prompt_variant": variant,
                "sample_count": len(group),
                "mean_answer_ume": mean(float(r["mean_answer_ume"]) for r in group),
                "mean_first_token_ume": mean(float(r["first_token_ume"]) for r in group),
                "target_exact_hits": sum(int(r["target_exact_phrase_hit"]) for r in group),
                "target_loose_hits": sum(int(r["target_loose_hit"]) for r in group),
                "prompt_object_hits": sum(int(r["prompt_object_hit"]) for r in group),
                "action_scripts": sum(int(r["action_script"]) for r in group),
                "no_answers": sum(int(r["no_answer"]) for r in group),
                "mean_target_object_ume": mean(
                    float(r["target_object_mean_ume"]) for r in group if int(r["target_object_token_count"]) > 0
                ),
            }
        )
    return out


def paired_rows(rows: list[dict]) -> list[dict]:
    by_base = {}
    for row in rows:
        by_base.setdefault(row["base_id"], {})[row["mode"]] = row
    out = []
    for bid, group in sorted(by_base.items()):
        d = group.get("describe")
        i = group.get("imageonly")
        if not d or not i:
            continue
        out.append(
            {
                "base_id": bid,
                "target_object": d["target_object"],
                "prompt_object": d["prompt_object"],
                "describe_mean_answer_ume": d["mean_answer_ume"],
                "imageonly_mean_answer_ume": i["mean_answer_ume"],
                "imageonly_minus_describe_ume": float(i["mean_answer_ume"]) - float(d["mean_answer_ume"]),
                "describe_target_loose_hit": d["target_loose_hit"],
                "imageonly_target_loose_hit": i["target_loose_hit"],
                "describe_prompt_object_hit": d["prompt_object_hit"],
                "imageonly_prompt_object_hit": i["prompt_object_hit"],
                "describe_action_script": d["action_script"],
                "imageonly_action_script": i["action_script"],
                "gen_visual_mean_ume": d["gen_visual_mean_ume"],
                "gen_visual_p90_ume": d["gen_visual_p90_ume"],
                "uas_8x8_ume_delta_nll_pearson": d["uas_8x8_ume_delta_nll_pearson"],
                "uas_8x8_ume_above_shuffle_p95": d["uas_8x8_ume_above_shuffle_p95"],
            }
        )
    return out


def link_rows(rows: list[dict]) -> list[dict]:
    out = []
    for mode in sorted({r["mode"] for r in rows}):
        group = [r for r in rows if r["mode"] == mode]
        out.append(
            {
                "mode": mode,
                "sample_count": len(group),
                "corr_gen_mean_ume_vs_answer_mean_ume": pearson(
                    [r["gen_visual_mean_ume"] for r in group], [r["mean_answer_ume"] for r in group]
                ),
                "corr_gen_p90_ume_vs_answer_mean_ume": pearson(
                    [r["gen_visual_p90_ume"] for r in group], [r["mean_answer_ume"] for r in group]
                ),
                "corr_uas_pearson_vs_answer_mean_ume": pearson(
                    [r["uas_8x8_ume_delta_nll_pearson"] for r in group], [r["mean_answer_ume"] for r in group]
                ),
                "corr_uas_pearson_vs_target_object_ume": pearson(
                    [r["uas_8x8_ume_delta_nll_pearson"] for r in group], [r["target_object_mean_ume"] for r in group]
                ),
            }
        )
    return out


def write_report(path: Path, mode_summary, link_summary, paired, rows) -> None:
    lines = [
        "# Object32 Generated-image Read Entropy Report",
        "",
        "This report reads images that were generated by the same main model object32 suite and links answer entropy to generation-side visual entropy and UAS summaries.",
        "",
        "## Mode Summary",
        "",
    ]
    for row in mode_summary:
        lines.append(
            f"- {row['mode']}: n={row['sample_count']}, mean UME={float(row['mean_answer_ume']):.4f}, "
            f"first-token UME={float(row['mean_first_token_ume']):.4f}, "
            f"target loose hits={row['target_loose_hits']}/{row['sample_count']}, "
            f"prompt-object hits={row['prompt_object_hits']}/{row['sample_count']}, "
            f"action scripts={row['action_scripts']}/{row['sample_count']}"
        )
    variants = aggregate_variants(rows)
    if len(variants) > len(mode_summary):
        lines.extend(["", "## Variant Summary", ""])
        for row in variants:
            lines.append(
                f"- {row['prompt_variant']}: n={row['sample_count']}, mean UME={float(row['mean_answer_ume']):.4f}, "
                f"target loose hits={row['target_loose_hits']}/{row['sample_count']}, "
                f"action scripts={row['action_scripts']}/{row['sample_count']}, "
                f"target-object UME={float(row['mean_target_object_ume']):.4f}"
            )
    lines.extend(["", "## Generation-to-Answer Correlations", ""])
    for row in link_summary:
        lines.append(
            f"- {row['mode']}: corr(gen mean visual UME, answer UME)={float(row['corr_gen_mean_ume_vs_answer_mean_ume']):+.4f}; "
            f"corr(gen p90 visual UME, answer UME)={float(row['corr_gen_p90_ume_vs_answer_mean_ume']):+.4f}; "
            f"corr(UAS pearson, answer UME)={float(row['corr_uas_pearson_vs_answer_mean_ume']):+.4f}"
        )
    lines.extend(["", "## Paired Shifts", ""])
    for row in sorted(paired, key=lambda r: float(r["imageonly_minus_describe_ume"]), reverse=True)[:20]:
        lines.append(
            f"- `{row['base_id']}` target={row['target_object']}: "
            f"describe UME={float(row['describe_mean_answer_ume']):.4f}, "
            f"imageonly UME={float(row['imageonly_mean_answer_ume']):.4f}, "
            f"delta={float(row['imageonly_minus_describe_ume']):+.4f}"
        )
    wrong_low = sorted(
        [r for r in rows if not int(r["target_loose_hit"]) and float(r["mean_answer_ume"]) <= 0.2],
        key=lambda r: float(r["mean_answer_ume"]),
    )[:12]
    lines.extend(["", "## Low-Entropy Target Misses", ""])
    for row in wrong_low:
        lines.append(
            f"- `{row['sample_id']}` target={row['target_object']} mean UME={float(row['mean_answer_ume']):.4f}: "
            f"`{str(row['raw_text'])[:180].replace('`', '')}`"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    with Path(args.qa_spec_json).open(encoding="utf-8") as f:
        qa_spec = json.load(f)
    qa_spec.pop("__replace_defaults__", None)
    uas = load_uas(args.uas_summary_csv)

    rows = []
    token_rows = []
    for trace_path in sorted((run_dir / "entropy_traces").glob("*_entropy.jsonl")):
        row, tokens = summarize_read_sample(run_dir, trace_path, qa_spec, Path(args.generation_trace_root), uas)
        rows.append(row)
        token_rows.extend(tokens)

    modes = aggregate_modes(rows)
    variants = aggregate_variants(rows)
    paired = paired_rows(rows)
    links = link_rows(rows)
    write_csv(rows, out_dir / "sample_object32_read_entropy_summary.csv")
    write_csv(token_rows, out_dir / "token_object32_read_entropy_rows.csv")
    write_csv(modes, out_dir / "mode_object32_read_entropy_summary.csv")
    write_csv(variants, out_dir / "variant_object32_read_entropy_summary.csv")
    write_csv(paired, out_dir / "paired_object32_read_entropy_shift.csv")
    write_csv(links, out_dir / "generation_answer_link_summary.csv")
    write_report(out_dir / "object32_generated_image_read_entropy_report.md", modes, links, paired, rows)
    print(f"[INFO] wrote {len(rows)} sample rows and {len(token_rows)} token rows to {out_dir}")


if __name__ == "__main__":
    main()
