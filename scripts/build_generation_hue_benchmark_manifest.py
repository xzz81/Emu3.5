#!/usr/bin/env python3
# Copyright 2026
# SPDX-License-Identifier: Apache-2.0

"""Build a filtered hue-generation benchmark manifest from seed-sweep CSVs."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_generation_residual_patch_recovery import load_cfg, normalize_prompt_rows  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/ume_main_t2i_counterfactual_pairs_hueonly_seed69.py")
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--csv", action="append", required=True, help="Seed-sweep CSV path. Can be repeated.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--min-generated-tokens", type=int, default=16)
    parser.add_argument("--suggested-seed-count", type=int, default=1)
    parser.add_argument("--suggested-generation-mode", choices=["no-cfg", "cfg"], default="cfg")
    parser.add_argument("--suggested-cfg-scale", type=float, default=2.0)
    parser.add_argument("--suggested-target-height", type=int, default=16)
    parser.add_argument("--suggested-target-width", type=int, default=16)
    parser.add_argument("--suggested-image-area", type=int, default=65536)
    parser.add_argument("--suggested-max-new-tokens", type=int, default=320)
    parser.add_argument("--suggested-max-total-seconds", type=float, default=2400.0)
    parser.add_argument("--suggested-per-seed-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--suggested-hard-timeout-seconds", type=float, default=2700.0)
    parser.add_argument("--suggested-output-root", default="outputs/generation_clean_hue_seed_sweeps_from_manifest")
    parser.add_argument("--no-require-image-exists", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["source_csv"] = str(path)
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_int(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def as_float(value, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def image_exists(path_text: str) -> bool:
    if not path_text:
        return False
    path = Path(path_text)
    if path.exists():
        return True
    return (REPO_ROOT / path_text).exists()


def expected_samples(cfg, max_pairs: int) -> tuple[list[dict], list[dict]]:
    samples = []
    pairs = []
    for pair_id, a_row, b_row in normalize_prompt_rows(cfg, max_pairs):
        samples.extend([a_row, b_row])
        pairs.append(
            {
                "pair_id": pair_id,
                "sample_a": a_row["sample_id"],
                "sample_b": b_row["sample_id"],
                "color_a": a_row["color"],
                "color_b": b_row["color"],
                "shape": a_row["shape"],
            }
        )
    return samples, pairs


def valid_row(row: dict, args: argparse.Namespace) -> bool:
    if as_int(row.get("decoded")) != 1:
        return False
    if as_int(row.get("hue_match")) != 1:
        return False
    if as_int(row.get("generated_tokens")) < args.min_generated_tokens:
        return False
    if not args.no_require_image_exists and not image_exists(str(row.get("image_path", ""))):
        return False
    return True


def valid_sort_key(row: dict):
    hue_distance = as_float(row.get("hue_distance_degrees"))
    if not math.isfinite(hue_distance):
        hue_distance = 1e9
    colorful = as_float(row.get("colorful_pixel_fraction"), 0.0)
    generated = as_int(row.get("generated_tokens"))
    return (hue_distance, -colorful, -generated, as_int(row.get("seed_idx")))


def sample_status_rows(samples: list[dict], rows_by_sample: dict[str, list[dict]], selected: dict[str, dict]) -> list[dict]:
    statuses = []
    for sample in samples:
        attempts = rows_by_sample.get(sample["sample_id"], [])
        decoded = [row for row in attempts if as_int(row.get("decoded")) == 1]
        hue_matches = [row for row in attempts if as_int(row.get("hue_match")) == 1]
        best = selected.get(sample["sample_id"])
        statuses.append(
            {
                "sample_id": sample["sample_id"],
                "pair_id": sample["pair_id"],
                "expected_color": sample["color"],
                "shape": sample["shape"],
                "attempts": len(attempts),
                "decoded": len(decoded),
                "hue_matches": len(hue_matches),
                "selected": int(best is not None),
                "selected_seed": best.get("seed", "") if best else "",
                "selected_seed_idx": best.get("seed_idx", "") if best else "",
                "selected_image_path": best.get("image_path", "") if best else "",
                "best_observed_prediction": best_observed_prediction(attempts),
            }
        )
    return statuses


def best_observed_prediction(rows: list[dict]) -> str:
    decoded = [row for row in rows if as_int(row.get("decoded")) == 1]
    if not decoded:
        return ""
    decoded.sort(key=lambda row: (-as_float(row.get("colorful_pixel_fraction"), 0.0), as_int(row.get("seed_idx"))))
    return str(decoded[0].get("predicted_color", ""))


def pair_status_rows(pairs: list[dict], selected: dict[str, dict]) -> list[dict]:
    statuses = []
    for pair in pairs:
        a_ok = pair["sample_a"] in selected
        b_ok = pair["sample_b"] in selected
        missing = []
        if not a_ok:
            missing.append(pair["sample_a"])
        if not b_ok:
            missing.append(pair["sample_b"])
        statuses.append(
            {
                "pair_id": pair["pair_id"],
                "shape": pair["shape"],
                "color_a": pair["color_a"],
                "color_b": pair["color_b"],
                "sample_a": pair["sample_a"],
                "sample_b": pair["sample_b"],
                "selected_a": int(a_ok),
                "selected_b": int(b_ok),
                "usable_bidirectional_pair": int(a_ok and b_ok),
                "missing_samples": ";".join(missing),
            }
        )
    return statuses


def max_seed_idx(rows: list[dict]) -> int:
    if not rows:
        return -1
    return max(as_int(row.get("seed_idx"), -1) for row in rows)


def latest_generation_mode(rows: list[dict]) -> str:
    if not rows:
        return ""
    rows = sorted(rows, key=lambda row: as_int(row.get("seed_idx"), -1), reverse=True)
    return str(rows[0].get("generation_mode", ""))


def latest_cfg_scale(rows: list[dict]) -> str:
    if not rows:
        return ""
    rows = sorted(rows, key=lambda row: as_int(row.get("seed_idx"), -1), reverse=True)
    return str(rows[0].get("classifier_free_guidance", ""))


def command_slug(sample_id: str) -> str:
    return sample_id.replace("cf_", "").replace("__", "_").replace("/", "_")


def suggested_out_dir(args: argparse.Namespace, sample: dict, next_seed_start_idx: int) -> str:
    return str(
        Path(args.suggested_output_root)
        / f"{command_slug(sample['sample_id'])}_idx{next_seed_start_idx}_{next_seed_start_idx + args.suggested_seed_count - 1}"
    )


def suggested_command(args: argparse.Namespace, sample: dict, next_seed_start_idx: int, out_dir: str) -> str:
    parts = [
        "CUDA_VISIBLE_DEVICES=0",
    ]
    if args.suggested_hard_timeout_seconds:
        parts.extend(["timeout", "--foreground", str(args.suggested_hard_timeout_seconds)])
    parts.extend(
        [
        "python",
        "scripts/run_generation_clean_hue_seed_sweep.py",
        "--cfg",
        args.cfg,
        "--max-pairs",
        str(args.max_pairs),
        "--sample-id",
        sample["sample_id"],
        "--seed-start-idx",
        str(next_seed_start_idx),
        "--seed-count",
        str(args.suggested_seed_count),
        "--generation-mode",
        args.suggested_generation_mode,
        "--target-height",
        str(args.suggested_target_height),
        "--target-width",
        str(args.suggested_target_width),
        "--image-area",
        str(args.suggested_image_area),
        "--generation-max-new-tokens",
        str(args.suggested_max_new_tokens),
        "--classifier-free-guidance",
        str(args.suggested_cfg_scale),
        "--resume",
        "--out-dir",
        out_dir,
        ]
    )
    if args.suggested_max_total_seconds:
        parts.extend(["--max-total-seconds", str(args.suggested_max_total_seconds)])
    if args.suggested_per_seed_timeout_seconds:
        parts.extend(["--per-seed-timeout-seconds", str(args.suggested_per_seed_timeout_seconds)])
    return " ".join(parts)


def missing_sweep_queue_rows(
    samples: list[dict],
    pairs: list[dict],
    rows_by_sample: dict[str, list[dict]],
    selected: dict[str, dict],
    args: argparse.Namespace,
) -> list[dict]:
    pair_lookup = {pair["sample_a"]: pair for pair in pairs}
    pair_lookup.update({pair["sample_b"]: pair for pair in pairs})
    queue = []
    for sample in samples:
        sample_id = sample["sample_id"]
        if sample_id in selected:
            continue
        attempts = rows_by_sample.get(sample_id, [])
        pair = pair_lookup.get(sample_id, {})
        paired_sample = ""
        paired_selected_seed = ""
        if pair:
            paired_sample = pair["sample_b"] if sample_id == pair["sample_a"] else pair["sample_a"]
            if paired_sample in selected:
                paired_selected_seed = str(selected[paired_sample].get("seed", ""))
        next_seed_start_idx = max_seed_idx(attempts) + 1
        out_dir = suggested_out_dir(args, sample, next_seed_start_idx)
        queue.append(
            {
                "priority": 0 if paired_selected_seed else 1,
                "sample_id": sample_id,
                "pair_id": sample["pair_id"],
                "expected_color": sample["color"],
                "shape": sample["shape"],
                "attempts": len(attempts),
                "decoded": sum(1 for row in attempts if as_int(row.get("decoded")) == 1),
                "hue_matches": sum(1 for row in attempts if as_int(row.get("hue_match")) == 1),
                "next_seed_start_idx": next_seed_start_idx,
                "suggested_seed_count": args.suggested_seed_count,
                "suggested_out_dir": out_dir,
                "suggested_command": suggested_command(args, sample, next_seed_start_idx, out_dir),
                "paired_sample_id": paired_sample,
                "paired_selected_seed": paired_selected_seed,
                "latest_generation_mode": latest_generation_mode(attempts),
                "latest_classifier_free_guidance": latest_cfg_scale(attempts),
            }
        )
    queue.sort(key=lambda row: (as_int(row["priority"]), -as_int(row["attempts"]), row["pair_id"], row["sample_id"]))
    return queue


def selected_rows(selected: dict[str, dict]) -> list[dict]:
    fields = [
        "sample_id",
        "pair_id",
        "expected_color",
        "shape",
        "seed",
        "seed_idx",
        "generation_mode",
        "classifier_free_guidance",
        "generated_tokens",
        "predicted_color",
        "hue_distance_degrees",
        "colorful_pixel_fraction",
        "image_path",
        "source_csv",
    ]
    rows = []
    for sample_id in sorted(selected):
        row = selected[sample_id]
        rows.append({field: row.get(field, "") for field in fields})
    return rows


def write_report(
    out_dir: Path,
    args: argparse.Namespace,
    selected_out: list[dict],
    samples_out: list[dict],
    pairs_out: list[dict],
    queue_out: list[dict],
):
    usable_pairs = [row for row in pairs_out if as_int(row["usable_bidirectional_pair"]) == 1]
    lines = [
        "# Filtered Generation Hue Benchmark Manifest",
        "",
        "This manifest selects decoded clean generations that pass hue validation and reports which clean/corrupt pairs are ready for generation-side causal patching.",
        "",
        f"- Config: `{args.cfg}`",
        f"- Input CSVs: {len(args.csv)}",
        f"- Min generated tokens: {args.min_generated_tokens}",
        f"- Suggested seed count: {args.suggested_seed_count}",
        f"- Suggested generation mode: {args.suggested_generation_mode}",
        f"- Suggested CFG scale: {args.suggested_cfg_scale}",
        f"- Require image exists: {not args.no_require_image_exists}",
        "",
        "| expected samples | selected valid samples | usable bidirectional pairs |",
        "|---:|---:|---:|",
        f"| {len(samples_out)} | {len(selected_out)} | {len(usable_pairs)} |",
        "",
        "## Usable Pairs",
        "",
        "| pair | colors | selected | missing |",
        "|---|---|---:|---|",
    ]
    for row in pairs_out:
        lines.append(
            "| {pair_id} | {color_a}/{color_b} | {usable_bidirectional_pair} | {missing_samples} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Selected Seeds",
            "",
            "| sample | expected | seed | predicted | hue distance | image |",
            "|---|---|---:|---|---:|---|",
        ]
    )
    for row in selected_out:
        lines.append(
            "| {sample_id} | {expected_color} | {seed} | {predicted_color} | {hue_distance_degrees} | {image_path} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Missing Sweep Queue",
            "",
            "| priority | sample | expected | attempts | next seed idx | suggested count | paired selected seed | out dir |",
            "|---:|---|---|---:|---:|---:|---|---|",
        ]
    )
    for row in queue_out:
        lines.append(
            "| {priority} | {sample_id} | {expected_color} | {attempts} | {next_seed_start_idx} | {suggested_seed_count} | {paired_selected_seed} | {suggested_out_dir} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
        ]
    )
    if usable_pairs:
        lines.append(
            "At least one bidirectional clean/corrupt pair is ready for generation-side patching under the manifest criteria."
        )
    else:
        lines.append(
            "No bidirectional clean/corrupt pair is ready yet. Continue targeted seed sweeps or change prompts/resolution before generation-side causal decomposition."
        )
    if queue_out:
        first = queue_out[0]
        lines.extend(
            [
                "",
                "Next targeted sweep candidate:",
                "",
                "```text",
                f"sample_id = {first['sample_id']}",
                f"expected_color = {first['expected_color']}",
                f"seed_start_idx = {first['next_seed_start_idx']}",
                f"seed_count = {first['suggested_seed_count']}",
                f"out_dir = {first['suggested_out_dir']}",
                "```",
                "",
                "Command:",
                "",
                "```bash",
                first["suggested_command"],
                "```",
            ]
        )
    (out_dir / "filtered_generation_hue_benchmark_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.cfg)
    samples, pairs = expected_samples(cfg, args.max_pairs)

    all_rows = []
    for path_text in args.csv:
        all_rows.extend(read_csv(Path(path_text)))

    rows_by_sample: dict[str, list[dict]] = {}
    for row in all_rows:
        rows_by_sample.setdefault(str(row.get("sample_id", "")), []).append(row)

    selected = {}
    for sample in samples:
        candidates = [row for row in rows_by_sample.get(sample["sample_id"], []) if valid_row(row, args)]
        if not candidates:
            continue
        candidates.sort(key=valid_sort_key)
        selected[sample["sample_id"]] = candidates[0]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_out = selected_rows(selected)
    samples_out = sample_status_rows(samples, rows_by_sample, selected)
    pairs_out = pair_status_rows(pairs, selected)
    queue_out = missing_sweep_queue_rows(samples, pairs, rows_by_sample, selected, args)

    write_csv(
        out_dir / "selected_valid_seeds.csv",
        selected_out,
        [
            "sample_id",
            "pair_id",
            "expected_color",
            "shape",
            "seed",
            "seed_idx",
            "generation_mode",
            "classifier_free_guidance",
            "generated_tokens",
            "predicted_color",
            "hue_distance_degrees",
            "colorful_pixel_fraction",
            "image_path",
            "source_csv",
        ],
    )
    write_csv(
        out_dir / "sample_status.csv",
        samples_out,
        [
            "sample_id",
            "pair_id",
            "expected_color",
            "shape",
            "attempts",
            "decoded",
            "hue_matches",
            "selected",
            "selected_seed",
            "selected_seed_idx",
            "selected_image_path",
            "best_observed_prediction",
        ],
    )
    write_csv(
        out_dir / "pair_status.csv",
        pairs_out,
        [
            "pair_id",
            "shape",
            "color_a",
            "color_b",
            "sample_a",
            "sample_b",
            "selected_a",
            "selected_b",
            "usable_bidirectional_pair",
            "missing_samples",
        ],
    )
    write_csv(
        out_dir / "missing_sweep_queue.csv",
        queue_out,
        [
            "priority",
            "sample_id",
            "pair_id",
            "expected_color",
            "shape",
            "attempts",
            "decoded",
            "hue_matches",
            "next_seed_start_idx",
            "suggested_seed_count",
            "suggested_out_dir",
            "suggested_command",
            "paired_sample_id",
            "paired_selected_seed",
            "latest_generation_mode",
            "latest_classifier_free_guidance",
        ],
    )
    write_report(out_dir, args, selected_out, samples_out, pairs_out, queue_out)
    usable = sum(as_int(row["usable_bidirectional_pair"]) for row in pairs_out)
    print(f"[INFO] selected {len(selected_out)}/{len(samples_out)} samples; usable pairs {usable}/{len(pairs_out)}")


if __name__ == "__main__":
    main()
