#!/usr/bin/env python3
# Copyright 2026
# SPDX-License-Identifier: Apache-2.0

"""Validate existing decoded generation images as retrospective T2I benchmark candidates."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_generation_target_hue_cache import PALETTE_HUES, classify_hue, hue_distance  # noqa: E402
from scripts.run_generation_residual_patch_recovery import load_cfg, normalize_prompt_rows  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--sat-threshold", type=float, default=0.25)
    parser.add_argument("--value-min", type=float, default=0.15)
    parser.add_argument("--value-max", type=float, default=0.98)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def image_path_for(run_dir: Path, sample_id: str) -> Path | None:
    decoded = run_dir / "decoded"
    matches = sorted(decoded.glob(f"{sample_id}_image_*.png"))
    return matches[0] if matches else None


def count_jsonl(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def score_image(path: Path | None, expected_color: str, args: argparse.Namespace) -> dict:
    expected_hue = PALETTE_HUES.get(expected_color)
    if path is None or not path.exists():
        return {
            "decoded": 0,
            "predicted_color": "",
            "hue_match": 0,
            "mean_hue_degrees": "",
            "expected_hue_degrees": expected_hue if expected_hue is not None else "",
            "hue_distance_degrees": "",
            "colorful_pixel_fraction": "",
            "validation_note": "missing_decoded_image",
        }
    if expected_hue is None:
        return {
            "decoded": 1,
            "predicted_color": "",
            "hue_match": 0,
            "mean_hue_degrees": "",
            "expected_hue_degrees": "",
            "hue_distance_degrees": "",
            "colorful_pixel_fraction": "",
            "validation_note": "expected_color_not_in_palette",
        }
    image = Image.open(path).convert("RGB")
    hue = classify_hue(
        image,
        sat_threshold=args.sat_threshold,
        value_min=args.value_min,
        value_max=args.value_max,
    )
    mean_hue = hue["mean_hue_degrees"]
    distance = hue_distance(float(mean_hue), expected_hue) if isinstance(mean_hue, float) else ""
    predicted = hue["predicted_color"]
    return {
        "decoded": 1,
        "predicted_color": predicted,
        "hue_match": int(predicted == expected_color),
        "mean_hue_degrees": mean_hue,
        "expected_hue_degrees": expected_hue,
        "hue_distance_degrees": distance,
        "colorful_pixel_fraction": hue["colorful_pixel_fraction"],
        "validation_note": "",
    }


def sample_rows(cfg, run_dir: Path, args: argparse.Namespace) -> list[dict]:
    rows = []
    for _pair_id, a_row, b_row in normalize_prompt_rows(cfg, args.max_pairs):
        for row in (a_row, b_row):
            sample_id = row["sample_id"]
            image_path = image_path_for(run_dir, sample_id)
            raw_path = first_existing([run_dir / "raw_generations" / f"{sample_id}.txt"])
            entropy_path = first_existing([run_dir / "entropy_traces" / f"{sample_id}_entropy.jsonl"])
            score = score_image(image_path, row["color"], args)
            rows.append(
                {
                    "sample_id": sample_id,
                    "pair_id": row["pair_id"],
                    "variant": row["variant"],
                    "expected_color": row["color"],
                    "shape": row["shape"],
                    "counterfactual_value": row["counterfactual_value"],
                    "image_path": str(image_path) if image_path else "",
                    "raw_generation_path": str(raw_path) if raw_path else "",
                    "entropy_trace_path": str(entropy_path) if entropy_path else "",
                    "trace_record_count": count_jsonl(entropy_path),
                    **score,
                }
            )
    return rows


def pair_rows(sample_out: list[dict]) -> list[dict]:
    by_pair: dict[str, list[dict]] = {}
    for row in sample_out:
        by_pair.setdefault(row["pair_id"], []).append(row)
    out = []
    for pair_id, rows in sorted(by_pair.items()):
        rows = sorted(rows, key=lambda row: row["variant"])
        usable = len(rows) == 2 and all(int(row["hue_match"]) == 1 for row in rows)
        out.append(
            {
                "pair_id": pair_id,
                "samples": ";".join(row["sample_id"] for row in rows),
                "expected_colors": ";".join(row["expected_color"] for row in rows),
                "predicted_colors": ";".join(row["predicted_color"] for row in rows),
                "decoded_samples": sum(int(row["decoded"]) for row in rows),
                "hue_matches": sum(int(row["hue_match"]) for row in rows),
                "usable_bidirectional_pair": int(usable),
                "notes": ";".join(row["validation_note"] for row in rows if row["validation_note"]),
            }
        )
    return out


def write_report(out_dir: Path, args: argparse.Namespace, sample_out: list[dict], pair_out: list[dict]) -> None:
    usable = [row for row in pair_out if int(row["usable_bidirectional_pair"]) == 1]
    lines = [
        "# Retrospective Generation Image Manifest",
        "",
        "This report validates existing decoded generation images with the same HSV hue classifier used by the seed-sweep benchmark.",
        "It is a retrospective candidate manifest: decoded images and raw generations exist, but it is not a fresh seed-sweep run.",
        "",
        f"- Config: `{args.cfg}`",
        f"- Run dir: `{args.run_dir}`",
        f"- Samples: {len(sample_out)}",
        f"- Usable bidirectional pairs: {len(usable)}/{len(pair_out)}",
        "",
        "## Pair Summary",
        "",
        "| pair | expected | predicted | decoded | hue matches | usable | notes |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in pair_out:
        lines.append(
            "| {pair_id} | {expected_colors} | {predicted_colors} | {decoded_samples} | {hue_matches} | {usable_bidirectional_pair} | {notes} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Sample Summary",
            "",
            "| sample | expected | predicted | match | hue distance | colorful fraction | image |",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    for row in sample_out:
        lines.append(
            "| {sample_id} | {expected_color} | {predicted_color} | {hue_match} | {hue_distance_degrees} | {colorful_pixel_fraction} | {image_path} |".format(
                **row
            )
        )
    (out_dir / "retrospective_generation_image_manifest_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.cfg)
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_out = sample_rows(cfg, run_dir, args)
    pair_out = pair_rows(sample_out)
    write_csv(
        out_dir / "sample_validation.csv",
        sample_out,
        [
            "sample_id",
            "pair_id",
            "variant",
            "expected_color",
            "shape",
            "counterfactual_value",
            "image_path",
            "raw_generation_path",
            "entropy_trace_path",
            "trace_record_count",
            "decoded",
            "predicted_color",
            "hue_match",
            "mean_hue_degrees",
            "expected_hue_degrees",
            "hue_distance_degrees",
            "colorful_pixel_fraction",
            "validation_note",
        ],
    )
    write_csv(
        out_dir / "pair_validation.csv",
        pair_out,
        [
            "pair_id",
            "samples",
            "expected_colors",
            "predicted_colors",
            "decoded_samples",
            "hue_matches",
            "usable_bidirectional_pair",
            "notes",
        ],
    )
    write_report(out_dir, args, sample_out, pair_out)
    print(
        f"[INFO] retrospective manifest saved to {out_dir}; "
        f"usable pairs {sum(int(row['usable_bidirectional_pair']) for row in pair_out)}/{len(pair_out)}"
    )


if __name__ == "__main__":
    main()
