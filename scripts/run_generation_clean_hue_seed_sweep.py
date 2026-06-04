#!/usr/bin/env python3
# Copyright 2026
# SPDX-License-Identifier: Apache-2.0

"""Sweep clean generation seeds and validate decoded hue."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import time

import torch
from transformers.generation import StoppingCriteria, StoppingCriteriaList
from tqdm import tqdm

try:
    _TORCHVISION_SCHEMA_LIB = torch.library.Library("torchvision", "DEF")
    _TORCHVISION_SCHEMA_LIB.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_generation_patched_decode_hue import (  # noqa: E402
    build_prompt_ids,
    classify_decoded,
    decode_image,
    generate_ids,
    load_cfg,
    normalize_prompt_rows,
)
from src.utils.model_utils import build_emu3p5  # noqa: E402


SEED_SWEEP_FIELDS = [
    "sample_id",
    "pair_id",
    "expected_color",
    "shape",
    "seed",
    "seed_idx",
    "generation_mode",
    "classifier_free_guidance",
    "elapsed_seconds",
    "generated_tokens",
    "image_path",
    "decoded",
    "predicted_color",
    "hue_match",
    "mean_hue_degrees",
    "expected_hue_degrees",
    "hue_distance_degrees",
    "colorful_pixel_fraction",
    "stop_reason",
]


class DeadlineStoppingCriteria(StoppingCriteria):
    def __init__(self, deadline_monotonic: float):
        self.deadline_monotonic = float(deadline_monotonic)
        self.triggered = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        self.triggered = time.monotonic() >= self.deadline_monotonic
        return self.triggered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/ume_main_t2i_counterfactual_pairs_hueonly_seed69.py")
    parser.add_argument("--max-pairs", type=int, default=1)
    parser.add_argument("--seed-count", type=int, default=2)
    parser.add_argument("--seed-start-idx", type=int, default=0)
    parser.add_argument("--seed-offset", type=int, default=120000)
    parser.add_argument(
        "--sample-id",
        action="append",
        default=[],
        help="Only sweep matching sample id(s). Can be passed multiple times.",
    )
    parser.add_argument(
        "--pair-id",
        action="append",
        default=[],
        help="Only sweep matching pair id(s). Can be passed multiple times.",
    )
    parser.add_argument(
        "--expected-color",
        action="append",
        default=[],
        help="Only sweep samples with matching expected color(s). Can be passed multiple times.",
    )
    parser.add_argument(
        "--list-samples",
        action="store_true",
        help="List samples selected by filters and exit before loading the model.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing clean_hue_seed_sweep.csv rows and skip matching sample/seed entries.",
    )
    parser.add_argument(
        "--stop-after-matches",
        type=int,
        default=0,
        help="Stop after this many hue-matching rows have been found. 0 disables early stop.",
    )
    parser.add_argument(
        "--max-total-seconds",
        type=float,
        default=0.0,
        help="Stop before starting a new seed once total sweep time exceeds this many seconds. 0 disables.",
    )
    parser.add_argument(
        "--per-seed-timeout-seconds",
        type=float,
        default=0.0,
        help="Stop model.generate for a seed after this many seconds and record the partial attempt. 0 disables.",
    )
    parser.add_argument("--generation-mode", choices=["no-cfg", "cfg"], default="cfg")
    parser.add_argument("--target-height", type=int, default=16)
    parser.add_argument("--target-width", type=int, default=16)
    parser.add_argument("--image-area", type=int, default=65536)
    parser.add_argument("--generation-max-new-tokens", type=int, default=700)
    parser.add_argument("--classifier-free-guidance", type=float, default=None)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def filter_samples(samples: list[dict], args: argparse.Namespace) -> list[dict]:
    sample_ids = set(args.sample_id)
    pair_ids = set(args.pair_id)
    expected_colors = {color.lower() for color in args.expected_color}
    filtered = []
    for sample in samples:
        if sample_ids and sample["sample_id"] not in sample_ids:
            continue
        if pair_ids and sample["pair_id"] not in pair_ids:
            continue
        if expected_colors and sample["color"].lower() not in expected_colors:
            continue
        filtered.append(sample)
    return filtered


def row_key(row: dict) -> tuple[str, str, str]:
    return (str(row["sample_id"]), str(row["seed_idx"]), str(row["seed"]))


def seed_for_sample(cfg, args: argparse.Namespace, sample: dict, seed_idx: int) -> int:
    return int(cfg.seed) + args.seed_offset + seed_idx + sum(ord(ch) for ch in sample["sample_id"])


def planned_keys(cfg, args: argparse.Namespace, samples: list[dict]) -> set[tuple[str, str, str]]:
    keys = set()
    for sample in samples:
        for seed_idx in range(args.seed_start_idx, args.seed_start_idx + args.seed_count):
            seed = seed_for_sample(cfg, args, sample, seed_idx)
            keys.add((sample["sample_id"], str(seed_idx), str(seed)))
    return keys


def summarize_rows(rows: list[dict], args: argparse.Namespace, cfg) -> list[dict]:
    return [
        {
            "n": len(rows),
            "decoded": sum(int(row["decoded"]) for row in rows),
            "hue_matches": sum(int(row["hue_match"]) for row in rows),
            "generation_mode": args.generation_mode,
            "classifier_free_guidance": getattr(cfg, "classifier_free_guidance", ""),
            "total_elapsed_seconds": sum(float(row.get("elapsed_seconds") or 0.0) for row in rows),
        }
    ]


def write_outputs(out_dir: Path, args: argparse.Namespace, cfg, rows: list[dict]) -> None:
    write_csv(out_dir / "clean_hue_seed_sweep.csv", rows)
    summary = summarize_rows(rows, args, cfg)
    write_csv(out_dir / "summary.csv", summary)
    matches = [row for row in rows if int(row["hue_match"])]
    lines = [
        "# Clean Generation Hue Seed Sweep",
        "",
        f"- Config: `{args.cfg}`",
        f"- Generation mode: `{args.generation_mode}`",
        f"- Classifier-free guidance: {getattr(cfg, 'classifier_free_guidance', '')}",
        f"- Target grid: {args.target_height}x{args.target_width}",
        f"- Seed start index: {args.seed_start_idx}",
        f"- Seeds per sample: {args.seed_count}",
        f"- Sample filters: sample_id={args.sample_id}, pair_id={args.pair_id}, expected_color={args.expected_color}",
        f"- Resume: {args.resume}",
        f"- Max total seconds: {args.max_total_seconds}",
        f"- Per-seed timeout seconds: {args.per_seed_timeout_seconds}",
        "",
        "| n | decoded | hue matches | total elapsed seconds |",
        "|---:|---:|---:|---:|",
        f"| {summary[0]['n']} | {summary[0]['decoded']} | {summary[0]['hue_matches']} | {summary[0]['total_elapsed_seconds']:.3f} |",
        "",
        "## Matching Seeds",
        "",
        "| sample | expected | seed | predicted | hue distance | image |",
        "|---|---|---:|---|---:|---|",
    ]
    for row in matches:
        lines.append(
            "| {sample_id} | {expected_color} | {seed} | {predicted_color} | {hue_distance_degrees} | {image_path} |".format(
                **row
            )
        )
    (out_dir / "clean_hue_seed_sweep_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.cfg)
    cfg.target_height = args.target_height
    cfg.target_width = args.target_width
    cfg.image_area = args.image_area
    cfg.max_new_tokens = args.generation_max_new_tokens
    cfg.sampling_params["max_new_tokens"] = args.generation_max_new_tokens
    if args.classifier_free_guidance is not None:
        cfg.classifier_free_guidance = args.classifier_free_guidance

    samples = []
    for _pair_id, a_row, b_row in normalize_prompt_rows(cfg, args.max_pairs):
        samples.extend([a_row, b_row])
    samples = filter_samples(samples, args)
    if not samples:
        raise SystemExit("No samples matched the requested filters.")
    if args.list_samples:
        writer = csv.DictWriter(sys.stdout, fieldnames=["sample_id", "pair_id", "color", "shape", "variant"])
        writer.writeheader()
        for sample in samples:
            writer.writerow({key: sample.get(key, "") for key in writer.fieldnames})
        return

    out_dir = Path(args.out_dir)
    image_dir = out_dir / "decoded"
    out_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(out_dir / "clean_hue_seed_sweep.csv") if args.resume else []
    completed = {row_key(row) for row in rows}
    pending = planned_keys(cfg, args, samples) - completed
    if args.resume and not pending:
        write_outputs(out_dir, args, cfg, rows)
        print(f"[INFO] all requested sample/seed rows already exist in {out_dir}; skipping model load")
        return

    model, tokenizer, vq_model = build_emu3p5(
        cfg.model_path,
        cfg.tokenizer_path,
        cfg.vq_path,
        vq_type=cfg.vq_type,
        model_device=cfg.hf_device,
        vq_device=cfg.vq_device,
        **getattr(cfg, "diffusion_decoder_kwargs", {}),
    )
    model.eval()
    cfg.special_token_ids = {key: tokenizer.encode(value)[0] for key, value in cfg.special_tokens.items()}
    sweep_started_at = time.monotonic()
    for sample in tqdm(samples):
        prompt_ids, unconditional_ids, full_unc_ids = build_prompt_ids(cfg, tokenizer, sample["prompt"], model.device)
        for seed_idx in range(args.seed_start_idx, args.seed_start_idx + args.seed_count):
            if args.max_total_seconds and time.monotonic() - sweep_started_at >= args.max_total_seconds:
                write_outputs(out_dir, args, cfg, rows)
                print(f"[INFO] stopping before next seed after {time.monotonic() - sweep_started_at:.3f}s")
                return
            seed = seed_for_sample(cfg, args, sample, seed_idx)
            key = (sample["sample_id"], str(seed_idx), str(seed))
            if key in completed:
                continue
            seed_started_at = time.monotonic()
            deadline = None
            previous_extra_criteria = getattr(cfg, "extra_stopping_criteria", None)
            if args.per_seed_timeout_seconds:
                deadline = DeadlineStoppingCriteria(seed_started_at + args.per_seed_timeout_seconds)
                cfg.extra_stopping_criteria = StoppingCriteriaList([deadline])
            try:
                token_ids = generate_ids(
                    cfg,
                    model,
                    tokenizer,
                    prompt_ids,
                    unconditional_ids,
                    full_unc_ids,
                    seed,
                    args.generation_mode,
                )
            finally:
                if previous_extra_criteria is None:
                    if hasattr(cfg, "extra_stopping_criteria"):
                        delattr(cfg, "extra_stopping_criteria")
                else:
                    cfg.extra_stopping_criteria = previous_extra_criteria
            elapsed_seconds = time.monotonic() - seed_started_at
            image = decode_image(tokenizer, vq_model, token_ids)
            image_path = ""
            if image is not None:
                image_path = str(image_dir / f"{sample['sample_id']}__seed{seed}.png")
                image.save(image_path)
            hue = classify_decoded(image, sample["color"])
            row = {
                "sample_id": sample["sample_id"],
                "pair_id": sample["pair_id"],
                "expected_color": sample["color"],
                "shape": sample["shape"],
                "seed": seed,
                "seed_idx": seed_idx,
                "generation_mode": args.generation_mode,
                "classifier_free_guidance": getattr(cfg, "classifier_free_guidance", ""),
                "elapsed_seconds": f"{elapsed_seconds:.6f}",
                "generated_tokens": len(token_ids),
                "image_path": image_path,
                **hue,
                "stop_reason": "per_seed_timeout" if deadline is not None and deadline.triggered else "",
            }
            rows.append({field: row.get(field, "") for field in SEED_SWEEP_FIELDS})
            completed.add(key)
            write_outputs(out_dir, args, cfg, rows)
            torch.cuda.empty_cache()
            matches = sum(int(existing["hue_match"]) for existing in rows)
            print(
                f"[INFO] wrote seed_idx={seed_idx} seed={seed} "
                f"elapsed={elapsed_seconds:.3f}s match={int(hue['hue_match'])}"
            )
            if args.stop_after_matches and matches >= args.stop_after_matches:
                print(f"[INFO] stopping after {matches} hue matches")
                return

    write_outputs(out_dir, args, cfg, rows)
    matches = sum(int(row["hue_match"]) for row in rows)
    print(f"[INFO] clean hue seed sweep saved to {out_dir}; matches {matches}/{len(rows)}")


if __name__ == "__main__":
    main()
