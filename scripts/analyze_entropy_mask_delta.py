#!/usr/bin/env python3
"""Relate generation-side visual UME to readback patch-mask sensitivity."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
import torch

try:
    _TORCHVISION_SCHEMA_LIB = torch.library.Library("torchvision", "DEF")
    _TORCHVISION_SCHEMA_LIB.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
except Exception:
    pass

from transformers import GenerationConfig
from transformers.generation import LogitsProcessor, LogitsProcessorList

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.input_utils import build_image
from src.utils.logits_processor import BOV, EOI
from src.utils.model_utils import build_emu3p5


class TextOnlyLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids, scores):
        scores[:, BOV:] = -math.inf
        return scores


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model-path", default="/workspace/home/AAAI 2027/models/BAAI/Emu3.5")
    parser.add_argument("--vq-path", default="/workspace/home/AAAI 2027/models/BAAI/Emu3.5-VisionTokenizer")
    parser.add_argument("--tokenizer-path", default="./src/tokenizer_emu3_ibq")
    parser.add_argument("--vq-type", default="ibq")
    parser.add_argument("--model-device", default="auto")
    parser.add_argument("--vq-device", default="cuda:0")
    parser.add_argument("--image-area", type=int, default=512 * 512)
    parser.add_argument("--describe-prompt", default="Describe the main objects, colors, text, and spatial layout in this image.")
    parser.add_argument("--describe-max-new-tokens", type=int, default=24)
    parser.add_argument("--entropy-field", default="ume_full")
    parser.add_argument("--mask-grid", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--out-dir-name", default=None)
    parser.add_argument("--shuffle-repeats", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260530)
    return parser.parse_args()


def write_csv(rows, path: Path):
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_entropy_map(trace_path: Path, field: str):
    records = []
    for line in trace_path.open(encoding="utf-8"):
        row = json.loads(line)
        records.append(row)
        if int(row.get("token_id", -1)) == EOI:
            break
    visual = [r for r in records if r.get("token_type") == "visual"]
    side = int(round(len(visual) ** 0.5))
    if side * side != len(visual):
        raise ValueError(f"{trace_path} has {len(visual)} visual tokens; expected a square grid")
    values = [float(r[field]) for r in visual if r.get(field) is not None]
    if len(values) != len(visual):
        raise KeyError(f"{field} is not available for all visual tokens in {trace_path}")
    return np.asarray(values, dtype=np.float32).reshape(side, side)


def sample_id_from_image_path(image_path: Path):
    stem = image_path.stem
    if stem.endswith("_image_00"):
        return stem[: -len("_image_00")]
    return stem


def parse_grid_shape(image_string):
    match = re.search(r"<\|image start\|>(\d+)\*(\d+)<\|image token\|>", image_string)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def build_read_prompt(tokenizer, vq_model, image: Image.Image, image_area: int, describe_prompt: str):
    cfg = SimpleNamespace(image_area=image_area)
    image_string = build_image(image.convert("RGB"), cfg, tokenizer, vq_model)
    grid_shape = parse_grid_shape(image_string)
    prompt = (
        f"{tokenizer.bos_token}You are a helpful assistant. USER: "
        f"{image_string} {describe_prompt} ASSISTANT:"
    )
    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
    return input_ids, grid_shape


def heat_color(x):
    x = float(np.clip(x, 0.0, 1.0))
    if x < 0.5:
        t = x * 2.0
        return int(40 * (1 - t)), int(60 * t), int(255 * (1 - t) + 30 * t)
    t = (x - 0.5) * 2.0
    return int(255 * t + 30 * (1 - t)), int(60 * (1 - t)), int(20 * (1 - t))


def save_heatmap(arr, path: Path, scale=32):
    arr = np.asarray(arr, dtype=np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    norm = (arr - lo) / (hi - lo + 1e-8)
    rgb = np.zeros((*arr.shape, 3), dtype=np.uint8)
    for y in range(arr.shape[0]):
        for x in range(arr.shape[1]):
            rgb[y, x] = heat_color(norm[y, x])
    Image.fromarray(rgb).resize((arr.shape[1] * scale, arr.shape[0] * scale), Image.Resampling.NEAREST).save(path)


def rankdata(x):
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def block_mean(arr, grid: int):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape[0] % grid or arr.shape[1] % grid:
        raise ValueError(f"cannot pool {arr.shape} to {grid}x{grid}")
    by = arr.shape[0] // grid
    bx = arr.shape[1] // grid
    return arr.reshape(grid, by, grid, bx).mean(axis=(1, 3))


def mask_image_cell(image: Image.Image, grid: int, row: int, col: int):
    out = image.convert("RGB").copy()
    w, h = out.size
    x0, x1 = col * w // grid, (col + 1) * w // grid
    y0, y1 = row * h // grid, (row + 1) * h // grid
    patch = np.asarray(out.crop((x0, y0, x1, y1)), dtype=np.float32)
    fill = tuple(int(v) for v in patch.reshape(-1, 3).mean(axis=0))
    draw = ImageDraw.Draw(out)
    draw.rectangle((x0, y0, x1, y1), fill=fill)
    return out


@torch.no_grad()
def generate_answer(model, tokenizer, prompt_ids, max_new_tokens: int):
    prompt_ids = prompt_ids.to(model.device)
    input_len = prompt_ids.shape[1]
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated = model.generate(
        prompt_ids,
        generation_config=generation_config,
        logits_processor=LogitsProcessorList([TextOnlyLogitsProcessor()]),
    )
    return generated[:, input_len:].to(model.device)


@torch.no_grad()
def score_answer(model, prompt_ids, answer_ids):
    prompt_ids = prompt_ids.to(model.device)
    answer_ids = answer_ids.to(model.device)
    full_ids = torch.cat([prompt_ids, answer_ids], dim=1)
    input_len = prompt_ids.shape[1]
    outputs = model(input_ids=full_ids, use_cache=False, return_dict=True)
    logits = outputs.logits[:, input_len - 1 : full_ids.shape[1] - 1, :BOV].float()
    targets = answer_ids[0]
    valid = targets < BOV
    logits = logits[:, valid, :]
    targets = targets[valid]
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    nll = -log_probs[0, torch.arange(targets.numel(), device=targets.device), targets].mean()
    entropy = -(probs * log_probs).sum(dim=-1).mean() / math.log(BOV)
    return float(nll.item()), float(entropy.item()), int(targets.numel())


def compare_maps(ume_map, delta_map):
    x = ume_map.reshape(-1)
    y = delta_map.reshape(-1)
    k = max(1, int(round(0.2 * len(x))))
    top_x = set(np.argsort(x)[-k:].tolist())
    top_y = set(np.argsort(y)[-k:].tolist())
    rest_idx = [i for i in range(len(y)) if i not in top_x]
    top_vals = y[list(top_x)]
    rest_vals = y[rest_idx]
    return {
        "pearson": corr(x, y),
        "spearman": corr(rankdata(x), rankdata(y)),
        "top20_overlap_fraction": len(top_x & top_y) / k,
        "delta_on_ume_top20": float(top_vals.mean()),
        "delta_on_ume_rest80": float(np.asarray(rest_vals).mean()),
        "top20_delta_minus_rest": float(top_vals.mean() - np.asarray(rest_vals).mean()),
        "top20_delta_enrichment": float(top_vals.mean() / (np.asarray(rest_vals).mean() + 1e-12)),
    }


def center_prior(grid: int):
    coords = np.arange(grid, dtype=np.float32)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    c = (grid - 1) / 2.0
    sigma = max(grid / 3.0, 1e-6)
    prior = np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2.0 * sigma**2))
    return prior.astype(np.float32)


def baseline_stats(signal_map, delta_map, rng, repeats: int, prefix: str):
    observed = compare_maps(signal_map, delta_map)
    flat = np.asarray(signal_map, dtype=np.float32).reshape(-1)
    pearsons = []
    spearmans = []
    overlaps = []
    for _ in range(max(0, repeats)):
        shuffled = rng.permutation(flat).reshape(signal_map.shape)
        stats = compare_maps(shuffled, delta_map)
        pearsons.append(stats["pearson"])
        spearmans.append(stats["spearman"])
        overlaps.append(stats["top20_overlap_fraction"])

    out = {
        f"{prefix}_pearson": observed["pearson"],
        f"{prefix}_spearman": observed["spearman"],
        f"{prefix}_top20_overlap": observed["top20_overlap_fraction"],
        f"{prefix}_top20_enrichment": observed["top20_delta_enrichment"],
    }
    if repeats > 0:
        pearsons_arr = np.asarray(pearsons, dtype=np.float64)
        spearmans_arr = np.asarray(spearmans, dtype=np.float64)
        overlaps_arr = np.asarray(overlaps, dtype=np.float64)
        out.update(
            {
                f"{prefix}_shuffle_mean_pearson": float(pearsons_arr.mean()),
                f"{prefix}_shuffle_p95_pearson": float(np.quantile(pearsons_arr, 0.95)),
                f"{prefix}_above_shuffle_mean_pearson": float(observed["pearson"] - pearsons_arr.mean()),
                f"{prefix}_shuffle_ge_pearson_fraction": float((pearsons_arr >= observed["pearson"]).mean()),
                f"{prefix}_shuffle_mean_spearman": float(spearmans_arr.mean()),
                f"{prefix}_shuffle_p95_spearman": float(np.quantile(spearmans_arr, 0.95)),
                f"{prefix}_above_shuffle_mean_spearman": float(observed["spearman"] - spearmans_arr.mean()),
                f"{prefix}_shuffle_ge_spearman_fraction": float((spearmans_arr >= observed["spearman"]).mean()),
                f"{prefix}_shuffle_mean_top20_overlap": float(overlaps_arr.mean()),
                f"{prefix}_shuffle_p95_top20_overlap": float(np.quantile(overlaps_arr, 0.95)),
                f"{prefix}_above_shuffle_mean_top20_overlap": float(
                    observed["top20_overlap_fraction"] - overlaps_arr.mean()
                ),
                f"{prefix}_shuffle_ge_top20_overlap_fraction": float(
                    (overlaps_arr >= observed["top20_overlap_fraction"]).mean()
                ),
            }
        )
    return out


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir_name = args.out_dir_name
    if out_dir_name is None:
        out_dir_name = "mask_delta_analysis" if args.mask_grid == 4 else f"mask_delta_analysis_{args.mask_grid}x{args.mask_grid}"
    out_dir = run_dir / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    model, tokenizer, vq_model = build_emu3p5(
        args.model_path,
        args.tokenizer_path,
        args.vq_path,
        vq_type=args.vq_type,
        model_device=args.model_device,
        vq_device=args.vq_device,
    )

    images = sorted((run_dir / "decoded").glob("*.png"))
    if args.max_samples > 0:
        images = images[: args.max_samples]

    cell_rows = []
    summary_rows = []
    for image_path in images:
        sample_id = sample_id_from_image_path(image_path)
        image = Image.open(image_path).convert("RGB")
        entropy_map = load_entropy_map(run_dir / "entropy_traces" / f"{sample_id}_entropy.jsonl", args.entropy_field)
        coarse_ume = block_mean(entropy_map, args.mask_grid)

        prompt_ids, _ = build_read_prompt(tokenizer, vq_model, image, args.image_area, args.describe_prompt)
        answer_ids = generate_answer(model, tokenizer, prompt_ids, args.describe_max_new_tokens)
        base_nll, base_entropy, answer_tokens = score_answer(model, prompt_ids, answer_ids)

        delta_nll = np.zeros((args.mask_grid, args.mask_grid), dtype=np.float32)
        delta_entropy = np.zeros((args.mask_grid, args.mask_grid), dtype=np.float32)
        for row in range(args.mask_grid):
            for col in range(args.mask_grid):
                masked = mask_image_cell(image, args.mask_grid, row, col)
                masked_prompt_ids, _ = build_read_prompt(tokenizer, vq_model, masked, args.image_area, args.describe_prompt)
                masked_nll, masked_entropy, _ = score_answer(model, masked_prompt_ids, answer_ids)
                delta_nll[row, col] = masked_nll - base_nll
                delta_entropy[row, col] = masked_entropy - base_entropy
                cell_rows.append(
                    {
                        "sample_id": sample_id,
                        "row": row,
                        "col": col,
                        "coarse_ume": float(coarse_ume[row, col]),
                        "delta_nll": float(delta_nll[row, col]),
                        "delta_entropy": float(delta_entropy[row, col]),
                    }
                )

        save_heatmap(coarse_ume, out_dir / f"{sample_id}_coarse_ume_heatmap.png")
        save_heatmap(delta_nll, out_dir / f"{sample_id}_delta_nll_heatmap.png")
        save_heatmap(delta_entropy, out_dir / f"{sample_id}_delta_entropy_heatmap.png")

        nll_stats = compare_maps(coarse_ume, delta_nll)
        ent_stats = compare_maps(coarse_ume, delta_entropy)
        center = center_prior(args.mask_grid)
        center_nll_stats = compare_maps(center, delta_nll)
        center_ent_stats = compare_maps(center, delta_entropy)
        nll_baselines = baseline_stats(
            coarse_ume,
            delta_nll,
            rng,
            args.shuffle_repeats,
            "ume_delta_nll",
        )
        ent_baselines = baseline_stats(
            coarse_ume,
            delta_entropy,
            rng,
            args.shuffle_repeats,
            "ume_delta_entropy",
        )
        row = {
            "sample_id": sample_id,
            "mask_grid": args.mask_grid,
            "shuffle_repeats": args.shuffle_repeats,
            "answer_tokens": answer_tokens,
            "base_nll": base_nll,
            "base_entropy": base_entropy,
            "mean_delta_nll": float(delta_nll.mean()),
            "mean_delta_entropy": float(delta_entropy.mean()),
            "pearson_ume_delta_nll": nll_stats["pearson"],
            "spearman_ume_delta_nll": nll_stats["spearman"],
            "top20_overlap_ume_delta_nll": nll_stats["top20_overlap_fraction"],
            "delta_nll_on_ume_top20": nll_stats["delta_on_ume_top20"],
            "delta_nll_on_ume_rest80": nll_stats["delta_on_ume_rest80"],
            "top20_delta_nll_minus_rest": nll_stats["top20_delta_minus_rest"],
            "top20_delta_nll_enrichment": nll_stats["top20_delta_enrichment"],
            "pearson_ume_delta_entropy": ent_stats["pearson"],
            "spearman_ume_delta_entropy": ent_stats["spearman"],
            "top20_overlap_ume_delta_entropy": ent_stats["top20_overlap_fraction"],
            "delta_entropy_on_ume_top20": ent_stats["delta_on_ume_top20"],
            "delta_entropy_on_ume_rest80": ent_stats["delta_on_ume_rest80"],
            "top20_delta_entropy_minus_rest": ent_stats["top20_delta_minus_rest"],
            "top20_delta_entropy_enrichment": ent_stats["top20_delta_enrichment"],
            "pearson_center_delta_nll": center_nll_stats["pearson"],
            "spearman_center_delta_nll": center_nll_stats["spearman"],
            "top20_overlap_center_delta_nll": center_nll_stats["top20_overlap_fraction"],
            "pearson_center_delta_entropy": center_ent_stats["pearson"],
            "spearman_center_delta_entropy": center_ent_stats["spearman"],
            "top20_overlap_center_delta_entropy": center_ent_stats["top20_overlap_fraction"],
            "description": tokenizer.decode(answer_ids[0], skip_special_tokens=False),
        }
        row.update(nll_baselines)
        row.update(ent_baselines)
        summary_rows.append(row)

    write_csv(cell_rows, out_dir / "mask_cell_metrics.csv")
    write_csv(summary_rows, out_dir / "summary_mask_delta.csv")
    print(f"[INFO] mask delta analysis saved to {out_dir}")


if __name__ == "__main__":
    main()
