#!/usr/bin/env python3
"""Patch-mask sensitivity for fixed, sample-specific visual answers."""

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
    import torchvision  # noqa: F401
except RuntimeError as exc:
    if "operator torchvision::nms does not exist" in str(exc):
        try:
            _TORCHVISION_SCHEMA_LIB = torch.library.Library("torchvision", "DEF")
            _TORCHVISION_SCHEMA_LIB.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
        except Exception:
            pass
except Exception:
    pass

from transformers.generation import LogitsProcessor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.input_utils import build_image
from src.utils.logits_processor import BOV, EOI
from src.utils.model_utils import build_emu3p5


DEFAULT_QA = {
    "t2i_topk64_mugs": {
        "question": "What color is the mug in the center?",
        "answer": "blue",
    },
    "t2i_topk64_sign": {
        "question": "What text is written on the sign?",
        "answer": "OPEN 24H",
    },
    "story_topk64_car_tree": {
        "question": "What is beside the tall green tree?",
        "answer": "a small red car",
    },
    "story_topk64_key_backpack": {
        "question": "What object is next to the blue backpack?",
        "answer": "a brass key",
    },
    "howto_topk64_seedling": {
        "question": "What color is the pot holding the seedling?",
        "answer": "red",
    },
    "howto_topk64_teapot": {
        "question": "What is on the wooden tray with the teapot?",
        "answer": "two clear cups",
    },
    "t2i_one_image_lantern_books": {
        "question": "What object is on top of the stacked purple books?",
        "answer": "a warm yellow lantern",
    },
    "t2i_one_image_bicycle_flowers": {
        "question": "What color is the bicycle leaning against the fence?",
        "answer": "blue",
    },
    "t2i_one_image_clock_cactus": {
        "question": "What is below the round black wall clock?",
        "answer": "a green cactus",
    },
    "t2i_one_image_cupcake_plate": {
        "question": "How many cupcakes are on the white plate?",
        "answer": "four",
    },
    "story_one_scene_girl_kite": {
        "question": "What color is the kite held by the child?",
        "answer": "red",
    },
    "story_one_scene_robot_cat": {
        "question": "What is the small silver robot offering?",
        "answer": "a blue ball",
    },
    "story_one_scene_boat_lighthouse": {
        "question": "What is near the tall striped lighthouse?",
        "answer": "a tiny green boat",
    },
    "story_one_scene_baker_window": {
        "question": "What is the baker holding?",
        "answer": "a tray of round bread",
    },
    "howto_one_step_thread_needle": {
        "question": "What color is the thread passing through the needle?",
        "answer": "red",
    },
    "howto_one_step_slice_lemon": {
        "question": "What fruit is being sliced on the cutting board?",
        "answer": "a yellow lemon",
    },
    "howto_one_step_wrap_gift": {
        "question": "What color is the wrapping paper around the gift box?",
        "answer": "green",
    },
    "howto_one_step_mix_paint": {
        "question": "What two paint colors are being mixed on the palette?",
        "answer": "blue and white",
    },
}


class TextOnlyLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids, scores):
        scores[:, BOV:] = -math.inf
        return scores


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--qa-spec", default=None)
    parser.add_argument("--replace-default-qa", action="store_true")
    parser.add_argument("--model-path", default="model/Emu3.5")
    parser.add_argument("--vq-path", default="model/Emu3.5-VisionTokenizer")
    parser.add_argument("--tokenizer-path", default="./src/tokenizer_emu3_ibq")
    parser.add_argument("--vq-type", default="ibq")
    parser.add_argument("--model-device", default="auto")
    parser.add_argument("--vq-device", default="cuda:0")
    parser.add_argument("--image-area", type=int, default=512 * 512)
    parser.add_argument("--entropy-field", default="ume_full")
    parser.add_argument("--mask-grid", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--out-dir-name", default="answer_specific_mask_delta")
    parser.add_argument("--shuffle-repeats", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260530)
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


def load_qa_spec(path: str | None, replace_default: bool = False):
    if path is None:
        return DEFAULT_QA
    with Path(path).open(encoding="utf-8") as f:
        data = json.load(f)
    replace_default = bool(replace_default or data.pop("__replace_defaults__", False))
    out = {} if replace_default else dict(DEFAULT_QA)
    out.update(data)
    return out


def sample_id_from_image_path(image_path: Path):
    stem = image_path.stem
    if stem.endswith("_image_00"):
        return stem[: -len("_image_00")]
    return stem


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


def build_qa_prompt(tokenizer, vq_model, image: Image.Image, image_area: int, question: str):
    cfg = SimpleNamespace(image_area=image_area)
    image_string = build_image(image.convert("RGB"), cfg, tokenizer, vq_model)
    prompt = (
        f"{tokenizer.bos_token}You are a helpful assistant. USER: "
        f"{image_string} Answer the question using only a short answer. "
        f"Question: {question} ASSISTANT:"
    )
    return tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)


def encode_answer(tokenizer, answer: str):
    text = " " + answer.strip()
    return tokenizer.encode(text, return_tensors="pt", add_special_tokens=False)


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


def compare_maps(signal_map, delta_map):
    x = np.asarray(signal_map, dtype=np.float64).reshape(-1)
    y = np.asarray(delta_map, dtype=np.float64).reshape(-1)
    k = max(1, int(round(0.2 * len(x))))
    top_x = set(np.argsort(x)[-k:].tolist())
    top_y = set(np.argsort(y)[-k:].tolist())
    rest_idx = [i for i in range(len(y)) if i not in top_x]
    top_vals = y[list(top_x)]
    rest_vals = y[rest_idx]
    return {
        "pearson": corr(x, y),
        "spearman": corr(rankdata(x), rankdata(y)),
        "top20_overlap": len(top_x & top_y) / k,
        "delta_on_signal_top20": float(top_vals.mean()),
        "delta_on_signal_rest80": float(np.asarray(rest_vals).mean()),
        "top20_delta_minus_rest": float(top_vals.mean() - np.asarray(rest_vals).mean()),
        "top20_delta_enrichment": float(top_vals.mean() / (np.asarray(rest_vals).mean() + 1e-12)),
    }


def baseline_stats(signal_map, delta_map, rng, repeats: int, prefix: str):
    observed = compare_maps(signal_map, delta_map)
    out = {
        f"{prefix}_pearson": observed["pearson"],
        f"{prefix}_spearman": observed["spearman"],
        f"{prefix}_top20_overlap": observed["top20_overlap"],
        f"{prefix}_top20_minus_rest": observed["top20_delta_minus_rest"],
        f"{prefix}_top20_enrichment": observed["top20_delta_enrichment"],
    }
    if repeats <= 0:
        return out
    flat = np.asarray(signal_map, dtype=np.float32).reshape(-1)
    pearsons = []
    spearmans = []
    overlaps = []
    for _ in range(repeats):
        shuffled = rng.permutation(flat).reshape(signal_map.shape)
        stats = compare_maps(shuffled, delta_map)
        pearsons.append(stats["pearson"])
        spearmans.append(stats["spearman"])
        overlaps.append(stats["top20_overlap"])
    pearsons_arr = np.asarray(pearsons, dtype=np.float64)
    spearmans_arr = np.asarray(spearmans, dtype=np.float64)
    overlaps_arr = np.asarray(overlaps, dtype=np.float64)
    out.update(
        {
            f"{prefix}_shuffle_mean_pearson": float(pearsons_arr.mean()),
            f"{prefix}_shuffle_p95_pearson": float(np.quantile(pearsons_arr, 0.95)),
            f"{prefix}_shuffle_ge_pearson_fraction": float((pearsons_arr >= observed["pearson"]).mean()),
            f"{prefix}_shuffle_mean_spearman": float(spearmans_arr.mean()),
            f"{prefix}_shuffle_p95_spearman": float(np.quantile(spearmans_arr, 0.95)),
            f"{prefix}_shuffle_ge_spearman_fraction": float((spearmans_arr >= observed["spearman"]).mean()),
            f"{prefix}_shuffle_mean_top20_overlap": float(overlaps_arr.mean()),
            f"{prefix}_shuffle_p95_top20_overlap": float(np.quantile(overlaps_arr, 0.95)),
            f"{prefix}_shuffle_ge_top20_overlap_fraction": float((overlaps_arr >= observed["top20_overlap"]).mean()),
        }
    )
    return out


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


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = run_dir / args.out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    qa_spec = load_qa_spec(args.qa_spec, args.replace_default_qa)
    rng = np.random.default_rng(args.seed)

    model, tokenizer, vq_model = build_emu3p5(
        args.model_path,
        args.tokenizer_path,
        args.vq_path,
        vq_type=args.vq_type,
        model_device=args.model_device,
        vq_device=args.vq_device,
    )

    image_paths = sorted((run_dir / "decoded").glob("*_image_00.png"))
    if args.max_samples > 0:
        image_paths = image_paths[: args.max_samples]

    cell_rows = []
    summary_rows = []
    for image_path in image_paths:
        sample_id = sample_id_from_image_path(image_path)
        if sample_id not in qa_spec:
            print(f"[WARNING] skip {sample_id}: no QA spec", flush=True)
            continue
        question = qa_spec[sample_id]["question"]
        answer = qa_spec[sample_id]["answer"]
        image = Image.open(image_path).convert("RGB")
        entropy_map = load_entropy_map(run_dir / "entropy_traces" / f"{sample_id}_entropy.jsonl", args.entropy_field)
        coarse_ume = block_mean(entropy_map, args.mask_grid)

        prompt_ids = build_qa_prompt(tokenizer, vq_model, image, args.image_area, question)
        answer_ids = encode_answer(tokenizer, answer)
        base_nll, base_entropy, answer_tokens = score_answer(model, prompt_ids, answer_ids)

        delta_nll = np.zeros((args.mask_grid, args.mask_grid), dtype=np.float32)
        delta_entropy = np.zeros((args.mask_grid, args.mask_grid), dtype=np.float32)
        for row in range(args.mask_grid):
            for col in range(args.mask_grid):
                masked = mask_image_cell(image, args.mask_grid, row, col)
                masked_prompt_ids = build_qa_prompt(tokenizer, vq_model, masked, args.image_area, question)
                masked_nll, masked_entropy, _ = score_answer(model, masked_prompt_ids, answer_ids)
                delta_nll[row, col] = masked_nll - base_nll
                delta_entropy[row, col] = masked_entropy - base_entropy
                cell_rows.append(
                    {
                        "sample_id": sample_id,
                        "question": question,
                        "target_answer": answer,
                        "row": row,
                        "col": col,
                        "coarse_ume": float(coarse_ume[row, col]),
                        "delta_nll": float(delta_nll[row, col]),
                        "delta_entropy": float(delta_entropy[row, col]),
                    }
                )

        save_heatmap(coarse_ume, out_dir / f"{sample_id}_coarse_ume_heatmap.png")
        save_heatmap(delta_nll, out_dir / f"{sample_id}_answer_delta_nll_heatmap.png")
        save_heatmap(delta_entropy, out_dir / f"{sample_id}_answer_delta_entropy_heatmap.png")

        nll_stats = baseline_stats(coarse_ume, delta_nll, rng, args.shuffle_repeats, "ume_delta_nll")
        ent_stats = baseline_stats(coarse_ume, delta_entropy, rng, args.shuffle_repeats, "ume_delta_entropy")
        row = {
            "sample_id": sample_id,
            "mask_grid": args.mask_grid,
            "shuffle_repeats": args.shuffle_repeats,
            "question": question,
            "target_answer": answer,
            "answer_tokens": answer_tokens,
            "base_nll": base_nll,
            "base_entropy": base_entropy,
            "mean_delta_nll": float(delta_nll.mean()),
            "mean_delta_entropy": float(delta_entropy.mean()),
        }
        row.update(nll_stats)
        row.update(ent_stats)
        summary_rows.append(row)

    write_csv(cell_rows, out_dir / "mask_cell_metrics.csv")
    write_csv(summary_rows, out_dir / "summary_answer_specific_mask_delta.csv")
    print(f"[INFO] answer-specific mask delta saved to {out_dir}")


if __name__ == "__main__":
    main()
