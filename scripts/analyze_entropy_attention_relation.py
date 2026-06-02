#!/usr/bin/env python3
"""Relate generation-time visual-token entropy to readback image attention."""

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
    parser.add_argument("--model-path", default="/workspace/data/models/Emu3.5/Emu3.5")
    parser.add_argument("--vq-path", default="/workspace/data/models/Emu3.5/Emu3.5-VisionTokenizer")
    parser.add_argument("--tokenizer-path", default="./src/tokenizer_emu3_ibq")
    parser.add_argument("--vq-type", default="ibq")
    parser.add_argument("--model-device", default="auto")
    parser.add_argument("--vq-device", default="cuda:0")
    parser.add_argument("--image-area", type=int, default=512 * 512)
    parser.add_argument("--describe-prompt", default="Describe this image.")
    parser.add_argument("--describe-max-new-tokens", type=int, default=64)
    parser.add_argument("--entropy-field", default="ume_full")
    return parser.parse_args()


def write_csv(rows, path: Path):
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_entropy_maps(trace_path: Path):
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

    maps = {}
    for field in ("u_tok_full", "u_tok_sample", "ume_full", "ume_sample"):
        vals = [float(r[field]) for r in visual if r.get(field) is not None]
        if len(vals) == len(visual):
            maps[field] = np.asarray(vals, dtype=np.float32).reshape(side, side)
    return maps


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


def build_read_prompt(tokenizer, vq_model, image_path: Path, image_area: int, describe_prompt: str):
    cfg = SimpleNamespace(image_area=image_area)
    image_string = build_image(Image.open(image_path).convert("RGB"), cfg, tokenizer, vq_model)
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


def save_heatmap(arr, path: Path, scale=12):
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


def scatter_image(x, y, path: Path, size=360):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    y = (y - y.min()) / (y.max() - y.min() + 1e-8)
    pad = 36
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle((pad, pad, size - pad, size - pad), outline=(0, 0, 0))
    for xi, yi in zip(x, y):
        px = pad + int(xi * (size - 2 * pad))
        py = size - pad - int(yi * (size - 2 * pad))
        draw.ellipse((px - 1, py - 1, px + 1, py + 1), fill=(40, 90, 200))
    draw.text((pad, size - pad + 8), "entropy", fill=(0, 0, 0))
    draw.text((4, pad - 22), "attention", fill=(0, 0, 0))
    img.save(path)


@torch.no_grad()
def collect_attention_maps(model, tokenizer, vq_model, image_path: Path, image_area: int, describe_prompt: str, max_new_tokens: int):
    input_ids, grid_shape = build_read_prompt(tokenizer, vq_model, image_path, image_area, describe_prompt)
    input_ids = input_ids.to(model.device)
    input_len = input_ids.shape[1]
    visual_positions = (input_ids[0] >= BOV).nonzero().flatten().tolist()
    if grid_shape is None:
        side = int(round(len(visual_positions) ** 0.5))
        grid_shape = (side, side)

    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated = model.generate(
        input_ids,
        generation_config=generation_config,
        logits_processor=LogitsProcessorList([TextOnlyLogitsProcessor()]),
    )
    answer_ids = generated[:, input_len:]
    full_ids = torch.cat([input_ids, answer_ids.to(input_ids.device)], dim=1)
    query_positions = list(range(input_len, full_ids.shape[1]))
    description = tokenizer.decode(answer_ids[0], skip_special_tokens=False)

    outputs = model(input_ids=full_ids, use_cache=False, output_attentions=True, return_dict=True)
    maps = []
    layer_masses = []
    for attn in outputs.attentions:
        attn = attn[0].float()
        q_idx = torch.tensor(query_positions, device=attn.device)
        v_idx = torch.tensor(visual_positions, device=attn.device)
        q_attn = attn.index_select(1, q_idx)
        image_attn = q_attn.index_select(2, v_idx)
        values = image_attn.mean(dim=(0, 1)).detach().cpu().numpy()
        maps.append(values.reshape(grid_shape))
        layer_masses.append(float(image_attn.sum(dim=2).mean().item()))
    return description, maps, layer_masses


def make_panel(sample_id, image_path, entropy_map, attention_map, scatter_path, out_path):
    original = Image.open(image_path).convert("RGB").resize((256, 256))
    entropy_img = Image.open(out_path.parent / f"{sample_id}_entropy_heatmap.png").convert("RGB").resize((256, 256))
    attention_img = Image.open(out_path.parent / f"{sample_id}_attention_best_corr_heatmap.png").convert("RGB").resize((256, 256))
    scatter = Image.open(scatter_path).convert("RGB").resize((256, 256))
    sheet = Image.new("RGB", (256 * 4, 292), "white")
    draw = ImageDraw.Draw(sheet)
    labels = ["image", "gen entropy", "read attention", "scatter"]
    for idx, (label, im) in enumerate(zip(labels, [original, entropy_img, attention_img, scatter])):
        x = idx * 256
        draw.text((x + 8, 8), label, fill=(0, 0, 0))
        sheet.paste(im, (x, 32))
    sheet.save(out_path)


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = run_dir / "relation_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, vq_model = build_emu3p5(
        args.model_path,
        args.tokenizer_path,
        args.vq_path,
        vq_type=args.vq_type,
        model_device=args.model_device,
        vq_device=args.vq_device,
        attn_implementation="eager",
    )

    layer_rows = []
    summary_rows = []
    panels = []
    for image_path in sorted((run_dir / "decoded").glob("*.png")):
        sample_id = sample_id_from_image_path(image_path)
        entropy_maps = load_entropy_maps(run_dir / "entropy_traces" / f"{sample_id}_entropy.jsonl")
        if args.entropy_field not in entropy_maps:
            raise KeyError(
                f"{args.entropy_field} not available in {sample_id}; "
                f"available fields: {sorted(entropy_maps)}"
            )
        entropy_map = entropy_maps[args.entropy_field]
        save_heatmap(entropy_map, out_dir / f"{sample_id}_entropy_heatmap.png")

        description, attention_maps, layer_masses = collect_attention_maps(
            model,
            tokenizer,
            vq_model,
            image_path,
            args.image_area,
            args.describe_prompt,
            args.describe_max_new_tokens,
        )

        x = entropy_map.reshape(-1)
        k = max(1, int(round(0.2 * len(x))))
        top_entropy = set(np.argsort(x)[-k:].tolist())
        rows = []
        for layer, attention_map in enumerate(attention_maps):
            y = attention_map.reshape(-1)
            top_attention = set(np.argsort(y)[-k:].tolist())
            overlap = len(top_entropy & top_attention)
            top_vals = y[list(top_entropy)]
            rest_idx = [i for i in range(len(y)) if i not in top_entropy]
            rest_vals = y[rest_idx]
            row = {
                "sample_id": sample_id,
                "layer": layer,
                "pearson_entropy_attention": corr(x, y),
                "spearman_entropy_attention": corr(rankdata(x), rankdata(y)),
                "top20_overlap_count": overlap,
                "top20_overlap_fraction": overlap / k,
                "attention_on_entropy_top20": float(top_vals.mean()),
                "attention_on_entropy_rest80": float(rest_vals.mean()),
                "top20_attention_enrichment": float(top_vals.mean() / (rest_vals.mean() + 1e-12)),
                "image_attention_mass": layer_masses[layer],
            }
            rows.append(row)
            layer_rows.append(row)
            np.save(out_dir / f"{sample_id}_layer_{layer:02d}_attention.npy", attention_map)
        best = max(rows, key=lambda r: abs(r["pearson_entropy_attention"]))
        best_layer = int(best["layer"])
        best_map = attention_maps[best_layer]
        save_heatmap(best_map, out_dir / f"{sample_id}_attention_best_corr_heatmap.png")
        scatter_path = out_dir / f"{sample_id}_entropy_attention_scatter.png"
        scatter_image(x, best_map.reshape(-1), scatter_path)
        panel_path = out_dir / f"{sample_id}_entropy_attention_panel.png"
        make_panel(sample_id, image_path, entropy_map, best_map, scatter_path, panel_path)
        panels.append(panel_path)
        summary = dict(best)
        summary["best_abs_corr_layer"] = best_layer
        summary["description"] = description
        summary_rows.append(summary)

    write_csv(layer_rows, out_dir / "layer_entropy_attention_correlation.csv")
    write_csv(summary_rows, out_dir / "summary_best_layers.csv")
    if panels:
        images = [Image.open(p).convert("RGB") for p in panels]
        sheet = Image.new("RGB", (images[0].width, images[0].height * len(images)), "white")
        for idx, im in enumerate(images):
            sheet.paste(im, (0, idx * im.height))
        sheet.save(out_dir / "entropy_attention_relation_overview.png")
    print(f"[INFO] relation analysis saved to {out_dir}")


if __name__ == "__main__":
    main()
