#!/usr/bin/env python3
"""Generate images with entropy traces, then read them back and collect image attention."""

from __future__ import annotations

import argparse
import csv
import gc
import importlib as imp
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
import torch
from transformers import GenerationConfig
from transformers.generation import LogitsProcessor, LogitsProcessorList

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.generation_utils import generate_with_entropy_trace, multimodal_decode
from src.utils.input_utils import build_image
from src.utils.logits_processor import BOI, BOV, EOI, IMG
from src.utils.model_utils import build_emu3p5


class TextOnlyLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids, scores):
        scores[:, BOV:] = -math.inf
        scores[:, [BOI, IMG, EOI]] = -math.inf
        return scores


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/ume_real_t2i_small.py")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-prompts", type=int, default=3)
    parser.add_argument("--generation-max-new-tokens", type=int, default=None)
    parser.add_argument("--target-height", type=int, default=None)
    parser.add_argument("--target-width", type=int, default=None)
    parser.add_argument("--image-area", type=int, default=None)
    parser.add_argument("--generator-model-path", default=None)
    parser.add_argument("--reader-model-path", default="same")
    parser.add_argument("--vq-path", default=None)
    parser.add_argument("--reader-image-area", type=int, default=512 * 512)
    parser.add_argument("--classifier-free-guidance", type=float, default=None)
    parser.add_argument("--describe-prompt", default="Describe this image.")
    parser.add_argument("--describe-max-new-tokens", type=int, default=96)
    parser.add_argument("--generator-device", default=None)
    parser.add_argument("--reader-device", default=None)
    parser.add_argument("--vq-device", default=None)
    return parser.parse_args()


def load_cfg(path: str):
    cfg_name = Path(path).stem
    cfg_package = Path(path).parent.__str__().replace("/", ".")
    return imp.import_module(f".{cfg_name}", package=cfg_package)


def normalize_prompts(cfg, max_prompts):
    prompts = cfg.prompts
    if isinstance(prompts, dict):
        prompts = [(name, prompt) for name, prompt in prompts.items()]
    else:
        prompts = [(f"{idx:03d}", prompt) for idx, prompt in enumerate(prompts)]
    return prompts[:max_prompts] if max_prompts is not None else prompts


def write_jsonl(rows, path: Path):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(rows, path: Path):
    rows = list(rows)
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def prepare_t2i_prompt(cfg, tokenizer, vq_model, question, model_device):
    reference_image = None
    if not isinstance(question, str):
        if isinstance(question["reference_image"], list):
            reference_image = [Image.open(img).convert("RGB") for img in question["reference_image"]]
        else:
            reference_image = Image.open(question["reference_image"]).convert("RGB")
        question = question["prompt"]

    prompt = cfg.template.format(question=question)
    if reference_image is not None:
        if isinstance(reference_image, list):
            image_str = "".join(build_image(img, cfg, tokenizer, vq_model) for img in reference_image)
        else:
            image_str = build_image(reference_image, cfg, tokenizer, vq_model)
        prompt = prompt.replace("<|IMAGE|>", image_str)
        unc_prompt = cfg.unc_prompt.replace("<|IMAGE|>", image_str)
    else:
        unc_prompt = cfg.unc_prompt

    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(model_device)
    if input_ids[0, 0] != cfg.special_token_ids["BOS"]:
        bos = torch.tensor([[cfg.special_token_ids["BOS"]]], device=input_ids.device, dtype=input_ids.dtype)
        input_ids = torch.cat([bos, input_ids], dim=1)

    unconditional_ids = tokenizer.encode(unc_prompt, return_tensors="pt", add_special_tokens=False).to(model_device)
    full_unc_ids = None
    if hasattr(cfg, "img_unc_prompt"):
        full_unc_ids = tokenizer.encode(cfg.img_unc_prompt, return_tensors="pt", add_special_tokens=False).to(model_device)
    return question, input_ids, unconditional_ids, full_unc_ids


def first_decoded_image(result_text, tokenizer, vq_model):
    for kind, payload in multimodal_decode(result_text, tokenizer, vq_model):
        if kind == "image":
            return payload
    return None


def trim_to_first_image(result_tokens, records, eoi_token_id):
    for idx, token_id in enumerate(result_tokens):
        if int(token_id) == int(eoi_token_id):
            return result_tokens[: idx + 1], records[: idx + 1]
    return result_tokens, records


def summarize_entropy(records, sample_id):
    visual = [r for r in records if r.get("token_type") == "visual"]
    def mean_field(key):
        vals = [float(r[key]) for r in visual if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    return {
        "sample_id": sample_id,
        "visual_tokens": len(visual),
        "mean_u_tok_sample": mean_field("u_tok_sample"),
        "mean_u_tok_full": mean_field("u_tok_full"),
        "mean_ume_sample": mean_field("ume_sample"),
        "mean_ume_full": mean_field("ume_full"),
    }


def parse_grid_shape(image_string):
    match = re.search(r"<\|image start\|>(\d+)\*(\d+)<\|image token\|>", image_string)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def build_read_prompt(tokenizer, vq_model, image_path: Path, image_area: int, describe_prompt: str):
    cfg = SimpleNamespace(image_area=image_area)
    image = Image.open(image_path).convert("RGB")
    image_string = build_image(image, cfg, tokenizer, vq_model)
    grid_shape = parse_grid_shape(image_string)
    prompt = (
        f"{tokenizer.bos_token}You are a helpful assistant. USER: "
        f"{image_string} {describe_prompt} ASSISTANT:"
    )
    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
    return input_ids, grid_shape


def make_heatmap(values, shape, out_path: Path, scale=12):
    arr = np.asarray(values, dtype=np.float32).reshape(shape)
    lo, hi = float(arr.min()), float(arr.max())
    norm = (arr - lo) / (hi - lo + 1e-8)
    rgb = np.zeros((*norm.shape, 3), dtype=np.uint8)
    rgb[..., 0] = (255 * norm).astype(np.uint8)
    rgb[..., 1] = (70 * (1.0 - np.abs(norm - 0.5) * 2.0)).clip(0, 70).astype(np.uint8)
    rgb[..., 2] = (255 * (1.0 - norm)).astype(np.uint8)
    img = Image.fromarray(rgb).resize((shape[1] * scale, shape[0] * scale), Image.Resampling.NEAREST)
    img.save(out_path)


def make_contact_sheet(image_paths, out_path: Path, label_height=24):
    images = [Image.open(p).convert("RGB") for p in image_paths]
    if not images:
        return
    w, h = images[0].size
    cols = min(6, len(images))
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * (h + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, (path, image) in enumerate(zip(image_paths, images)):
        x = (idx % cols) * w
        y = (idx // cols) * (h + label_height)
        sheet.paste(image, (x, y + label_height))
        draw.text((x + 4, y + 4), Path(path).stem, fill=(0, 0, 0))
    sheet.save(out_path)


@torch.no_grad()
def describe_with_attention(
    model,
    tokenizer,
    vq_model,
    image_path: Path,
    out_dir: Path,
    image_area: int,
    describe_prompt: str,
    max_new_tokens: int,
):
    input_ids, grid_shape = build_read_prompt(tokenizer, vq_model, image_path, image_area, describe_prompt)
    input_ids = input_ids.to(model.device)
    input_len = input_ids.shape[1]
    visual_positions = (input_ids[0] >= BOV).nonzero().flatten().tolist()
    eoi_positions = (input_ids[0] == EOI).nonzero().flatten().tolist()
    if not visual_positions:
        raise RuntimeError(f"No visual tokens found in read prompt for {image_path}")
    if grid_shape is None:
        n = len(visual_positions)
        side = int(round(n ** 0.5))
        grid_shape = (side, side) if side * side == n else (1, n)

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
    answer_text = tokenizer.decode(answer_ids[0], skip_special_tokens=False)
    (out_dir / "description.txt").write_text(answer_text, encoding="utf-8")

    if answer_ids.shape[1] > 0:
        full_ids = torch.cat([input_ids, answer_ids.to(input_ids.device)], dim=1)
        query_positions = list(range(input_len, full_ids.shape[1]))
    else:
        full_ids = input_ids
        start = eoi_positions[-1] + 1 if eoi_positions else max(0, input_len - 8)
        query_positions = list(range(start, input_len))

    outputs = model(
        input_ids=full_ids,
        use_cache=False,
        output_attentions=True,
        return_dict=True,
    )

    layer_rows = []
    map_paths = []
    for layer_idx, attn in enumerate(outputs.attentions):
        attn = attn[0].float()
        q_idx = torch.tensor(query_positions, device=attn.device)
        v_idx = torch.tensor(visual_positions, device=attn.device)
        q_attn = attn.index_select(1, q_idx)
        image_mass = q_attn.index_select(2, v_idx)
        map_values = image_mass.mean(dim=(0, 1)).detach().cpu().numpy()
        mass_per_query = image_mass.sum(dim=2).mean().item()
        max_visual = float(map_values.max()) if map_values.size else 0.0
        mean_visual = float(map_values.mean()) if map_values.size else 0.0
        flat = map_values / (map_values.sum() + 1e-12)
        spatial_entropy = float(-(flat * np.log(flat + 1e-12)).sum() / math.log(len(flat))) if len(flat) > 1 else 0.0
        layer_rows.append({
            "layer": layer_idx,
            "image_attention_mass": mass_per_query,
            "mean_visual_token_attention": mean_visual,
            "max_visual_token_attention": max_visual,
            "spatial_attention_entropy": spatial_entropy,
            "query_tokens": len(query_positions),
            "visual_tokens": len(visual_positions),
            "grid_h": grid_shape[0],
            "grid_w": grid_shape[1],
        })
        if grid_shape[0] * grid_shape[1] == len(map_values):
            map_path = out_dir / f"layer_{layer_idx:02d}_image_attention.png"
            make_heatmap(map_values, grid_shape, map_path)
            map_paths.append(map_path)

    write_csv(layer_rows, out_dir / "layer_image_attention.csv")
    make_contact_sheet(map_paths, out_dir / "layer_attention_contact_sheet.png")
    return answer_text, layer_rows


def main():
    args = parse_args()
    cfg = load_cfg(args.cfg)
    cfg.rank = 0
    cfg.world_size = 1
    cfg.prompts = normalize_prompts(cfg, args.max_prompts)
    cfg.special_token_ids = None
    if args.generator_model_path is not None:
        cfg.model_path = args.generator_model_path
    if args.vq_path is not None:
        cfg.vq_path = args.vq_path
    if args.generation_max_new_tokens is not None:
        cfg.max_new_tokens = args.generation_max_new_tokens
        cfg.sampling_params["max_new_tokens"] = args.generation_max_new_tokens
    if args.target_height is not None:
        cfg.target_height = args.target_height
    if args.target_width is not None:
        cfg.target_width = args.target_width
    if args.image_area is not None:
        cfg.image_area = args.image_area
    if args.classifier_free_guidance is not None:
        cfg.classifier_free_guidance = args.classifier_free_guidance
    if args.generator_device is not None:
        cfg.hf_device = args.generator_device
    if args.vq_device is not None:
        cfg.vq_device = args.vq_device

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    out_root = Path(cfg.save_path) / "entropy_attention_runs" / run_id
    for sub in ("decoded", "entropy_traces", "raw_generations", "attention", "tables"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    reader_model_path = cfg.model_path if args.reader_model_path == "same" else args.reader_model_path
    same_model_readback = reader_model_path == cfg.model_path

    model_kwargs = dict(getattr(cfg, "diffusion_decoder_kwargs", {}))
    if same_model_readback:
        model_kwargs["attn_implementation"] = "eager"

    model, tokenizer, vq_model = build_emu3p5(
        cfg.model_path,
        cfg.tokenizer_path,
        cfg.vq_path,
        vq_type=cfg.vq_type,
        model_device=cfg.hf_device,
        vq_device=cfg.vq_device,
        **model_kwargs,
    )
    cfg.special_token_ids = {k: tokenizer.encode(v)[0] for k, v in cfg.special_tokens.items()}
    random.seed(cfg.seed)

    entropy_rows = []
    decoded_images = []
    for name, question in cfg.prompts:
        torch.cuda.empty_cache()
        prompt_text, input_ids, unconditional_ids, full_unc_ids = prepare_t2i_prompt(
            cfg, tokenizer, vq_model, question, model.device
        )
        result_tokens, records = generate_with_entropy_trace(
            cfg,
            model,
            tokenizer,
            input_ids,
            unconditional_ids,
            full_unc_ids,
            force_same_image_size=True,
            sample_id=name,
        )
        result_tokens, records = trim_to_first_image(result_tokens, records, cfg.special_token_ids["EOI"])
        result_text = tokenizer.decode(result_tokens, skip_special_tokens=False)
        write_jsonl(records, out_root / "entropy_traces" / f"{name}_entropy.jsonl")
        (out_root / "raw_generations" / f"{name}.txt").write_text(result_text, encoding="utf-8")
        image = first_decoded_image(result_text, tokenizer, vq_model)
        if image is None:
            print(f"[WARNING] no decoded image for {name}", flush=True)
            continue
        image_path = out_root / "decoded" / f"{name}.png"
        image.save(image_path)
        decoded_images.append((name, image_path))
        row = summarize_entropy(records, name)
        row["prompt"] = prompt_text
        row["image_path"] = str(image_path)
        entropy_rows.append(row)

    write_csv(entropy_rows, out_root / "tables" / "generation_entropy_summary.csv")

    if same_model_readback:
        reader_model, reader_tokenizer, reader_vq = model, tokenizer, vq_model
    else:
        del model
        del vq_model
        torch.cuda.empty_cache()
        gc.collect()

        reader_device = args.reader_device or cfg.hf_device
        reader_vq_device = args.vq_device or cfg.vq_device
        reader_model, reader_tokenizer, reader_vq = build_emu3p5(
            reader_model_path,
            cfg.tokenizer_path,
            cfg.vq_path,
            vq_type=cfg.vq_type,
            model_device=reader_device,
            vq_device=reader_vq_device,
            attn_implementation="eager",
        )
    attention_summary = []
    for name, image_path in decoded_images:
        sample_attention_dir = out_root / "attention" / name
        sample_attention_dir.mkdir(parents=True, exist_ok=True)
        answer, rows = describe_with_attention(
            reader_model,
            reader_tokenizer,
            reader_vq,
            image_path,
            sample_attention_dir,
            args.reader_image_area,
            args.describe_prompt,
            args.describe_max_new_tokens,
        )
        if rows:
            best = max(rows, key=lambda r: r["image_attention_mass"])
            attention_summary.append({
                "sample_id": name,
                "description": answer,
                "best_layer_by_image_mass": best["layer"],
                "best_layer_image_attention_mass": best["image_attention_mass"],
                "mean_layer_image_attention_mass": sum(r["image_attention_mass"] for r in rows) / len(rows),
                "attention_dir": str(sample_attention_dir),
            })

    write_csv(attention_summary, out_root / "tables" / "readback_attention_summary.csv")
    report_lines = [
        f"# Entropy-to-Attention Probe {run_id}",
        "",
        f"- generator_model: `{cfg.model_path}`",
        f"- reader_model: `{reader_model_path}`",
        f"- describe_prompt: `{args.describe_prompt}`",
        f"- output: `{out_root}`",
        "",
        "## Cases",
    ]
    for row in attention_summary:
        report_lines.extend([
            "",
            f"### {row['sample_id']}",
            f"- image: `{dict(decoded_images)[row['sample_id']]}`",
            f"- best_layer_by_image_mass: {row['best_layer_by_image_mass']}",
            f"- best_layer_image_attention_mass: {row['best_layer_image_attention_mass']:.6f}",
            f"- mean_layer_image_attention_mass: {row['mean_layer_image_attention_mass']:.6f}",
            f"- description: {row['description']}",
        ])
    (out_root / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"[INFO] entropy-attention run saved to {out_root}")


if __name__ == "__main__":
    main()
