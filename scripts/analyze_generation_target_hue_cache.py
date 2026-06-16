#!/usr/bin/env python3
# Copyright 2026
# SPDX-License-Identifier: Apache-2.0

"""Decode cached generation targets and score simple hue agreement."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os.path as osp
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import colorsys
import numpy as np
from PIL import Image
import torch
from transformers import AutoTokenizer

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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.generation_utils import multimodal_decode  # noqa: E402
from src.vision_tokenizer import build_vision_tokenizer  # noqa: E402


PALETTE_HUES = {
    "red": 0.0,
    "orange": 30.0,
    "yellow": 60.0,
    "green": 120.0,
    "cyan": 180.0,
    "blue": 240.0,
    "purple": 280.0,
    "magenta": 310.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/ume_main_t2i_counterfactual_pairs_hueonly_seed69.py")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--vq-device", default=None)
    parser.add_argument("--sat-threshold", type=float, default=0.25)
    parser.add_argument("--value-min", type=float, default=0.15)
    parser.add_argument("--value-max", type=float, default=0.98)
    return parser.parse_args()


def load_cfg(path: str):
    cfg_path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(cfg_path.stem, cfg_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"could not load config: {cfg_path}")
    sys.path.insert(0, str(REPO_ROOT))
    spec.loader.exec_module(module)
    return module


def build_tokenizer(tokenizer_path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        special_tokens_file=osp.join(tokenizer_path, "emu3_vision_tokens.txt"),
        trust_remote_code=True,
    )
    tokenizer.bos_token = "<|extra_203|>"
    tokenizer.eos_token = "<|extra_204|>"
    tokenizer.pad_token = "<|endoftext|>"
    tokenizer.eol_token = "<|extra_200|>"
    tokenizer.eof_token = "<|extra_201|>"
    tokenizer.tms_token = "<|extra_202|>"
    tokenizer.img_token = "<|image token|>"
    tokenizer.boi_token = "<|image start|>"
    tokenizer.eoi_token = "<|image end|>"
    tokenizer.bss_token = "<|extra_100|>"
    tokenizer.ess_token = "<|extra_101|>"
    tokenizer.bog_token = "<|extra_60|>"
    tokenizer.eog_token = "<|extra_61|>"
    tokenizer.boc_token = "<|extra_50|>"
    tokenizer.eoc_token = "<|extra_51|>"
    return tokenizer


def normalize_prompt_rows(cfg) -> Dict[str, dict]:
    metadata = getattr(cfg, "pair_metadata", {})
    rows = {}
    for sample_id, meta in metadata.items():
        value = str(meta.get("counterfactual_value", ""))
        parts = value.split()
        rows[sample_id] = {
            "sample_id": sample_id,
            "pair_id": meta.get("pair_id", ""),
            "variant": meta.get("variant", ""),
            "expected_color": parts[0] if parts else "",
            "shape": parts[1] if len(parts) > 1 else "",
            "counterfactual_value": value,
        }
    return rows


def hue_distance(a: float, b: float) -> float:
    delta = abs(float(a) - float(b)) % 360.0
    return min(delta, 360.0 - delta)


def classify_hue(image: Image.Image, sat_threshold: float, value_min: float, value_max: float) -> dict:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    flat = rgb.reshape(-1, 3)
    hsv = np.array([colorsys.rgb_to_hsv(float(r), float(g), float(b)) for r, g, b in flat], dtype=np.float32)
    hues = hsv[:, 0] * 360.0
    sat = hsv[:, 1]
    val = hsv[:, 2]
    mask = (sat >= sat_threshold) & (val >= value_min) & (val <= value_max)
    if int(mask.sum()) == 0:
        return {
            "predicted_color": "",
            "mean_hue_degrees": float("nan"),
            "hue_distance_degrees": float("nan"),
            "colorful_pixel_fraction": 0.0,
        }
    selected = hues[mask]
    angles = np.deg2rad(selected)
    mean_angle = math.atan2(float(np.sin(angles).mean()), float(np.cos(angles).mean()))
    mean_hue = (math.degrees(mean_angle) + 360.0) % 360.0
    predicted = min(PALETTE_HUES, key=lambda color: hue_distance(mean_hue, PALETTE_HUES[color]))
    return {
        "predicted_color": predicted,
        "mean_hue_degrees": mean_hue,
        "colorful_pixel_fraction": float(mask.mean()),
    }


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.cfg)
    tokenizer = build_tokenizer(cfg.tokenizer_path)
    vq_model = build_vision_tokenizer(
        cfg.vq_type,
        cfg.vq_path,
        device=args.vq_device or getattr(cfg, "vq_device", "cuda:0"),
        **getattr(cfg, "diffusion_decoder_kwargs", {}),
    )
    rows_by_sample = normalize_prompt_rows(cfg)
    out_dir = Path(args.out_dir)
    image_dir = out_dir / "decoded_targets"
    image_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for cache_path in sorted(Path(args.cache_dir).glob("*.json")):
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        metadata = payload.get("metadata", {})
        sample_id = metadata.get("sample_id", cache_path.stem)
        target_ids = payload.get("target_ids", [])
        result_text = tokenizer.decode(target_ids, skip_special_tokens=False)
        image = None
        for kind, decoded in multimodal_decode(result_text, tokenizer, vq_model):
            if kind == "image":
                image = decoded
                break
        base_row = rows_by_sample.get(sample_id, {"sample_id": sample_id})
        row = dict(base_row)
        row.update(
            {
                "cache_path": str(cache_path),
                "target_tokens": len(target_ids),
                "decoded": int(image is not None),
                "image_path": "",
                "predicted_color": "",
                "mean_hue_degrees": "",
                "expected_hue_degrees": "",
                "hue_distance_degrees": "",
                "colorful_pixel_fraction": "",
                "hue_match": 0,
            }
        )
        if image is not None:
            image_path = image_dir / f"{sample_id}.png"
            image.save(image_path)
            hue = classify_hue(image, args.sat_threshold, args.value_min, args.value_max)
            expected_color = row.get("expected_color", "")
            expected_hue = PALETTE_HUES.get(expected_color)
            distance = (
                hue_distance(float(hue["mean_hue_degrees"]), expected_hue)
                if expected_hue is not None and math.isfinite(float(hue["mean_hue_degrees"]))
                else float("nan")
            )
            row.update(
                {
                    "image_path": str(image_path),
                    "predicted_color": hue["predicted_color"],
                    "mean_hue_degrees": hue["mean_hue_degrees"],
                    "expected_hue_degrees": expected_hue if expected_hue is not None else "",
                    "hue_distance_degrees": distance,
                    "colorful_pixel_fraction": hue["colorful_pixel_fraction"],
                    "hue_match": int(hue["predicted_color"] == expected_color),
                }
            )
        rows.append(row)

    write_csv(out_dir / "target_hue_validation.csv", rows)
    decoded = [row for row in rows if int(row["decoded"])]
    match_count = sum(int(row["hue_match"]) for row in decoded)
    lines = [
        "# Generation Target Hue Cache Validation",
        "",
        "This report decodes cached clean generation targets and applies a simple HSV hue classifier to colorful pixels.",
        "",
        f"- Cache dir: `{args.cache_dir}`",
        f"- Decoded targets: {len(decoded)}/{len(rows)}",
        f"- Hue matches: {match_count}/{len(decoded)}" if decoded else "- Hue matches: n/a",
        "",
        "| sample | expected | predicted | mean hue | hue distance | colorful fraction |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {sample_id} | {expected_color} | {predicted_color} | {mean_hue_degrees} | "
            "{hue_distance_degrees} | {colorful_pixel_fraction} |".format(**row)
        )
    (out_dir / "target_hue_validation_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] decoded {len(decoded)}/{len(rows)} targets; hue matches {match_count}/{len(decoded) if decoded else 0}")


if __name__ == "__main__":
    main()
