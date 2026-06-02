#!/usr/bin/env python3
# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate Emu3.5 synthetic-concept LoRA adapters.

The script reports two quick signals:
1. Teacher-forced target loss on train/test synthetic image targets.
2. A small deterministic generation set, decoded to images and graded with the
   modal-aphasia synthetic image classifiers when available.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image
import torch

try:
    _TORCHVISION_SCHEMA_LIB = torch.library.Library("torchvision", "DEF")
    _TORCHVISION_SCHEMA_LIB.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
except Exception:
    pass

from datasets import load_from_disk
from peft import PeftModel
from tqdm import tqdm
from transformers import BitsAndBytesConfig, GenerationConfig

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_emu35_synthetic_lora import (  # noqa: E402
    build_prompt,
    build_tokenizer,
    encode_image_to_string,
    encode_labeled_example,
    load_split,
    synthetic_prompt,
)
from src.emu3p5 import Emu3Config, Emu3ForCausalLM  # noqa: E402
from src.utils.generation_utils import multimodal_decode, non_streaming_generate  # noqa: E402
from src.vision_tokenizer import build_vision_tokenizer  # noqa: E402


SPECIAL_TOKENS = {
    "BOS": "<|extra_203|>",
    "EOS": "<|extra_204|>",
    "PAD": "<|endoftext|>",
    "EOL": "<|extra_200|>",
    "EOF": "<|extra_201|>",
    "TMS": "<|extra_202|>",
    "IMG": "<|image token|>",
    "BOI": "<|image start|>",
    "EOI": "<|image end|>",
    "BSS": "<|extra_100|>",
    "ESS": "<|extra_101|>",
    "BOG": "<|extra_60|>",
    "EOG": "<|extra_61|>",
    "BOC": "<|extra_50|>",
    "EOC": "<|extra_51|>",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--vq-path", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-template", choices=("words_only", "typed", "sentence"), default="words_only")
    parser.add_argument("--image-area", type=int, default=384 * 384)
    parser.add_argument("--vq-type", default="ibq")
    parser.add_argument("--eval-samples", type=int, default=16)
    parser.add_argument("--generate-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--compare-base", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=620)
    parser.add_argument("--image-top-k", type=int, default=128)
    parser.add_argument("--image-temperature", type=float, default=0.8)
    parser.add_argument("--classifier-free-guidance", type=float, default=1.0)
    return parser.parse_args()


def load_dataset_dict(path: str):
    dataset = load_from_disk(path)
    if not hasattr(dataset, "keys"):
        raise ValueError("Expected a DatasetDict with train/test splits.")
    return dataset


def load_model(args: argparse.Namespace):
    kwargs: dict[str, Any] = {
        "config": Emu3Config.from_pretrained(args.model_path, trust_remote_code=True),
        "torch_dtype": torch.bfloat16,
        "attn_implementation": args.attn_implementation,
        "trust_remote_code": True,
    }
    if args.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": args.device}
    else:
        kwargs["device_map"] = {"": args.device}
    model = Emu3ForCausalLM.from_pretrained(args.model_path, **kwargs)
    model.eval()
    return model


def build_target(row: dict[str, Any], args: argparse.Namespace, tokenizer: Any, vq_model: torch.nn.Module) -> str:
    image_string = encode_image_to_string(row["image"], args.image_area, tokenizer, vq_model)
    return f"{image_string}{tokenizer.ess_token}{tokenizer.eos_token}"


def row_to_labeled(row: dict[str, Any], args: argparse.Namespace, tokenizer: Any, vq_model: torch.nn.Module):
    question = synthetic_prompt(row, args.prompt_template)
    prompt = build_prompt(question, "image")
    target = build_target(row, args, tokenizer, vq_model)
    return encode_labeled_example(tokenizer, prompt, target)


@torch.no_grad()
def mean_target_loss(
    model: torch.nn.Module,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    tokenizer: Any,
    vq_model: torch.nn.Module,
    label: str,
) -> dict[str, float]:
    losses = []
    token_counts = []
    for row in tqdm(rows, desc=f"loss:{label}"):
        ex = row_to_labeled(row, args, tokenizer, vq_model)
        input_ids = torch.tensor([ex["input_ids"]], dtype=torch.long, device=model.device)
        labels = torch.tensor([ex["labels"]], dtype=torch.long, device=model.device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
        losses.append(float(out.loss.detach().cpu()))
        token_counts.append(int(ex["target_len"]))
    return {
        "examples": len(rows),
        "mean_loss": float(np.mean(losses)) if losses else math.nan,
        "perplexity": float(math.exp(np.mean(losses))) if losses else math.nan,
        "mean_target_tokens": float(np.mean(token_counts)) if token_counts else math.nan,
    }


def build_generation_cfg(args: argparse.Namespace, tokenizer: Any):
    cfg = SimpleNamespace()
    cfg.special_tokens = SPECIAL_TOKENS
    cfg.special_token_ids = {k: tokenizer.encode(v, add_special_tokens=False)[0] for k, v in SPECIAL_TOKENS.items()}
    cfg.unconditional_type = "no_text"
    cfg.classifier_free_guidance = args.classifier_free_guidance
    cfg.image_cfg_scale = 1.0
    cfg.target_height = 24
    cfg.target_width = 24
    cfg.stop_after_completed_images = 1
    cfg.stop_after_eoi_extra_tokens = 0
    cfg.sampling_params = {
        "use_cache": True,
        "do_sample": True,
        "top_k": 131072,
        "top_p": 1.0,
        "temperature": 1.0,
        "num_beams": 1,
        "num_beam_groups": 1,
        "diversity_penalty": 0.0,
        "max_new_tokens": args.max_new_tokens,
        "guidance_scale": 1.0,
        "use_differential_sampling": True,
        "text_top_k": 128,
        "text_top_p": 0.9,
        "text_temperature": 0.8,
        "image_top_k": args.image_top_k,
        "image_top_p": 1.0,
        "image_temperature": args.image_temperature,
    }
    return cfg


def classifier_map():
    try:
        from modal_aphasia.evals import _synthetic_image_classifier as classifier
    except Exception:
        return None
    return {
        "color": classifier.ColorClassifier(),
        "pattern": classifier.PatternClassifier(),
        "position": classifier.PositionClassifier(),
        "shape": classifier.ShapeClassifier(),
    }


def grade_image(image: Image.Image, classifiers: dict[str, Any] | None) -> dict[str, str | None]:
    if classifiers is None:
        return {}
    arr = np.array(image.convert("RGB"))
    out = {}
    for name, clf in classifiers.items():
        try:
            out[name] = clf.classify(arr)
        except Exception:
            out[name] = None
    return out


@torch.no_grad()
def generate_samples(
    model: torch.nn.Module,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    tokenizer: Any,
    vq_model: torch.nn.Module,
    out_dir: Path,
) -> list[dict[str, Any]]:
    cfg = build_generation_cfg(args, tokenizer)
    unc_prompt = f"{tokenizer.bos_token}You are a helpful assistant. USER:  ASSISTANT: {tokenizer.bss_token}"
    unconditional_ids = tokenizer.encode(unc_prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    classifiers = classifier_map()
    records = []
    for idx, row in enumerate(tqdm(rows, desc="generate")):
        question = synthetic_prompt(row, args.prompt_template)
        prompt = build_prompt(question, "image")
        input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        gen_ids = non_streaming_generate(
            cfg,
            model,
            tokenizer,
            input_ids,
            unconditional_ids,
            force_same_image_size=True,
        )
        raw = tokenizer.decode(gen_ids, skip_special_tokens=False)
        sample_id = f"test_{idx:03d}"
        (out_dir / f"{sample_id}.txt").write_text(raw, encoding="utf-8")
        images = [payload for kind, payload in multimodal_decode(raw, tokenizer, vq_model) if kind == "image"]
        detected = {}
        image_path = None
        if images:
            image_path = out_dir / f"{sample_id}.png"
            images[0].save(image_path)
            detected = grade_image(images[0], classifiers)
        expected = {k: row[k] for k in ("color", "pattern", "position", "shape")}
        record = {
            "sample_id": sample_id,
            "question": question,
            "expected": expected,
            "detected": detected,
            "all_correct": bool(detected) and all(detected.get(k) == v for k, v in expected.items()),
            "image_path": str(image_path) if image_path else None,
            "raw_prefix": raw[:200],
        }
        records.append(record)
    return records


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = build_tokenizer(args.tokenizer_path)
    vq_model = build_vision_tokenizer(args.vq_type, args.vq_path, device=args.device)
    vq_model.eval().requires_grad_(False)

    dataset = load_dataset_dict(args.dataset)
    train_rows = [dict(dataset["train"][i]) for i in range(min(args.eval_samples, len(dataset["train"])))]
    test_rows = [dict(dataset["test"][i]) for i in range(min(args.eval_samples, len(dataset["test"])))]
    gen_rows = [dict(dataset["test"][i]) for i in range(min(args.generate_samples, len(dataset["test"])))]

    model = load_model(args)
    results: dict[str, Any] = {
        "args": vars(args),
        "dataset": {"train": len(dataset["train"]), "test": len(dataset["test"])},
    }

    if args.compare_base:
        results["base"] = {
            "train": mean_target_loss(model, train_rows, args, tokenizer, vq_model, "base_train"),
            "test": mean_target_loss(model, test_rows, args, tokenizer, vq_model, "base_test"),
        }

    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()
    results["adapter"] = {
        "train": mean_target_loss(model, train_rows, args, tokenizer, vq_model, "adapter_train"),
        "test": mean_target_loss(model, test_rows, args, tokenizer, vq_model, "adapter_test"),
    }
    if args.generate_samples > 0:
        results["generations"] = generate_samples(model, gen_rows, args, tokenizer, vq_model, out_dir)

    (out_dir / "eval_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
