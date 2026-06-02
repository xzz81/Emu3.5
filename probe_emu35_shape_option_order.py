#!/usr/bin/env python3
"""Probe whether synthetic shape errors follow option labels or shape semantics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from datasets import load_from_disk
from tqdm import tqdm

try:
    _TORCHVISION_SCHEMA_LIB = torch.library.Library("torchvision", "DEF")
    _TORCHVISION_SCHEMA_LIB.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench_emu35_modal_memory import PeftModel, parse_mc_answer  # noqa: E402
from compare_emu35_task_entropy import (  # noqa: E402
    build_attr_question,
    build_image_understanding_prompt,
    options_for_attr,
)
from eval_emu35_synthetic_lora import load_model  # noqa: E402
from train_emu35_synthetic_lora import build_tokenizer  # noqa: E402
from src.utils.input_utils import build_image  # noqa: E402
from src.utils.entropy_metrics import (  # noqa: E402
    summarize_token_entropy_trace,
    text_token_entropy_trace,
    write_entropy_trace,
)
from src.vision_tokenizer import build_vision_tokenizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--vq-path", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("all", "train", "test"), default="all")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--image-area", type=int, default=384 * 384)
    parser.add_argument("--vq-type", default="ibq")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    return parser.parse_args()


def load_rows(dataset_path: str, split: str, max_samples: int) -> list[tuple[str, int, dict[str, Any]]]:
    loaded = load_from_disk(dataset_path)
    if split == "all":
        rows: list[tuple[str, int, dict[str, Any]]] = []
        for split_name in ("train", "test"):
            rows.extend((split_name, idx, dict(loaded[split_name][idx])) for idx in range(len(loaded[split_name])))
    else:
        rows = [(split, idx, dict(loaded[split][idx])) for idx in range(len(loaded[split]))]
    if max_samples > 0:
        rows = rows[:max_samples]
    return rows


def order_variants(options: list[str]) -> dict[str, list[str]]:
    return {
        "default": list(options),
        "reverse": list(reversed(options)),
        "rotate_left": list(options[1:] + options[:1]),
    }


@torch.no_grad()
def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = build_tokenizer(args.tokenizer_path)
    vq_model = build_vision_tokenizer(args.vq_type, args.vq_path, device=args.device)
    vq_model.eval().requires_grad_(False)
    model = load_model(args)
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    rows_path = out_dir / f"shape_option_order_worker{args.worker_id}.jsonl"
    done: set[str] = set()
    existing: list[dict[str, Any]] = []
    if rows_path.exists():
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            if line:
                row = json.loads(line)
                existing.append(row)
                done.add(row["sample_id"])

    items = load_rows(args.dataset, args.split, args.max_samples)
    work = items[args.worker_id :: args.num_workers]
    cfg = SimpleNamespace(image_area=args.image_area)
    base_options = options_for_attr("shape")
    variants = order_variants(base_options)
    written: list[dict[str, Any]] = []

    for split_name, idx, sample in tqdm(work, desc=f"shape-option-order:w{args.worker_id}"):
        image_string = build_image(sample["image"].convert("RGB"), cfg, tokenizer, vq_model)
        expected_value = sample["shape"]
        for variant_name, options in variants.items():
            sample_id = f"shape_order_{variant_name}_{split_name}_{idx:04d}"
            if sample_id in done:
                continue
            question, options_text = build_attr_question("shape", options)
            prompt = build_image_understanding_prompt(image_string, question)
            input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
            generated = model.generate(
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                return_dict_in_generate=True,
                output_scores=True,
            )
            completion_ids = generated.sequences[0, input_ids.shape[1] :]
            completion = tokenizer.decode(completion_ids, skip_special_tokens=False)
            answer_key, strict = parse_mc_answer(completion, len(options))
            answer_value = None
            if answer_key is not None:
                answer_index = ord(answer_key) - ord("A")
                if 0 <= answer_index < len(options):
                    answer_value = options[answer_index]
            expected_key = chr(ord("A") + options.index(expected_value))
            trace = text_token_entropy_trace(generated.scores, completion_ids, tokenizer)
            entropy_path, _ = write_entropy_trace(out_dir, sample_id, trace)
            output = {
                "sample_id": sample_id,
                "split": split_name,
                "dataset_index": idx,
                "variant": variant_name,
                "attribute": "shape",
                "expected_value": expected_value,
                "expected_key": expected_key,
                "options": options,
                "options_text": options_text,
                "inference_completion": completion,
                "grading_answer_key": answer_key,
                "answer_value": answer_value,
                "grading_format_correct": strict,
                "grading_correct": answer_value == expected_value,
                "label_correct": answer_key == expected_key,
                "entropy_path": entropy_path,
                "entropy_summary": summarize_token_entropy_trace(trace),
            }
            written.append(output)
            with rows_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(output, ensure_ascii=False) + "\n")

    if existing:
        combined = existing + written
        seen = set()
        unique = []
        for row in combined:
            if row["sample_id"] in seen:
                continue
            unique.append(row)
            seen.add(row["sample_id"])
        with rows_path.open("w", encoding="utf-8") as handle:
            for row in unique:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
