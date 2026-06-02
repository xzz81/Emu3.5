#!/usr/bin/env python3
# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Modal-aphasia style image-vs-text memory benchmark for Emu3.5 LoRA."""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import sys
from pathlib import Path
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

from datasets import Dataset, load_from_disk
from peft import PeftModel
from tqdm import tqdm
from transformers import GenerationConfig
from transformers.generation import LogitsProcessor, LogitsProcessorList

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODAL_APHASIA_ROOT = Path("/workspace/home/AAAI 2027/modal-aphasia")
if MODAL_APHASIA_ROOT.exists() and str(MODAL_APHASIA_ROOT) not in sys.path:
    sys.path.insert(0, str(MODAL_APHASIA_ROOT))

from eval_emu35_synthetic_lora import load_model  # noqa: E402
from train_emu35_synthetic_lora import build_prompt, build_tokenizer, synthetic_prompt  # noqa: E402
from src.utils.generation_utils import multimodal_decode  # noqa: E402
from src.utils.entropy_metrics import (  # noqa: E402
    attach_selected_tokens,
    entropy_from_logits,
    text_token_entropy_trace,
    write_entropy_trace,
)
from src.utils.logits_processor import BOV  # noqa: E402
from src.vision_tokenizer import build_vision_tokenizer  # noqa: E402


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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--mode", choices=("both", "image", "text", "summarize"), default="both")
    parser.add_argument("--image-split", default="all", choices=("all", "train", "test"))
    parser.add_argument("--image-samples", type=int, default=8)
    parser.add_argument("--image-offset", type=int, default=0)
    parser.add_argument("--image-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--max-new-tokens-image", type=int, default=620)
    parser.add_argument("--image-top-k", type=int, default=128)
    parser.add_argument("--image-temperature", type=float, default=0.8)
    parser.add_argument("--classifier-free-guidance", type=float, default=1.0)
    parser.add_argument("--max-new-tokens-text", type=int, default=16)
    parser.add_argument("--record-entropy", action="store_true")
    return parser.parse_args()


def image_to_base64(image: Image.Image) -> str:
    with io.BytesIO() as buffer:
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


def load_classifier_map():
    try:
        from modal_aphasia.evals import _synthetic_image_classifier as classifier
    except Exception as exc:
        print(f"[WARNING] Could not load modal_aphasia classifiers: {exc}", flush=True)
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
    result = {}
    for key, classifier in classifiers.items():
        try:
            result[key] = classifier.classify(arr)
        except Exception:
            result[key] = None
    return result


def build_text_mc_dataset(seed: int) -> Dataset:
    from modal_aphasia.data import constants

    prompt_template = (
        "Which of the following best matches {query_concept_value}?\n"
        "{options}\n"
        "Output a single letter, corresponding to the correct answer, nothing else."
    )
    rng = np.random.default_rng(seed)
    records = []
    for concept_type, mapping in constants.CONCEPT_TO_SYNTHETIC_MAP.items():
        (rng_concept_type,) = rng.spawn(1)
        for real_value, synthetic_value in mapping.items():
            (rng_value,) = rng_concept_type.spawn(1)
            for is_synthetic_query, query_value, expected_value, option_values in (
                (False, real_value, synthetic_value, mapping.values()),
                (True, synthetic_value, real_value, mapping.keys()),
            ):
                all_options = list(option_values)
                rng_value.shuffle(all_options)
                option_keys = [chr(ord("A") + idx) for idx in range(len(all_options))]
                options = "\n".join(f"{key}: {value}" for key, value in zip(option_keys, all_options))
                expected_key = option_keys[all_options.index(expected_value)]
                records.append(
                    {
                        "prompt": prompt_template.format(query_concept_value=query_value, options=options),
                        "expected_key": expected_key,
                        "options": all_options,
                        "concept_type": concept_type,
                        "concept_value": real_value,
                        "concept_value_synthetic": synthetic_value,
                        "is_synthetic_query": is_synthetic_query,
                    }
                )
    return Dataset.from_list(records)


def parse_mc_answer(text: str, num_options: int) -> tuple[str | None, bool]:
    cleaned = (
        text.replace("<|extra_101|>", " ")
        .replace("<|extra_204|>", " ")
        .replace("<|endoftext|>", " ")
        .strip()
    )
    valid = {chr(ord("A") + idx) for idx in range(num_options)}
    if len(cleaned) == 1 and cleaned.upper() in valid:
        return cleaned.upper(), True
    match = re.search(r"\b([A-Z])\b", cleaned.upper())
    if match and match.group(1) in valid:
        return match.group(1), False
    match = re.match(r"^\s*([A-Z])[\).:\s]", cleaned.upper())
    if match and match.group(1) in valid:
        return match.group(1), False
    return None, False


class BatchedFixedImageLogitsProcessor(LogitsProcessor):
    """Batch-safe fixed-size Emu3 image logits processor for CFG=1 evaluation."""

    def __init__(
        self,
        tokenizer: Any,
        height: int = 24,
        width: int = 24,
        image_top_k: int = 1,
        image_temperature: float = 1.0,
        record_entropy: bool = False,
    ):
        self.tokenizer = tokenizer
        self.height = int(height)
        self.width = int(width)
        self.image_top_k = int(image_top_k)
        self.image_temperature = float(image_temperature)
        self.record_entropy = bool(record_entropy)
        self.entropy_traces: list[list[dict[str, Any]]] | None = None
        self.boi = tokenizer.encode(tokenizer.boi_token, add_special_tokens=False)[0]
        self.img = tokenizer.encode(tokenizer.img_token, add_special_tokens=False)[0]
        self.eoi = tokenizer.encode(tokenizer.eoi_token, add_special_tokens=False)[0]
        self.eol = tokenizer.encode(tokenizer.eol_token, add_special_tokens=False)[0]
        self.hw_tokens = tokenizer.encode(f"{self.height}*{self.width}", add_special_tokens=False)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        processed = scores.clone()
        if self.record_entropy and self.entropy_traces is None:
            self.entropy_traces = [[] for _ in range(input_ids.shape[0])]
        for batch_idx in range(input_ids.shape[0]):
            seq = input_ids[batch_idx]
            row_scores = processed[batch_idx]
            phase = "text"
            visual_index = None
            boi_positions = (seq == self.boi).nonzero().flatten()
            eoi_positions = (seq == self.eoi).nonzero().flatten()
            in_image = bool(
                boi_positions.numel() > 0
                and (eoi_positions.numel() == 0 or boi_positions[-1].item() > eoi_positions[-1].item())
            )
            if not in_image:
                mask = torch.full_like(row_scores, -torch.inf)
                mask[:BOV] = 0
                row_scores = row_scores + mask
                processed[batch_idx] = row_scores
                if self.record_entropy and self.entropy_traces is not None:
                    entropy, normalized, finite_count = entropy_from_logits(row_scores)
                    self.entropy_traces[batch_idx].append(
                        {
                            "step": len(self.entropy_traces[batch_idx]),
                            "phase": phase,
                            "entropy": entropy,
                            "normalized_entropy": normalized,
                            "finite_token_count": finite_count,
                            "visual_index": visual_index,
                        }
                    )
                continue

            boi_idx = int(boi_positions[-1].item())
            img_positions = (seq == self.img).nonzero().flatten()
            in_visual = bool(img_positions.numel() > 0 and img_positions[-1].item() > boi_idx)
            mask = torch.full_like(row_scores, -torch.inf)
            if in_visual:
                img_idx = int(img_positions[-1].item())
                vis_idx = seq.shape[0] - img_idx
                if vis_idx == self.height * (self.width + 1):
                    mask[self.eoi] = 0
                    phase = "image_end"
                elif vis_idx % (self.width + 1) == 0:
                    mask[self.eol] = 0
                    phase = "image_eol"
                else:
                    mask[BOV:] = 0
                    phase = "visual"
                    visual_index = vis_idx - 1 - ((vis_idx - 1) // (self.width + 1))
                row_scores = row_scores + mask
                if self.image_temperature != 1.0:
                    row_scores = row_scores / self.image_temperature
                entropy_row = row_scores
                if self.image_top_k > 0 and self.image_top_k < row_scores.numel():
                    threshold = torch.topk(row_scores, k=self.image_top_k).values[-1]
                    row_scores = row_scores.masked_fill(row_scores < threshold, -torch.inf)
                processed[batch_idx] = row_scores
            else:
                hw_idx = seq.shape[0] - boi_idx
                if hw_idx <= len(self.hw_tokens):
                    mask[self.hw_tokens[hw_idx - 1]] = 0
                    phase = "image_size"
                else:
                    mask[self.img] = 0
                    phase = "image_token"
                processed[batch_idx] = row_scores + mask
                row_scores = processed[batch_idx]
                entropy_row = row_scores
            if self.record_entropy and self.entropy_traces is not None:
                entropy, normalized, finite_count = entropy_from_logits(entropy_row)
                post_entropy, post_normalized, post_finite_count = entropy_from_logits(row_scores)
                self.entropy_traces[batch_idx].append(
                    {
                        "step": len(self.entropy_traces[batch_idx]),
                        "phase": phase,
                        "entropy": entropy,
                        "normalized_entropy": normalized,
                        "finite_token_count": finite_count,
                        "post_sampling_entropy": post_entropy,
                        "post_sampling_normalized_entropy": post_normalized,
                        "post_sampling_finite_token_count": post_finite_count,
                        "visual_index": visual_index,
                    }
                )
        return processed


@torch.no_grad()
def generate_image_batch(
    model,
    tokenizer,
    prompts: list[str],
    args: argparse.Namespace,
) -> tuple[list[str], torch.Tensor, list[list[dict[str, Any]]] | None]:
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        batch = tokenizer(
            prompts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
    finally:
        tokenizer.padding_side = old_padding_side
    input_ids = batch["input_ids"].to(model.device)
    attention_mask = batch["attention_mask"].to(model.device)
    generation_config = GenerationConfig(
        use_cache=True,
        do_sample=True,
        top_k=131072,
        top_p=1.0,
        temperature=1.0,
        num_beams=1,
        max_new_tokens=args.max_new_tokens_image,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    logits_processor = BatchedFixedImageLogitsProcessor(
        tokenizer,
        height=24,
        width=24,
        image_top_k=args.image_top_k,
        image_temperature=args.image_temperature,
        record_entropy=args.record_entropy,
    )
    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        generation_config=generation_config,
        logits_processor=LogitsProcessorList([logits_processor]),
    )
    completion_ids = output_ids[:, input_ids.shape[1] :]
    return (
        [tokenizer.decode(row, skip_special_tokens=False) for row in completion_ids],
        completion_ids,
        logits_processor.entropy_traces,
    )


@torch.no_grad()
def run_text_bench(model, tokenizer, args: argparse.Namespace, out_dir: Path) -> list[dict[str, Any]]:
    dataset = build_text_mc_dataset(args.seed)
    rows = []
    for idx, row in enumerate(tqdm(dataset, desc="text-memory")):
        prompt = build_prompt(row["prompt"], "text")
        input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        generated = model.generate(
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens_text,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            return_dict_in_generate=args.record_entropy,
            output_scores=args.record_entropy,
        )
        if args.record_entropy:
            completion_ids = generated.sequences[0, input_ids.shape[1] :]
        else:
            completion_ids = generated[0, input_ids.shape[1] :]
        completion = tokenizer.decode(completion_ids, skip_special_tokens=False)
        answer_key, strict_format = parse_mc_answer(completion, len(row["options"]))
        output = dict(row)
        entropy_path = None
        entropy_summary = None
        if args.record_entropy:
            sample_id = f"text_{idx:03d}"
            trace = text_token_entropy_trace(generated.scores, completion_ids, tokenizer)
            entropy_path, entropy_summary = write_entropy_trace(out_dir, sample_id, trace)
        output.update(
            {
                "sample_id": f"text_{idx:03d}",
                "inference_completion": completion,
                "grading_answer_key": answer_key,
                "grading_format_correct": strict_format,
                "grading_correct": answer_key == row["expected_key"],
                "entropy_path": entropy_path,
                "entropy_summary": entropy_summary,
            }
        )
        rows.append(output)
    path = out_dir / f"text_memory_worker{args.worker_id}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


@torch.no_grad()
def run_image_bench(model, tokenizer, vq_model, args: argparse.Namespace, out_dir: Path) -> list[dict[str, Any]]:
    loaded = load_from_disk(args.dataset)
    if args.image_split == "all":
        all_items = []
        for split_name in ("train", "test"):
            all_items.extend((split_name, idx) for idx in range(len(loaded[split_name])))
    else:
        all_items = [(args.image_split, idx) for idx in range(len(loaded[args.image_split]))]
    start = args.image_offset
    stop = min(start + args.image_samples, len(all_items)) if args.image_samples > 0 else len(all_items)
    worker_items = all_items[start:stop][args.worker_id :: args.num_workers]
    classifiers = load_classifier_map()

    rows = []
    path = out_dir / f"image_memory_worker{args.worker_id}.jsonl"
    done_keys = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line:
                old = json.loads(line)
                rows.append(old)
                done_keys.add((old.get("split"), int(old.get("dataset_index"))))

    pending_items = [item for item in worker_items if item not in done_keys]
    batch_size = max(1, int(args.image_batch_size))
    progress = tqdm(total=len(pending_items), desc=f"image-memory:w{args.worker_id}")
    for batch_start in range(0, len(pending_items), batch_size):
        batch_items = pending_items[batch_start : batch_start + batch_size]
        batch_rows = [dict(loaded[split_name][sample_idx]) for split_name, sample_idx in batch_items]
        questions = [synthetic_prompt(row, args.prompt_template) for row in batch_rows]
        prompts = [build_prompt(question, "image") for question in questions]
        raw_outputs, completion_ids, entropy_traces = generate_image_batch(model, tokenizer, prompts, args)

        batch_outputs = []
        for batch_idx, ((split_name, sample_idx), row, question, prompt, raw) in enumerate(
            zip(batch_items, batch_rows, questions, prompts, raw_outputs)
        ):
            sample_id = f"image_{split_name}_{sample_idx:04d}"
            decoded_images = [payload for kind, payload in multimodal_decode(raw, tokenizer, vq_model) if kind == "image"]
            expected = {key: row[key] for key in ("color", "pattern", "position", "shape")}
            detected = {}
            image_path = None
            image_b64 = None
            if decoded_images:
                image = decoded_images[0]
                image_dir = out_dir / f"images_worker{args.worker_id}" / split_name
                image_dir.mkdir(parents=True, exist_ok=True)
                image_path = image_dir / f"image_{split_name}_{sample_idx:04d}.png"
                image.save(image_path)
                image_b64 = image_to_base64(image)
                detected = grade_image(image, classifiers)
            attr_correct = {key: detected.get(key) == value for key, value in expected.items()}
            entropy_path = None
            entropy_summary = None
            if args.record_entropy and entropy_traces is not None:
                trace = attach_selected_tokens(entropy_traces[batch_idx], completion_ids[batch_idx], tokenizer)
                entropy_path, entropy_summary = write_entropy_trace(out_dir, sample_id, trace)
            output = {
                "sample_id": sample_id,
                "dataset_index": sample_idx,
                "split": split_name,
                "prompt": question,
                "inference_prompt": prompt,
                "inference_raw": raw,
                "inference_image_base64": image_b64,
                "image_path": str(image_path) if image_path else None,
                **expected,
                **{f"synthetic_{key}": row[f"synthetic_{key}"] for key in ("color", "pattern", "position", "shape")},
                **{f"grading_detected_{key}": detected.get(key) for key in ("color", "pattern", "position", "shape")},
                **{f"grading_correct_{key}": attr_correct[key] for key in ("color", "pattern", "position", "shape")},
                "grading_all_correct": all(attr_correct.values()) if detected else False,
                "entropy_path": entropy_path,
                "entropy_summary": entropy_summary,
            }
            batch_outputs.append(output)
        rows.extend(batch_outputs)
        with path.open("a", encoding="utf-8") as handle:
            for output in batch_outputs:
                handle.write(json.dumps(output, ensure_ascii=False) + "\n")
        progress.update(len(batch_outputs))
    progress.close()
    if done_keys:
        rows = sorted(rows, key=lambda r: (str(r.get("split")), int(r.get("dataset_index"))))
        seen = set()
        unique_rows = []
        for row in rows:
            key = (row.get("split"), int(row.get("dataset_index")))
            if key in seen:
                continue
            unique_rows.append(row)
            seen.add(key)
        rows = unique_rows
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def summarize(out_dir: Path) -> dict[str, Any]:
    text_rows = []
    for path in sorted(out_dir.glob("text_memory_worker*.jsonl")):
        text_rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    image_rows = []
    for path in sorted(out_dir.glob("image_memory_worker*.jsonl")):
        image_rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)

    def counts(rows, key):
        vals = [bool(row.get(key)) for row in rows if row.get(key) is not None]
        return {
            "correct": int(sum(vals)),
            "total": int(len(vals)),
            "accuracy": float(np.mean(vals)) if vals else None,
        }

    def image_summary(rows):
        return {
            "num_rows": len(rows),
            "all_attribute_accuracy": counts(rows, "grading_all_correct"),
            "color_accuracy": counts(rows, "grading_correct_color"),
            "pattern_accuracy": counts(rows, "grading_correct_pattern"),
            "position_accuracy": counts(rows, "grading_correct_position"),
            "shape_accuracy": counts(rows, "grading_correct_shape"),
        }

    def text_attr_summary(rows):
        attr_rows = {}
        for attr in ("color", "pattern", "position", "shape"):
            selected = [row for row in rows if row.get("concept_type") == attr]
            attr_rows[attr] = counts(selected, "grading_correct")
        return attr_rows

    def entropy_summary(rows):
        summaries = [row.get("entropy_summary") for row in rows if row.get("entropy_summary")]
        if not summaries:
            return {"num_rows": 0}

        def mean_key(key):
            vals = [summary.get(key) for summary in summaries if summary.get(key) is not None]
            return float(np.mean(vals)) if vals else None

        return {
            "num_rows": len(summaries),
            "mean_entropy": mean_key("mean_entropy"),
            "mean_visual_entropy": mean_key("mean_visual_entropy"),
            "mean_text_entropy": mean_key("mean_text_entropy"),
            "mean_num_steps": mean_key("num_steps"),
            "mean_num_visual_steps": mean_key("num_visual_steps"),
            "mean_num_text_steps": mean_key("num_text_steps"),
        }

    text_synth = [row for row in text_rows if row.get("is_synthetic_query")]
    text_real = [row for row in text_rows if not row.get("is_synthetic_query")]
    summary = {
        "text": {
            "num_rows": len(text_rows),
            "synthetic_query_accuracy": counts(text_synth, "grading_correct"),
            "real_query_accuracy": counts(text_real, "grading_correct"),
            "overall_accuracy": counts(text_rows, "grading_correct"),
            "synthetic_query_by_attribute": text_attr_summary(text_synth),
            "real_query_by_attribute": text_attr_summary(text_real),
        },
        "image": image_summary(image_rows),
        "image_by_split": {
            split: image_summary([row for row in image_rows if row.get("split") == split])
            for split in ("train", "test")
        },
        "entropy": {
            "text": entropy_summary(text_rows),
            "image": entropy_summary(image_rows),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "summarize":
        print(json.dumps(summarize(out_dir), indent=2, ensure_ascii=False))
        return

    tokenizer = build_tokenizer(args.tokenizer_path)
    vq_model = None
    if args.mode in ("both", "image"):
        vq_model = build_vision_tokenizer(args.vq_type, args.vq_path, device=args.device)
        vq_model.eval().requires_grad_(False)

    model = load_model(args)
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    if args.mode in ("both", "text"):
        run_text_bench(model, tokenizer, args, out_dir)
    if args.mode in ("both", "image"):
        assert vq_model is not None
        run_image_bench(model, tokenizer, vq_model, args, out_dir)

    print(json.dumps(summarize(out_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
