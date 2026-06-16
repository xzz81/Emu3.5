#!/usr/bin/env python3
"""Compare entropy across text QA, image generation, and image understanding."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from datasets import load_from_disk
from tqdm import tqdm

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

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench_emu35_modal_memory import PeftModel, parse_mc_answer  # noqa: E402
from eval_emu35_synthetic_lora import load_model  # noqa: E402
from train_emu35_synthetic_lora import SPECIAL_TOKENS, build_tokenizer  # noqa: E402
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
    parser.add_argument("--bench-dir", required=True, help="Directory with text/image generation entropy benchmark.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=("image_understanding", "summarize"), default="image_understanding")
    parser.add_argument("--split", choices=("all", "train", "test"), default="all")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all selected images.")
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


def system_prompt() -> str:
    return "You are a helpful assistant for image understanding task."


def build_image_understanding_prompt(image_string: str, question: str) -> str:
    return (
        f"{SPECIAL_TOKENS['bos_token']}{system_prompt()} "
        f"USER: {image_string}\n{question} ASSISTANT: {SPECIAL_TOKENS['bss_token']}"
    )


def options_for_attr(attr: str) -> list[str]:
    from modal_aphasia.data import constants

    return list(constants.CONCEPT_TO_SYNTHETIC_MAP[attr].keys())


def build_attr_question(attr: str, options: list[str]) -> tuple[str, str]:
    letters = [chr(ord("A") + idx) for idx in range(len(options))]
    options_text = "\n".join(f"{letter}: {value}" for letter, value in zip(letters, options))
    noun = "object" if attr in {"color", "shape"} else "image"
    question = (
        f"Which {attr} is shown in the {noun}?\n"
        f"{options_text}\n"
        "Output a single letter, corresponding to the correct answer, nothing else."
    )
    return question, options_text


def load_rows_for_understanding(dataset_path: str, split: str, max_samples: int) -> list[tuple[str, int, dict[str, Any]]]:
    loaded = load_from_disk(dataset_path)
    if split == "all":
        pairs = []
        for split_name in ("train", "test"):
            pairs.extend((split_name, idx, dict(loaded[split_name][idx])) for idx in range(len(loaded[split_name])))
    else:
        pairs = [(split, idx, dict(loaded[split][idx])) for idx in range(len(loaded[split]))]
    if max_samples > 0:
        pairs = pairs[:max_samples]
    return pairs


@torch.no_grad()
def run_image_understanding(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = build_tokenizer(args.tokenizer_path)
    vq_model = build_vision_tokenizer(args.vq_type, args.vq_path, device=args.device)
    vq_model.eval().requires_grad_(False)
    model = load_model(args)
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    rows_path = out_dir / f"image_understanding_worker{args.worker_id}.jsonl"
    done = set()
    existing = []
    if rows_path.exists():
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            if line:
                row = json.loads(line)
                existing.append(row)
                done.add(row["sample_id"])

    items = load_rows_for_understanding(args.dataset, args.split, args.max_samples)
    work = items[args.worker_id :: args.num_workers]
    cfg = SimpleNamespace(image_area=args.image_area)
    all_outputs = []
    attr_options = {attr: options_for_attr(attr) for attr in ("color", "pattern", "position", "shape")}

    for split_name, idx, sample in tqdm(work, desc=f"image-understanding:w{args.worker_id}"):
        image_string = build_image(sample["image"].convert("RGB"), cfg, tokenizer, vq_model)
        for attr, options in attr_options.items():
            sample_id = f"understand_{split_name}_{idx:04d}_{attr}"
            if sample_id in done:
                continue
            question, options_text = build_attr_question(attr, options)
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
            expected_key = chr(ord("A") + options.index(sample[attr]))
            trace = text_token_entropy_trace(generated.scores, completion_ids, tokenizer)
            entropy_path, _ = write_entropy_trace(out_dir, sample_id, trace)
            output = {
                "sample_id": sample_id,
                "split": split_name,
                "dataset_index": idx,
                "task": "image_understanding",
                "attribute": attr,
                "expected_value": sample[attr],
                "expected_key": expected_key,
                "options": options,
                "options_text": options_text,
                "inference_completion": completion,
                "grading_answer_key": answer_key,
                "grading_format_correct": strict,
                "grading_correct": answer_key == expected_key,
                "entropy_path": entropy_path,
                "entropy_summary": summarize_token_entropy_trace(trace),
            }
            all_outputs.append(output)
            with rows_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(output, ensure_ascii=False) + "\n")

    if existing:
        combined = existing + all_outputs
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


def summarize_task_entropy(args: argparse.Namespace) -> dict[str, Any]:
    bench_dir = Path(args.bench_dir)
    out_dir = Path(args.output_dir)
    text_rows = []
    for path in sorted(bench_dir.glob("text_memory_worker*.jsonl")):
        text_rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    image_rows = []
    for path in sorted(bench_dir.glob("image_memory_worker*.jsonl")):
        image_rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    understanding_rows = []
    for path in sorted(out_dir.glob("image_understanding_worker*.jsonl")):
        understanding_rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)

    def collect(rows: list[dict[str, Any]], key: str) -> list[float]:
        vals = []
        for row in rows:
            summary = row.get("entropy_summary") or {}
            value = summary.get(key)
            if value is not None:
                vals.append(float(value))
        return vals

    def stats(vals: list[float]) -> dict[str, Any]:
        arr = np.asarray(vals, dtype=float)
        if arr.size == 0:
            return {"n": 0}
        return {
            "n": int(arr.size),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "max": float(np.max(arr)),
        }

    text_first = collect(text_rows, "mean_text_entropy")
    # Existing text rows store mean across three generated tokens; first-token values are in entropy files.
    text_first = []
    for row in text_rows:
        path = row.get("entropy_path")
        if path and Path(path).exists():
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            steps = payload.get("steps", [])
            if steps:
                text_first.append(float(steps[0]["entropy"]))

    image_visual = [float((row.get("entropy_summary") or {}).get("mean_visual_entropy")) for row in image_rows if (row.get("entropy_summary") or {}).get("mean_visual_entropy") is not None]
    understanding_first = collect(understanding_rows, "first_token_entropy")
    understanding_mean = collect(understanding_rows, "mean_entropy")

    def acc(rows: list[dict[str, Any]]) -> dict[str, Any]:
        vals = [bool(row.get("grading_correct")) for row in rows]
        return {"correct": int(sum(vals)), "total": len(vals), "accuracy": float(np.mean(vals)) if vals else None}

    by_attr = {}
    for attr in ("color", "pattern", "position", "shape"):
        attr_rows = [row for row in understanding_rows if row.get("attribute") == attr]
        by_attr[attr] = {
            "accuracy": acc(attr_rows),
            "first_token_entropy": stats(collect(attr_rows, "first_token_entropy")),
        }

    summary = {
        "source_bench_dir": str(bench_dir),
        "output_dir": str(out_dir),
        "text_question_answer_first_token_entropy": stats(text_first),
        "image_generation_mean_visual_token_entropy": stats(image_visual),
        "image_understanding_first_answer_token_entropy": stats(understanding_first),
        "image_understanding_mean_answer_entropy": stats(understanding_mean),
        "image_understanding_accuracy": acc(understanding_rows),
        "image_understanding_by_attribute": by_attr,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "task_entropy_comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out_dir / "task_entropy_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("task", "entropy_measure", "n", "mean", "median", "p90", "p95", "max"))
        writer.writeheader()
        for task, measure, values in (
            ("text_question_answer", "first_answer_token_entropy", summary["text_question_answer_first_token_entropy"]),
            ("image_generation", "mean_visual_token_entropy_per_image", summary["image_generation_mean_visual_token_entropy"]),
            ("image_understanding", "first_answer_token_entropy", summary["image_understanding_first_answer_token_entropy"]),
            ("image_understanding", "mean_answer_token_entropy", summary["image_understanding_mean_answer_entropy"]),
        ):
            writer.writerow({"task": task, "entropy_measure": measure, **values})
    return summary


def main() -> None:
    args = parse_args()
    if args.mode == "image_understanding":
        run_image_understanding(args)
    summary = summarize_task_entropy(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
