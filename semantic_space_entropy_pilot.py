#!/usr/bin/env python3
"""Pilot semantic-space entropy comparison for text vs image channels."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from PIL import Image
import torch

try:
    _TORCHVISION_SCHEMA_LIB = torch.library.Library("torchvision", "DEF")
    _TORCHVISION_SCHEMA_LIB.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODAL_APHASIA_ROOT = Path("/workspace/home/AAAI 2027/modal-aphasia")
if MODAL_APHASIA_ROOT.exists() and str(MODAL_APHASIA_ROOT) not in sys.path:
    sys.path.insert(0, str(MODAL_APHASIA_ROOT))

from bench_emu35_modal_memory import (  # noqa: E402
    PeftModel,
    generate_image_batch,
    grade_image,
    load_classifier_map,
)
from eval_emu35_synthetic_lora import load_model  # noqa: E402
from train_emu35_synthetic_lora import build_prompt, build_tokenizer, synthetic_prompt  # noqa: E402
from src.utils.generation_utils import multimodal_decode  # noqa: E402
from src.utils.synthetic_entropy import semantic_entropy_from_distribution  # noqa: E402
from src.vision_tokenizer import build_vision_tokenizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--vq-path", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-concepts", type=int, default=4)
    parser.add_argument("--image-samples-per-concept", type=int, default=8)
    parser.add_argument("--image-batch-size", type=int, default=32)
    parser.add_argument("--prompt-template", default="words_only", choices=("words_only", "typed", "sentence"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--vq-type", default="ibq")
    parser.add_argument("--seed", type=int, default=20270602)
    parser.add_argument("--max-new-tokens-image", type=int, default=620)
    parser.add_argument("--image-top-k", type=int, default=128)
    parser.add_argument("--image-temperature", type=float, default=0.8)
    parser.add_argument("--classifier-free-guidance", type=float, default=1.0)
    parser.set_defaults(record_entropy=False)
    return parser.parse_args()


def attr_options() -> dict[str, list[str]]:
    from modal_aphasia.data import constants

    return {attr: list(constants.CONCEPT_TO_SYNTHETIC_MAP[attr].keys()) for attr in ("color", "pattern", "position", "shape")}


def build_text_question(full_fake_prompt: str, attr: str, options: list[str]) -> str:
    letters = [chr(ord("A") + i) for i in range(len(options))]
    choices = "\n".join(f"{letter}: {option}" for letter, option in zip(letters, options))
    return (
        f"Given the synthetic concept prompt:\n{full_fake_prompt}\n"
        f"Which {attr} does it represent?\n"
        f"{choices}\n"
        "Output a single letter, corresponding to the correct answer, nothing else."
    )


@torch.no_grad()
def text_choice_distribution(model, tokenizer, full_fake_prompt: str, attr: str, options: list[str]) -> dict[str, float]:
    question = build_text_question(full_fake_prompt, attr, options)
    prompt = build_prompt(question, "text")
    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    logits = model(input_ids=input_ids, use_cache=False).logits[0, -1].float()
    letters = [chr(ord("A") + i) for i in range(len(options))]
    letter_ids = [tokenizer.encode(letter, add_special_tokens=False)[0] for letter in letters]
    choice_logits = logits[letter_ids]
    probs = torch.softmax(choice_logits, dim=-1).detach().cpu().numpy()
    return {option: float(prob) for option, prob in zip(options, probs)}


def empirical_image_distribution(values: list[str | None], options: list[str]) -> dict[str, float]:
    total = len(values)
    counts = Counter(value for value in values if value in options)
    return {option: counts.get(option, 0) / total if total else 0.0 for option in options}


def save_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    image_dir = out_dir / "generated_images"
    out_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(exist_ok=True)

    tokenizer = build_tokenizer(args.tokenizer_path)
    vq_model = build_vision_tokenizer(args.vq_type, args.vq_path, device=args.device)
    vq_model.eval().requires_grad_(False)
    base = load_model(args)
    model = PeftModel.from_pretrained(base, args.adapter_path)
    model.eval()
    classifiers = load_classifier_map()
    options_by_attr = attr_options()
    dataset = load_from_disk(args.dataset)
    split = dataset[args.split]
    rows = [dict(split[i]) for i in range(min(args.num_concepts, len(split)))]

    concept_rows = []
    sample_rows = []
    for concept_idx, row in enumerate(rows):
        fake_prompt = synthetic_prompt(row, args.prompt_template)
        expected = {attr: row[attr] for attr in ("color", "pattern", "position", "shape")}

        text_semantic = {}
        for attr, options in options_by_attr.items():
            dist = text_choice_distribution(model, tokenizer, fake_prompt, attr, options)
            stats = semantic_entropy_from_distribution(dist, options)
            stats["correct"] = stats["predicted_value"] == expected[attr]
            text_semantic[attr] = stats

        prompts = [build_prompt(fake_prompt, "image") for _ in range(args.image_samples_per_concept)]
        raw_outputs = []
        for start in range(0, len(prompts), args.image_batch_size):
            batch_prompts = prompts[start : start + args.image_batch_size]
            raws, _, _ = generate_image_batch(model, tokenizer, batch_prompts, args)
            raw_outputs.extend(raws)

        detected_by_attr = {attr: [] for attr in ("color", "pattern", "position", "shape")}
        for sample_idx, raw in enumerate(raw_outputs):
            decoded = [payload for kind, payload in multimodal_decode(raw, tokenizer, vq_model) if kind == "image"]
            detected = {}
            image_path = None
            if decoded:
                image = decoded[0]
                image_path = image_dir / f"concept_{concept_idx:03d}_sample_{sample_idx:03d}.png"
                save_image(image, image_path)
                detected = grade_image(image, classifiers)
            for attr in detected_by_attr:
                detected_by_attr[attr].append(detected.get(attr))
            sample_rows.append(
                {
                    "concept_idx": concept_idx,
                    "sample_idx": sample_idx,
                    "fake_prompt": fake_prompt,
                    **{f"gt_{attr}": expected[attr] for attr in expected},
                    **{f"detected_{attr}": detected.get(attr) for attr in expected},
                    "image_path": str(image_path) if image_path else None,
                }
            )

        image_semantic = {}
        for attr, options in options_by_attr.items():
            dist = empirical_image_distribution(detected_by_attr[attr], options)
            stats = semantic_entropy_from_distribution(dist, options)
            stats["correct"] = stats["predicted_value"] == expected[attr]
            image_semantic[attr] = stats

        for attr in ("color", "pattern", "position", "shape"):
            concept_rows.append(
                {
                    "concept_idx": concept_idx,
                    "fake_prompt": fake_prompt,
                    "attribute": attr,
                    "gt_value": expected[attr],
                    "text_entropy": text_semantic[attr]["entropy"],
                    "text_normalized_entropy": text_semantic[attr]["normalized_entropy"],
                    "text_effective_candidates": text_semantic[attr]["effective_candidates"],
                    "text_predicted_value": text_semantic[attr]["predicted_value"],
                    "text_correct": text_semantic[attr]["correct"],
                    "text_distribution": json.dumps(text_semantic[attr]["distribution"], ensure_ascii=False),
                    "image_entropy": image_semantic[attr]["entropy"],
                    "image_normalized_entropy": image_semantic[attr]["normalized_entropy"],
                    "image_effective_candidates": image_semantic[attr]["effective_candidates"],
                    "image_predicted_value": image_semantic[attr]["predicted_value"],
                    "image_correct": image_semantic[attr]["correct"],
                    "image_distribution": json.dumps(image_semantic[attr]["distribution"], ensure_ascii=False),
                    "image_samples_per_concept": args.image_samples_per_concept,
                }
            )

    with (out_dir / "semantic_entropy_by_concept_attribute.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(concept_rows[0].keys()))
        writer.writeheader()
        writer.writerows(concept_rows)
    with (out_dir / "image_semantic_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sample_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sample_rows)

    aggregate = {}
    for attr in ("color", "pattern", "position", "shape"):
        selected = [r for r in concept_rows if r["attribute"] == attr]
        aggregate[attr] = {
            "n_concepts": len(selected),
            "text_mean_normalized_entropy": float(np.mean([r["text_normalized_entropy"] for r in selected])),
            "image_mean_normalized_entropy": float(np.mean([r["image_normalized_entropy"] for r in selected])),
            "text_mean_entropy": float(np.mean([r["text_entropy"] for r in selected])),
            "image_mean_entropy": float(np.mean([r["image_entropy"] for r in selected])),
            "text_accuracy": float(np.mean([bool(r["text_correct"]) for r in selected])),
            "image_accuracy": float(np.mean([bool(r["image_correct"]) for r in selected])),
        }
    aggregate["overall"] = {
        "n_rows": len(concept_rows),
        "text_mean_normalized_entropy": float(np.mean([r["text_normalized_entropy"] for r in concept_rows])),
        "image_mean_normalized_entropy": float(np.mean([r["image_normalized_entropy"] for r in concept_rows])),
        "text_accuracy": float(np.mean([bool(r["text_correct"]) for r in concept_rows])),
        "image_accuracy": float(np.mean([bool(r["image_correct"]) for r in concept_rows])),
        "num_concepts": len(rows),
        "image_samples_per_concept": args.image_samples_per_concept,
        "image_top_k": args.image_top_k,
        "image_temperature": args.image_temperature,
    }
    (out_dir / "semantic_entropy_summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
