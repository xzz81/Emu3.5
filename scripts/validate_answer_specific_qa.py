#!/usr/bin/env python3
"""Validate fixed answer-specific QA targets by asking the model to read images."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

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

from transformers import GenerationConfig
from transformers.generation import LogitsProcessorList

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_answer_specific_mask_delta import (  # noqa: E402
    DEFAULT_QA,
    TextOnlyLogitsProcessor,
    build_qa_prompt,
    encode_answer,
    load_qa_spec,
    sample_id_from_image_path,
    score_answer,
)
from src.utils.model_utils import build_emu3p5  # noqa: E402


STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "is",
    "of",
    "on",
    "the",
    "with",
}
NUMBER_ALIASES = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/emu3p5-main")
    parser.add_argument("--run-id-contains", default="topk64")
    parser.add_argument(
        "--keep-batches",
        default="batch2,expand4",
        help="Comma-separated batch labels to keep; empty keeps all.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--qa-spec", default=None)
    parser.add_argument("--replace-default-qa", action="store_true")
    parser.add_argument("--model-path", default="model/Emu3.5")
    parser.add_argument("--vq-path", default="model/Emu3.5-VisionTokenizer")
    parser.add_argument("--tokenizer-path", default="./src/tokenizer_emu3_ibq")
    parser.add_argument("--vq-type", default="ibq")
    parser.add_argument("--model-device", default="auto")
    parser.add_argument("--vq-device", default="cuda:0")
    parser.add_argument("--image-area", type=int, default=512 * 512)
    parser.add_argument("--max-new-tokens", type=int, default=12)
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


def normalize(text: str):
    text = re.sub(r"<\|[^>]+?\|>", " ", text)
    text = text.lower()
    text = text.replace("24 h", "24h")
    words = re.findall(r"[a-z0-9]+", text)
    return [NUMBER_ALIASES.get(word, word) for word in words]


def significant_words(text: str):
    return [word for word in normalize(text) if word not in STOP_WORDS]


def match_stats(target: str, generated: str):
    target_words = significant_words(target)
    generated_words = set(significant_words(generated))
    if not target_words:
        return 0.0, True, ""
    hits = [word for word in target_words if word in generated_words]
    fraction = len(hits) / len(target_words)
    supported = fraction >= 0.67
    return fraction, supported, " ".join(hits)


def find_run_dirs(root: Path, run_id_contains: str):
    for run_dir in sorted(root.glob("*/ume_trace_runs/*")):
        if run_id_contains and run_id_contains not in run_dir.name:
            continue
        decoded = run_dir / "decoded"
        if not decoded.exists():
            continue
        yield run_dir.parent.parent.name, run_dir


@torch.no_grad()
def generate_short_answer(model, tokenizer, prompt_ids, max_new_tokens: int):
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
    answer_ids = generated[:, input_len:]
    return tokenizer.decode(answer_ids[0], skip_special_tokens=False), answer_ids


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    model, tokenizer, vq_model = build_emu3p5(
        args.model_path,
        args.tokenizer_path,
        args.vq_path,
        vq_type=args.vq_type,
        model_device=args.model_device,
        vq_device=args.vq_device,
    )

    rows = []
    qa_spec = load_qa_spec(args.qa_spec, args.replace_default_qa)
    keep_batches = {item.strip() for item in args.keep_batches.split(",") if item.strip()}
    for task, run_dir in find_run_dirs(Path(args.root), args.run_id_contains):
        batch = "expand4" if "expand4" in run_dir.name else "batch2" if "batch2" in run_dir.name else "other"
        if keep_batches and batch not in keep_batches:
            continue
        for image_path in sorted((run_dir / "decoded").glob("*_image_00.png")):
            sample_id = sample_id_from_image_path(image_path)
            if sample_id not in qa_spec:
                continue
            from PIL import Image

            image = Image.open(image_path).convert("RGB")
            spec = qa_spec[sample_id]
            prompt_ids = build_qa_prompt(tokenizer, vq_model, image, args.image_area, spec["question"])
            generated_text, generated_ids = generate_short_answer(
                model, tokenizer, prompt_ids, args.max_new_tokens
            )
            target_ids = encode_answer(tokenizer, spec["answer"])
            target_nll, target_entropy, target_tokens = score_answer(model, prompt_ids, target_ids)
            generated_nll, _, generated_tokens = score_answer(model, prompt_ids, generated_ids)
            fraction, supported, matched_words = match_stats(spec["answer"], generated_text)
            rows.append(
                {
                    "task": task,
                    "batch": batch,
                    "run_id": run_dir.name,
                    "sample_id": sample_id,
                    "question": spec["question"],
                    "target_answer": spec["answer"],
                    "qa_source": spec.get("source", "default"),
                    "generated_answer": generated_text.strip(),
                    "target_match_fraction": fraction,
                    "qa_supported": supported,
                    "matched_target_words": matched_words,
                    "target_nll": target_nll,
                    "target_entropy": target_entropy,
                    "target_tokens": target_tokens,
                    "generated_nll": generated_nll,
                    "generated_tokens": int(generated_tokens),
                    "target_minus_generated_nll": target_nll - generated_nll,
                    "image_path": str(image_path),
                }
            )
            print(
                f"[INFO] {sample_id}: target={spec['answer']!r} generated={generated_text.strip()!r} "
                f"match={fraction:.3f} supported={supported}",
                flush=True,
            )

    write_csv(rows, out_dir / "qa_validation.csv")
    print(f"[INFO] wrote {len(rows)} QA validation rows to {out_dir}")


if __name__ == "__main__":
    main()
