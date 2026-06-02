#!/usr/bin/env python3
# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""LoRA finetuning for modal-aphasia synthetic concepts on Emu3.5."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

try:
    _TORCHVISION_SCHEMA_LIB = torch.library.Library("torchvision", "DEF")
    _TORCHVISION_SCHEMA_LIB.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
except Exception:
    pass

from transformers import AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, set_seed

try:
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
except Exception as exc:  # pragma: no cover - dependency guard for friendlier CLI errors
    raise SystemExit(
        "Missing dependency 'peft'. Install finetune dependencies first, e.g. `pip install datasets peft`."
    ) from exc

try:
    import datasets
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency 'datasets'. Install finetune dependencies first, e.g. `pip install datasets peft`."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.emu3p5 import Emu3Config, Emu3ForCausalLM  # noqa: E402
from src.utils.input_utils import format_image_string, smart_resize  # noqa: E402
from src.vision_tokenizer import build_vision_tokenizer  # noqa: E402


SPECIAL_TOKENS = {
    "bos_token": "<|extra_203|>",
    "eos_token": "<|extra_204|>",
    "pad_token": "<|endoftext|>",
    "eol_token": "<|extra_200|>",
    "eof_token": "<|extra_201|>",
    "tms_token": "<|extra_202|>",
    "img_token": "<|image token|>",
    "boi_token": "<|image start|>",
    "eoi_token": "<|image end|>",
    "bss_token": "<|extra_100|>",
    "ess_token": "<|extra_101|>",
    "bog_token": "<|extra_60|>",
    "eog_token": "<|extra_61|>",
    "boc_token": "<|extra_50|>",
    "eoc_token": "<|extra_51|>",
}

LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--vq-path", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-template", choices=("words_only", "typed", "sentence"), default="words_only")
    parser.add_argument("--training-mode", choices=("image_only", "text_only", "joint"), default="image_only")
    parser.add_argument("--split", default="train")
    parser.add_argument("--image-area", type=int, default=384 * 384)
    parser.add_argument("--vq-type", default="ibq")
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-epochs", type=float, default=5.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--bnb-4bit-quant-type", default="nf4")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--fsdp", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--fsdp-transformer-layer-cls-to-wrap", default="Emu3DecoderLayer")
    parser.add_argument(
        "--fsdp-state-dict-type",
        choices=("FULL_STATE_DICT", "SHARDED_STATE_DICT", "LOCAL_STATE_DICT"),
        default="SHARDED_STATE_DICT",
    )
    return parser.parse_args()


def build_tokenizer(tokenizer_path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        special_tokens_file=str(Path(tokenizer_path) / "emu3_vision_tokens.txt"),
        trust_remote_code=True,
    )
    for attr, value in SPECIAL_TOKENS.items():
        setattr(tokenizer, attr, value)
    tokenizer.padding_side = "right"
    return tokenizer


def local_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return f"cuda:{local_rank}"


@torch.no_grad()
def encode_image_to_string(image: Image.Image, image_area: int, tokenizer: Any, vq_model: torch.nn.Module) -> str:
    image = smart_resize(image.convert("RGB"), image_area)
    width, height = image.size
    device = next(vq_model.parameters()).device
    dtype = next(vq_model.parameters()).dtype
    image_tensor = torch.tensor(np.array(image) / 127.5 - 1.0, device=device, dtype=dtype)
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
    _, _, token = vq_model.encode(image_tensor)
    token = token[-1].view(height // 16, width // 16)
    return format_image_string(tokenizer, token)


def synthetic_prompt(row: dict[str, Any], template: str) -> str:
    words = {
        "color": row["synthetic_color"],
        "pattern": row["synthetic_pattern"],
        "position": row["synthetic_position"],
        "shape": row["synthetic_shape"],
    }
    if template == "words_only":
        return "{color} {pattern} {position} {shape}".format(**words)
    if template == "typed":
        return "color={color}, pattern={pattern}, position={position}, shape={shape}".format(**words)
    if template == "sentence":
        return "A {shape} on {color} {pattern} in {position}".format(**words)
    raise ValueError(f"Unsupported prompt template: {template}")


def system_prompt(mode: str) -> str:
    if mode == "text":
        return "You are a helpful assistant for concept definition task."
    return "You are a helpful assistant for t2i task."


def build_prompt(question: str, mode: str) -> str:
    return f"{SPECIAL_TOKENS['bos_token']}{system_prompt(mode)} USER: {question} ASSISTANT: {SPECIAL_TOKENS['bss_token']}"


def text_definition(row: dict[str, Any]) -> str:
    pairs = (
        (row["synthetic_color"], row["color"]),
        (row["synthetic_pattern"], row["pattern"]),
        (row["synthetic_position"], row["position"]),
        (row["synthetic_shape"], row["shape"]),
    )
    return "; ".join(f"{fake} = {real}" for fake, real in pairs)


def encode_labeled_example(tokenizer: Any, prompt: str, target: str) -> dict[str, list[int]]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    return {
        "input_ids": prompt_ids + target_ids,
        "labels": [-100] * len(prompt_ids) + target_ids,
        "prompt_len": len(prompt_ids),
        "target_len": len(target_ids),
    }


class EncodedListDataset(Dataset):
    def __init__(self, records: list[dict[str, list[int]]]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        item = self.records[idx]
        return {"input_ids": item["input_ids"], "labels": item["labels"]}


@dataclass
class CausalCollator:
    pad_token_id: int

    def __call__(self, batch: Iterable[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        batch = list(batch)
        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids, labels, attention_mask = [], [], []
        for item in batch:
            pad_len = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * pad_len)
            labels.append(item["labels"] + [-100] * pad_len)
            attention_mask.append([1] * len(item["input_ids"]) + [0] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def load_split(dataset_path: str, split: str):
    dataset = datasets.load_from_disk(dataset_path)
    if isinstance(dataset, datasets.DatasetDict):
        if split not in dataset:
            raise ValueError(f"Split {split!r} not found. Available splits: {list(dataset.keys())}")
        return dataset[split]
    return dataset


def prepare_records(args: argparse.Namespace, tokenizer: Any, vq_model: torch.nn.Module) -> list[dict[str, Any]]:
    split = load_split(args.dataset, args.split)
    required = {
        "image",
        "synthetic_color",
        "synthetic_pattern",
        "synthetic_position",
        "synthetic_shape",
        "color",
        "pattern",
        "position",
        "shape",
    }
    missing = required.difference(split.column_names)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if args.max_train_samples is not None:
        split = split.select(range(min(args.max_train_samples, len(split))))

    records: list[dict[str, Any]] = []
    for idx, row in enumerate(tqdm(split, desc="Encoding synthetic records")):
        question = synthetic_prompt(row, args.prompt_template)
        modes = ("image",) if args.training_mode == "image_only" else ("text",)
        if args.training_mode == "joint":
            modes = ("image", "text")

        image_target = None
        for mode in modes:
            prompt = build_prompt(question, mode)
            if mode == "image":
                if image_target is None:
                    image_target = encode_image_to_string(row["image"], args.image_area, tokenizer, vq_model)
                target = image_target + SPECIAL_TOKENS["ess_token"] + SPECIAL_TOKENS["eos_token"]
            else:
                target = text_definition(row) + SPECIAL_TOKENS["ess_token"] + SPECIAL_TOKENS["eos_token"]

            encoded = encode_labeled_example(tokenizer, prompt, target)
            encoded.update({"sample_idx": idx, "mode": mode, "question": question, "target_preview": target[:160]})
            records.append(encoded)

    return records


def assert_first_example(records: list[dict[str, Any]], tokenizer: Any, expect_image: bool) -> None:
    if not records:
        raise ValueError("No training records were produced.")
    first = records[0]
    prompt_len = first["prompt_len"]
    labels = first["labels"]
    if labels[:prompt_len] != [-100] * prompt_len:
        raise AssertionError("Prompt labels are not fully masked with -100.")
    if not any(label != -100 for label in labels[prompt_len:]):
        raise AssertionError("Target labels contain no supervised tokens.")
    decoded_target = tokenizer.decode(first["input_ids"][prompt_len:], skip_special_tokens=False)
    if expect_image and not decoded_target.startswith("<|image start|>24*24<|image token|>"):
        raise AssertionError(f"Unexpected image target prefix: {decoded_target[:80]!r}")
    print(
        json.dumps(
            {
                "num_records": len(records),
                "first_mode": first["mode"],
                "first_question": first["question"],
                "first_prompt_len": prompt_len,
                "first_target_len": first["target_len"],
                "first_target_prefix": decoded_target[:120],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


def build_model(args: argparse.Namespace) -> Emu3ForCausalLM:
    attn_implementation = args.attn_implementation
    if attn_implementation is None:
        try:
            import flash_attn  # noqa: F401

            attn_implementation = "flash_attention_2"
        except Exception:
            attn_implementation = "eager"

    config = Emu3Config.from_pretrained(args.model_path, trust_remote_code=True)
    quantization_config = None
    device_map = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        device_map = {"": local_device()}
    model = Emu3ForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32),
        attn_implementation=attn_implementation,
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
        device_map=device_map,
    )
    model.config.use_cache = False
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )
    if args.gradient_checkpointing and not should_use_fsdp(args):
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=list(LORA_TARGET_MODULES),
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def should_use_fsdp(args: argparse.Namespace) -> bool:
    if args.load_in_4bit:
        return False
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return args.fsdp == "on" or (args.fsdp == "auto" and world_size > 1)


def fsdp_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    if not should_use_fsdp(args):
        return {}
    return {
        "fsdp": "full_shard auto_wrap",
        "fsdp_config": {
            "transformer_layer_cls_to_wrap": [args.fsdp_transformer_layer_cls_to_wrap],
            "use_orig_params": True,
            "activation_checkpointing": bool(args.gradient_checkpointing),
            "state_dict_type": args.fsdp_state_dict_type,
        },
    }


def save_metadata(args: argparse.Namespace, records: list[dict[str, Any]], tokenizer: Any) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_path": args.model_path,
        "vq_path": args.vq_path,
        "tokenizer_path": args.tokenizer_path,
        "dataset": args.dataset,
        "split": args.split,
        "prompt_template": args.prompt_template,
        "training_mode": args.training_mode,
        "image_area": args.image_area,
        "num_records": len(records),
        "lora_target_modules": list(LORA_TARGET_MODULES),
        "special_tokens": SPECIAL_TOKENS,
    }
    (output_dir / "synthetic_lora_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tokenizer.save_pretrained(output_dir / "tokenizer")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    tokenizer = build_tokenizer(args.tokenizer_path)
    vq_model = build_vision_tokenizer(args.vq_type, args.vq_path, device=local_device())
    vq_model.eval()
    vq_model.requires_grad_(False)

    records = prepare_records(args, tokenizer, vq_model)
    assert_first_example(records, tokenizer, expect_image=args.training_mode in {"image_only", "joint"})
    save_metadata(args, records, tokenizer)
    if args.dry_run:
        print("[INFO] Dry run finished before model load/training.", flush=True)
        return

    del vq_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = build_model(args)
    train_dataset = EncodedListDataset(records)
    collator = CausalCollator(pad_token_id=tokenizer.pad_token_id)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=args.bf16,
        fp16=args.fp16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_strategy="steps",
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        gradient_checkpointing=args.gradient_checkpointing and not should_use_fsdp(args),
        ddp_find_unused_parameters=False,
        **fsdp_kwargs(args),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(Path(args.output_dir) / "tokenizer")
    print(f"[INFO] LoRA adapter saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
