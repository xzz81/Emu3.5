#!/usr/bin/env python3
# Copyright 2026
# SPDX-License-Identifier: Apache-2.0

"""Probe Emu3.5 MLP-neuron activations for generation vs understanding tasks."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import random
import sys
import time
from typing import List, Tuple

from PIL import Image
import torch
from tqdm import tqdm

try:
    _TORCHVISION_SCHEMA_LIB = torch.library.Library("torchvision", "DEF")
    _TORCHVISION_SCHEMA_LIB.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.activation_probe import (  # noqa: E402
    ActivationAccumulator,
    MLPIntermediateActivationProbe,
    token_type_for_id,
    write_csv_rows,
    write_jsonl_rows,
)
from src.utils.input_utils import build_image  # noqa: E402
from src.utils.logits_processor import BOV  # noqa: E402
from src.utils.model_utils import build_emu3p5  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture Emu3.5 MLP intermediate activations, grouped by task and token type. "
            "Pass configs as LABEL=path, e.g. generation=configs/example_config_t2i.py."
        )
    )
    parser.add_argument("--cfg", action="append", required=True, help="Task config as LABEL=PATH. Repeatable.")
    parser.add_argument(
        "--raw-dir",
        action="append",
        default=[],
        help="Optional replay outputs as LABEL=DIR. When present, append DIR/<sample>.txt to the prompt.",
    )
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--max-prompts", type=int, default=2)
    parser.add_argument("--layers", default="all", help="'all' or comma-separated layer indices, e.g. 0,15,31")
    parser.add_argument("--max-seq-len", type=int, default=None, help="Keep BOS plus the last N-1 tokens if needed.")
    parser.add_argument("--image-area", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=100)
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


def parse_cfg_specs(specs: List[str]) -> List[Tuple[str, Path]]:
    parsed = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--cfg must use LABEL=PATH, got {spec!r}")
        label, path = spec.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"empty task label in --cfg {spec!r}")
        parsed.append((label, Path(path)))
    return parsed


def parse_label_paths(specs: List[str]) -> dict[str, Path]:
    parsed = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"expected LABEL=PATH, got {spec!r}")
        label, path = spec.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"empty label in {spec!r}")
        parsed[label] = Path(path)
    return parsed


def parse_layers(value: str):
    if value.lower() == "all":
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def normalize_prompts(cfg, max_prompts):
    prompts = cfg.prompts
    if isinstance(prompts, dict):
        prompts = list(prompts.items())
    else:
        prompts = [(f"{idx:03d}", prompt) for idx, prompt in enumerate(prompts)]
    if max_prompts is not None:
        prompts = prompts[:max_prompts]
    return prompts


def prepare_prompt(cfg, tokenizer, vq_model, question, model_device):
    reference_image = None
    prompt_text = question
    if not isinstance(question, str):
        prompt_text = question.get("prompt", "")
        reference_image = question.get("reference_image")

    if hasattr(cfg, "build_unc_and_template"):
        _unc_prompt, template = cfg.build_unc_and_template(
            getattr(cfg, "task_type", "unknown"),
            with_image=reference_image is not None,
        )
    else:
        template = cfg.template

    prompt = template.format(question=prompt_text)
    if reference_image is not None:
        if isinstance(reference_image, list):
            image_str = "".join(
                build_image(Image.open(path).convert("RGB"), cfg, tokenizer, vq_model)
                for path in reference_image
            )
        else:
            image_str = build_image(Image.open(reference_image).convert("RGB"), cfg, tokenizer, vq_model)
        prompt = prompt.replace("<|IMAGE|>", image_str)

    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(model_device)
    if input_ids[0, 0] != cfg.special_token_ids["BOS"]:
        bos = torch.tensor([[cfg.special_token_ids["BOS"]]], device=input_ids.device, dtype=input_ids.dtype)
        input_ids = torch.cat([bos, input_ids], dim=1)
    return str(prompt_text), input_ids


def maybe_truncate(input_ids: torch.Tensor, max_seq_len: int | None) -> torch.Tensor:
    if max_seq_len is None or input_ids.shape[1] <= max_seq_len:
        return input_ids
    if max_seq_len < 2:
        return input_ids[:, -max_seq_len:]
    return torch.cat([input_ids[:, :1], input_ids[:, -(max_seq_len - 1) :]], dim=1)


def labels_for_ids(input_ids, task_label, phase, special_token_ids):
    labels = []
    for token_id in input_ids[0].detach().cpu().tolist():
        token_type = token_type_for_id(token_id, special_token_ids, visual_token_start=BOV)
        labels.append(
            (
                f"task={task_label}",
                f"phase={phase}",
                f"token_type={token_type}",
                f"task={task_label}|phase={phase}",
                f"task={task_label}|token_type={token_type}",
                f"phase={phase}|token_type={token_type}",
                f"task={task_label}|phase={phase}|token_type={token_type}",
            )
        )
    return labels


def build_probe_sequence(cfg, tokenizer, prompt_ids, task_label, sample_name, raw_dirs):
    prompt_labels = labels_for_ids(prompt_ids, task_label, "prompt", cfg.special_token_ids)
    raw_dir = raw_dirs.get(task_label)
    if raw_dir is None:
        return prompt_ids, prompt_labels, False, None

    raw_path = raw_dir / f"{sample_name}.txt"
    if not raw_path.exists():
        return prompt_ids, prompt_labels, False, str(raw_path)

    raw_text = raw_path.read_text(encoding="utf-8")
    output_ids = tokenizer.encode(raw_text, return_tensors="pt", add_special_tokens=False).to(prompt_ids.device)
    if output_ids.numel() == 0:
        return prompt_ids, prompt_labels, False, str(raw_path)
    output_labels = labels_for_ids(output_ids, task_label, "output", cfg.special_token_ids)
    return torch.cat([prompt_ids, output_ids], dim=1), prompt_labels + output_labels, True, str(raw_path)


def main() -> None:
    args = parse_args()
    cfg_specs = parse_cfg_specs(args.cfg)
    raw_dirs = parse_label_paths(args.raw_dir)
    first_cfg = load_cfg(str(cfg_specs[0][1]))
    if args.image_area is not None:
        first_cfg.image_area = args.image_area

    model, tokenizer, vq_model = build_emu3p5(
        first_cfg.model_path,
        first_cfg.tokenizer_path,
        first_cfg.vq_path,
        vq_type=first_cfg.vq_type,
        model_device=first_cfg.hf_device,
        vq_device=first_cfg.vq_device,
        **getattr(first_cfg, "diffusion_decoder_kwargs", {}),
    )
    model.eval()

    accumulator = ActivationAccumulator()
    probe = MLPIntermediateActivationProbe(model, accumulator, layer_indices=parse_layers(args.layers))
    probe.install()

    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir or (Path(first_cfg.save_path) / "activation_probe_runs" / run_id))
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_rows = []
    try:
        for task_label, cfg_path in cfg_specs:
            cfg = load_cfg(str(cfg_path))
            if args.image_area is not None:
                cfg.image_area = args.image_area
            cfg.special_token_ids = {k: tokenizer.encode(v)[0] for k, v in cfg.special_tokens.items()}
            random.seed(getattr(cfg, "seed", 0))
            prompts = normalize_prompts(cfg, args.max_prompts)

            for sample_name, question in tqdm(prompts, desc=f"probe {task_label}", total=len(prompts)):
                torch.cuda.empty_cache()
                prompt_text, input_ids = prepare_prompt(cfg, tokenizer, vq_model, question, model.device)
                input_ids, labels, replayed_raw, raw_path = build_probe_sequence(
                    cfg, tokenizer, input_ids, task_label, sample_name, raw_dirs
                )
                original_len = int(input_ids.shape[1])
                input_ids = maybe_truncate(input_ids, args.max_seq_len)
                if len(labels) != original_len:
                    raise RuntimeError(f"label count mismatch for {sample_name}: {len(labels)} vs {original_len}")
                if input_ids.shape[1] != original_len:
                    kept = int(input_ids.shape[1])
                    labels = [labels[0]] + labels[-(kept - 1) :] if kept > 1 else labels[-kept:]
                probe.set_context(labels)
                with torch.no_grad():
                    model(input_ids=input_ids, use_cache=False, return_dict=True)
                probe.clear_context()
                sample_rows.append(
                    {
                        "task": task_label,
                        "sample_id": sample_name,
                        "prompt_chars": len(prompt_text),
                        "original_seq_len": original_len,
                        "analyzed_seq_len": int(input_ids.shape[1]),
                        "truncated": original_len != int(input_ids.shape[1]),
                        "replayed_raw": replayed_raw,
                        "raw_path": raw_path,
                    }
                )
    finally:
        probe.remove()

    group_rows = accumulator.group_rows()
    active_rows = accumulator.top_active_rows(top_k=args.top_k)
    write_csv_rows(sample_rows, out_dir / "samples.csv")
    write_csv_rows(group_rows, out_dir / "group_summary.csv")
    write_csv_rows(active_rows, out_dir / "top_active_neurons.csv")
    write_jsonl_rows(group_rows, out_dir / "group_summary.jsonl")

    task_labels = [label for label, _path in cfg_specs]
    if len(task_labels) >= 2:
        for positive in task_labels:
            for negative in task_labels:
                if positive == negative:
                    continue
                rows = accumulator.top_contrast_rows(f"task={positive}", f"task={negative}", top_k=args.top_k)
                safe_name = f"contrast_{positive}_vs_{negative}_overall.csv".replace("/", "_")
                write_csv_rows(rows, out_dir / safe_name)

                for phase in ("prompt", "output"):
                    rows = accumulator.top_contrast_rows(
                        f"task={positive}|phase={phase}",
                        f"task={negative}|phase={phase}",
                        top_k=args.top_k,
                    )
                    safe_name = f"contrast_{positive}_vs_{negative}_{phase}.csv".replace("/", "_")
                    write_csv_rows(rows, out_dir / safe_name)

                pos_group = f"task={positive}|token_type=text"
                neg_group = f"task={negative}|token_type=text"
                rows = accumulator.top_contrast_rows(pos_group, neg_group, top_k=args.top_k)
                safe_name = f"contrast_{positive}_vs_{negative}_text.csv".replace("/", "_")
                write_csv_rows(rows, out_dir / safe_name)

                pos_visual = f"task={positive}|token_type=visual"
                neg_visual = f"task={negative}|token_type=visual"
                rows = accumulator.top_contrast_rows(pos_visual, neg_visual, top_k=args.top_k)
                safe_name = f"contrast_{positive}_vs_{negative}_visual.csv".replace("/", "_")
                write_csv_rows(rows, out_dir / safe_name)

                for phase in ("prompt", "output"):
                    for token_type in ("text", "visual", "structure"):
                        rows = accumulator.top_contrast_rows(
                            f"task={positive}|phase={phase}|token_type={token_type}",
                            f"task={negative}|phase={phase}|token_type={token_type}",
                            top_k=args.top_k,
                        )
                        safe_name = (
                            f"contrast_{positive}_vs_{negative}_{phase}_{token_type}.csv".replace("/", "_")
                        )
                        write_csv_rows(rows, out_dir / safe_name)

    print(f"[INFO] activation probe saved to {out_dir}")


if __name__ == "__main__":
    main()
