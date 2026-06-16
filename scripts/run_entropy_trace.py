#!/usr/bin/env python3
# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Run Emu3.5 inference with token-level UME traces."""

from __future__ import annotations

import argparse
import importlib as imp
import os
import os.path as osp
from pathlib import Path
import random
import sys
import time

from PIL import Image
import torch
from tqdm import tqdm

try:
    import torchvision  # noqa: F401
except RuntimeError as exc:
    if "operator torchvision::nms does not exist" in str(exc):
        try:
            _TORCHVISION_SCHEMA_LIB = torch.library.Library("torchvision", "DEF")
            _TORCHVISION_SCHEMA_LIB.define(
                "nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor"
            )
        except Exception:
            pass
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.entropy_trace import (  # noqa: E402
    calibration_bin_rows,
    component_correlation_rows,
    hallucination_diagnostic_rows,
    render_summary_report,
    summarize_records,
    thinking_transition_rows,
    write_csv,
    write_html_report,
    write_jsonl,
    write_visual_artifacts,
)
from src.utils.generation_utils import generate_with_entropy_trace, multimodal_decode  # noqa: E402
from src.utils.input_utils import build_image  # noqa: E402
from src.utils.model_utils import build_emu3p5  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--image-area", type=int, default=None)
    parser.add_argument("--num-workers", default=1, type=int)
    parser.add_argument("--worker-id", default=0, type=int)
    return parser.parse_args()


def load_cfg(path: str):
    cfg_name = Path(path).stem
    cfg_package = Path(path).parent.__str__().replace("/", ".")
    return imp.import_module(f".{cfg_name}", package=cfg_package)


def normalize_prompts(cfg, max_prompts):
    prompts = cfg.prompts
    if isinstance(prompts, dict):
        prompts = [(n, p) for n, p in prompts.items()]
    else:
        prompts = [(f"{idx:03d}", p) for idx, p in enumerate(prompts)]
    if max_prompts is not None:
        prompts = prompts[:max_prompts]
    return prompts[cfg.rank :: cfg.world_size]


def prepare_prompt(cfg, tokenizer, vq_model, question, model_device):
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

    unconditional_ids = tokenizer.encode(unc_prompt, return_tensors="pt", add_special_tokens=False).to(input_ids.device)
    full_unc_ids = None
    if hasattr(cfg, "img_unc_prompt"):
        full_unc_ids = tokenizer.encode(cfg.img_unc_prompt, return_tensors="pt", add_special_tokens=False).to(input_ids.device)
    return question, input_ids, unconditional_ids, full_unc_ids


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.cfg)
    if args.max_new_tokens is not None:
        cfg.max_new_tokens = args.max_new_tokens
        cfg.sampling_params["max_new_tokens"] = args.max_new_tokens
    if args.image_area is not None:
        cfg.image_area = args.image_area
    cfg.rank = args.worker_id
    cfg.world_size = args.num_workers
    cfg.prompts = normalize_prompts(cfg, args.max_prompts)

    model, tokenizer, vq_model = build_emu3p5(
        cfg.model_path,
        cfg.tokenizer_path,
        cfg.vq_path,
        vq_type=cfg.vq_type,
        model_device=cfg.hf_device,
        vq_device=cfg.vq_device,
        **getattr(cfg, "diffusion_decoder_kwargs", {}),
    )
    cfg.special_token_ids = {k: tokenizer.encode(v)[0] for k, v in cfg.special_tokens.items()}
    random.seed(cfg.seed + cfg.rank)

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    out_root = Path(cfg.save_path) / "ume_trace_runs" / run_id
    raw_dir = out_root / "raw_generations"
    trace_dir = out_root / "entropy_traces"
    case_dir = out_root / "case_studies"
    decoded_dir = out_root / "decoded"
    for path in (raw_dir, trace_dir, case_dir, decoded_dir, out_root / "tables", out_root / "figures"):
        path.mkdir(parents=True, exist_ok=True)

    all_records = []
    for name, question in tqdm(cfg.prompts, total=len(cfg.prompts)):
        torch.cuda.empty_cache()
        prompt_text, input_ids, unconditional_ids, full_unc_ids = prepare_prompt(
            cfg, tokenizer, vq_model, question, model.device
        )
        force_same_image_size = not (not isinstance(question, str) and isinstance(question.get("reference_image"), list))

        result_tokens, records = generate_with_entropy_trace(
            cfg,
            model,
            tokenizer,
            input_ids,
            unconditional_ids,
            full_unc_ids,
            force_same_image_size=force_same_image_size,
            sample_id=name,
        )
        result = tokenizer.decode(result_tokens, skip_special_tokens=False)
        write_jsonl(records, trace_dir / f"{name}_entropy.jsonl")
        (raw_dir / f"{name}.txt").write_text(result, encoding="utf-8")
        (case_dir / f"{name}_summary.md").write_text(
            render_summary_report(records, f"Sample {name}: {prompt_text[:120]}"),
            encoding="utf-8",
        )
        try:
            decoded_counts = {}
            for kind, payload in multimodal_decode(result, tokenizer, vq_model):
                idx = decoded_counts.get(kind, 0)
                decoded_counts[kind] = idx + 1
                if kind == "image":
                    payload.save(decoded_dir / f"{name}_image_{idx:02d}.png")
                else:
                    (decoded_dir / f"{name}_{kind}_{idx:02d}.txt").write_text(str(payload), encoding="utf-8")
        except Exception as exc:
            print(f"[WARNING] multimodal_decode failed for {name}: {exc}", flush=True)
        all_records.extend(records)

    write_csv(summarize_records(all_records, "token_type"), out_root / "tables" / "ume_by_token_type.csv")
    write_csv(summarize_records(all_records, "segment"), out_root / "tables" / "ume_by_segment.csv")
    write_csv(summarize_records(all_records, "task"), out_root / "tables" / "ume_by_task.csv")
    write_csv(component_correlation_rows(all_records), out_root / "tables" / "ume_component_correlations.csv")
    write_csv(thinking_transition_rows(all_records), out_root / "tables" / "thinking_transition.csv")
    write_csv(hallucination_diagnostic_rows(all_records), out_root / "tables" / "hallucination_diagnostics.csv")
    write_csv(calibration_bin_rows(all_records), out_root / "tables" / "calibration_bins.csv")
    figure_paths = write_visual_artifacts(all_records, out_root)
    report = render_summary_report(all_records, f"UME Trace Run {run_id}")
    if figure_paths:
        report += "\n\n## Generated figures\n\n" + "\n".join(
            f"- `{Path(path).relative_to(out_root)}`" for path in figure_paths
        )
    (out_root / "entropy_distribution_report.md").write_text(report + "\n", encoding="utf-8")
    write_html_report(all_records, out_root, f"UME Trace Run {run_id}", figure_paths)
    print(f"[INFO] UME trace run saved to {out_root}")


if __name__ == "__main__":
    main()
