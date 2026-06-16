#!/usr/bin/env python3
# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Run image-reading text generation with token-level entropy traces."""

from __future__ import annotations

import argparse
import importlib as imp
import math
from pathlib import Path
import random
import sys
import time

from PIL import Image
import torch
from tqdm import tqdm
from transformers import GenerationConfig
from transformers.generation import LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList

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

from src.utils.entropy_trace import (  # noqa: E402
    build_trace_records,
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
from src.utils.generation_utils import multimodal_decode  # noqa: E402
from src.utils.input_utils import build_image  # noqa: E402
from src.utils.logits_processor import BOI, BOV, EOI, EOF, EOL, IMG  # noqa: E402
from src.utils.model_utils import build_emu3p5  # noqa: E402


class TextAnswerOnlyLogitsProcessor(LogitsProcessor):
    """Keep generation in text space and apply text top-k/top-p/temperature."""

    def __init__(self, top_k: int = 1024, top_p: float = 0.9, temperature: float = 1.0):
        self.top_k = int(top_k)
        self.top_p = float(top_p)
        self.temperature = float(temperature)
        self.forbidden_ids = {BOI, EOI, IMG, EOL, EOF}

    def __call__(self, input_ids, scores):
        scores = scores.clone()
        scores[:, BOV:] = -math.inf
        for token_id in self.forbidden_ids:
            if 0 <= int(token_id) < scores.shape[-1]:
                scores[:, int(token_id)] = -math.inf
        if self.temperature != 1.0:
            scores = scores / self.temperature
        if 0 < self.top_k < scores.shape[-1]:
            threshold = torch.topk(scores, self.top_k, dim=-1).values[:, -1, None]
            scores = scores.masked_fill(scores < threshold, -math.inf)
        if self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(scores, descending=True, dim=-1)
            cumulative = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_remove = cumulative > self.top_p
            sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
            sorted_remove[..., 0] = False
            remove = torch.zeros_like(sorted_remove).scatter(1, sorted_indices, sorted_remove)
            scores = scores.masked_fill(remove, -math.inf)
        return scores


class StopOnGeneratedTokenCriteria(StoppingCriteria):
    """Stop once a configured token appears in the generated answer."""

    def __init__(self, prompt_len: int, stop_token_ids: list[int]):
        self.prompt_len = int(prompt_len)
        self.stop_token_ids = {int(token_id) for token_id in stop_token_ids}

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        generated = input_ids[0, self.prompt_len :]
        if generated.numel() == 0:
            return False
        return int(generated[-1].item()) in self.stop_token_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--image-area", type=int, default=None)
    parser.add_argument("--prompt-name-contains", default=None)
    parser.add_argument("--skip-existing", action="store_true")
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
    return prompts


def prepare_prompt(cfg, tokenizer, vq_model, question, model_device):
    if isinstance(question["reference_image"], list):
        image_str = "".join(
            build_image(Image.open(img).convert("RGB"), cfg, tokenizer, vq_model)
            for img in question["reference_image"]
        )
    else:
        image_str = build_image(Image.open(question["reference_image"]).convert("RGB"), cfg, tokenizer, vq_model)
    prompt_text = question.get("prompt", "")
    prompt = cfg.template.format(question=prompt_text).replace("<|IMAGE|>", image_str)
    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(model_device)
    if input_ids[0, 0] != cfg.special_token_ids["BOS"]:
        bos = torch.tensor([[cfg.special_token_ids["BOS"]]], device=input_ids.device, dtype=input_ids.dtype)
        input_ids = torch.cat([bos, input_ids], dim=1)
    return prompt_text, input_ids


@torch.no_grad()
def generate_text_trace(cfg, model, tokenizer, input_ids, sample_id: str):
    input_len = input_ids.shape[1]
    text_processor = TextAnswerOnlyLogitsProcessor(
        top_k=cfg.sampling_params.get("text_top_k", 1024),
        top_p=cfg.sampling_params.get("text_top_p", 0.9),
        temperature=cfg.sampling_params.get("text_temperature", 1.0),
    )
    generation_config = GenerationConfig(
        max_new_tokens=cfg.max_new_tokens,
        do_sample=bool(cfg.sampling_params.get("do_sample", True)),
        num_beams=1,
        use_cache=True,
        pad_token_id=cfg.special_token_ids["PAD"],
        eos_token_id=cfg.special_token_ids["EOS"],
    )
    stop_token_ids = [
        cfg.special_token_ids[name]
        for name in getattr(cfg, "text_stop_after_token_names", [])
        if name in cfg.special_token_ids
    ]
    stopping_criteria = None
    if stop_token_ids:
        stopping_criteria = StoppingCriteriaList([StopOnGeneratedTokenCriteria(input_len, stop_token_ids)])
    outputs = model.generate(
        input_ids,
        generation_config=generation_config,
        logits_processor=LogitsProcessorList([text_processor]),
        stopping_criteria=stopping_criteria,
        return_dict_in_generate=True,
        output_scores=True,
    )
    token_ids = outputs.sequences[:, input_len:]
    generated = token_ids[0].detach().cpu()
    records = build_trace_records(
        outputs.scores,
        generated,
        sample_id=sample_id,
        tokenizer=tokenizer,
        special_token_ids=cfg.special_token_ids,
        metadata={
            "task": getattr(cfg, "task_type", "image_read_entropy"),
            "text_stop_after_token_names": ",".join(getattr(cfg, "text_stop_after_token_names", [])),
        },
    )
    return generated.tolist(), records


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.cfg)
    if args.max_new_tokens is not None:
        cfg.max_new_tokens = args.max_new_tokens
        cfg.sampling_params["max_new_tokens"] = args.max_new_tokens
    if args.image_area is not None:
        cfg.image_area = args.image_area
    cfg.prompts = normalize_prompts(cfg, args.max_prompts)
    if args.prompt_name_contains:
        cfg.prompts = [(n, p) for n, p in cfg.prompts if args.prompt_name_contains in n]

    model, tokenizer, vq_model = build_emu3p5(
        cfg.model_path,
        cfg.tokenizer_path,
        cfg.vq_path,
        vq_type=cfg.vq_type,
        model_device=cfg.hf_device,
        vq_device=cfg.vq_device,
    )
    cfg.special_token_ids = {k: tokenizer.encode(v)[0] for k, v in cfg.special_tokens.items()}
    random.seed(cfg.seed)

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
        if args.skip_existing and (raw_dir / f"{name}.txt").exists() and (trace_dir / f"{name}_entropy.jsonl").exists():
            continue
        torch.cuda.empty_cache()
        prompt_text, input_ids = prepare_prompt(cfg, tokenizer, vq_model, question, model.device)
        result_tokens, records = generate_text_trace(cfg, model, tokenizer, input_ids, name)
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
    report = render_summary_report(all_records, f"Image Read Entropy Run {run_id}")
    if figure_paths:
        report += "\n\n## Generated figures\n\n" + "\n".join(
            f"- `{Path(path).relative_to(out_root)}`" for path in figure_paths
        )
    (out_root / "entropy_distribution_report.md").write_text(report + "\n", encoding="utf-8")
    write_html_report(all_records, out_root, f"Image Read Entropy Run {run_id}", figure_paths)
    print(f"[INFO] image-read entropy run saved to {out_root}")


if __name__ == "__main__":
    main()
