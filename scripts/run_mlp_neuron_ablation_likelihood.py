#!/usr/bin/env python3
# Copyright 2026
# SPDX-License-Identifier: Apache-2.0

"""Measure causal effect of ablating selected Emu3.5 MLP intermediate neurons.

This is a Stage 1 minimal causal-intervention script for replayed outputs. It
scores raw generated output tokens with and without selected MLP-neuron
contributions.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Tuple

from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    _TORCHVISION_SCHEMA_LIB = torch.library.Library("torchvision", "DEF")
    _TORCHVISION_SCHEMA_LIB.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.activation_probe import token_type_for_id, write_csv_rows  # noqa: E402
from src.utils.input_utils import build_image  # noqa: E402
from src.utils.logits_processor import BOV  # noqa: E402
from src.utils.model_utils import build_emu3p5  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--candidate", action="append", default=[], help="Layer/neuron as LAYER:NEURON.")
    parser.add_argument("--candidate-csv", default=None, help="CSV with layer and neuron columns.")
    parser.add_argument("--top-n", type=int, default=0, help="Use top N rows from --candidate-csv.")
    parser.add_argument("--max-prompts", type=int, default=4)
    parser.add_argument("--target-token-type", choices=["all", "text", "visual", "structure"], default="all")
    parser.add_argument("--ablate-phase", choices=["all", "prompt", "output"], default="all")
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


def parse_candidates(items: Iterable[str]) -> List[Tuple[int, int]]:
    candidates = []
    for item in items:
        layer, neuron = item.split(":", 1)
        candidates.append((int(layer), int(neuron)))
    return candidates


def read_candidate_csv(path: str, top_n: int) -> List[Tuple[int, int]]:
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append((int(row["layer"]), int(row["neuron"])))
            if top_n and len(rows) >= top_n:
                break
    return rows


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


def build_replay_ids(cfg, tokenizer, vq_model, question, raw_path, model_device):
    prompt_text, prompt_ids = prepare_prompt(cfg, tokenizer, vq_model, question, model_device)
    raw_text = Path(raw_path).read_text(encoding="utf-8")
    output_ids = tokenizer.encode(raw_text, return_tensors="pt", add_special_tokens=False).to(model_device)
    full_ids = torch.cat([prompt_ids, output_ids], dim=1)
    phases = ["prompt"] * prompt_ids.shape[1] + ["output"] * output_ids.shape[1]
    token_types = [
        token_type_for_id(token_id, cfg.special_token_ids, visual_token_start=BOV)
        for token_id in full_ids[0].detach().cpu().tolist()
    ]
    return prompt_text, full_ids, phases, token_types


class MLPNeuronAblator:
    def __init__(self, model, candidates: List[Tuple[int, int]], position_mask: torch.Tensor | None = None):
        self.model = model
        self.by_layer: Dict[int, List[int]] = {}
        for layer, neuron in candidates:
            self.by_layer.setdefault(int(layer), []).append(int(neuron))
        self.position_mask = position_mask
        self.handles = []
        self._gate = {}
        self._intermediate = {}

    def install(self):
        for layer_idx, neurons in self.by_layer.items():
            mlp = self.model.model.layers[layer_idx].mlp
            self.handles.append(mlp.gate_proj.register_forward_hook(self._gate_hook(layer_idx, mlp.act_fn)))
            self.handles.append(mlp.up_proj.register_forward_hook(self._up_hook(layer_idx)))
            self.handles.append(mlp.register_forward_hook(self._mlp_hook(layer_idx, neurons)))

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self._gate.clear()
        self._intermediate.clear()

    def _gate_hook(self, layer_idx, act_fn):
        def hook(_module, _inputs, output):
            self._gate[layer_idx] = act_fn(output.detach())

        return hook

    def _up_hook(self, layer_idx):
        def hook(_module, _inputs, output):
            gate = self._gate.pop(layer_idx, None)
            if gate is not None:
                self._intermediate[layer_idx] = gate * output.detach()

        return hook

    def _mlp_hook(self, layer_idx, neurons):
        def hook(module, _inputs, output):
            intermediate = self._intermediate.pop(layer_idx, None)
            if intermediate is None:
                return output
            neuron_ids = torch.tensor(neurons, device=output.device, dtype=torch.long)
            selected = intermediate.index_select(dim=-1, index=neuron_ids).to(module.down_proj.weight.dtype)
            weight = module.down_proj.weight.index_select(dim=1, index=neuron_ids)
            contribution = F.linear(selected, weight).to(output.dtype)
            if self.position_mask is not None:
                mask = self.position_mask.to(output.device).to(output.dtype).view(1, -1, 1)
                contribution = contribution * mask
            return output - contribution

        return hook


@torch.no_grad()
def score_replay(model, input_ids, score_mask):
    outputs = model(input_ids=input_ids, use_cache=False, return_dict=True)
    logits = outputs.logits[:, :-1, :].float()
    labels = input_ids[:, 1:]
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        reduction="none",
    ).view_as(labels)
    shifted_mask = score_mask[1:].to(losses.device)
    selected = losses[0][shifted_mask]
    if selected.numel() == 0:
        return {"n_tokens": 0, "nll_mean": float("nan"), "nll_sum": float("nan")}
    return {
        "n_tokens": int(selected.numel()),
        "nll_mean": float(selected.mean().item()),
        "nll_sum": float(selected.sum().item()),
    }


def main() -> None:
    args = parse_args()
    candidates = parse_candidates(args.candidate)
    if args.candidate_csv:
        candidates.extend(read_candidate_csv(args.candidate_csv, args.top_n))
    candidates = sorted(set(candidates))
    if not candidates:
        raise ValueError("provide at least one --candidate or --candidate-csv row")

    cfg = load_cfg(args.cfg)
    model, tokenizer, vq_model = build_emu3p5(
        cfg.model_path,
        cfg.tokenizer_path,
        cfg.vq_path,
        vq_type=cfg.vq_type,
        model_device=cfg.hf_device,
        vq_device=cfg.vq_device,
        **getattr(cfg, "diffusion_decoder_kwargs", {}),
    )
    model.eval()
    cfg.special_token_ids = {k: tokenizer.encode(v)[0] for k, v in cfg.special_tokens.items()}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(args.raw_dir)

    rows = []
    for sample_name, question in tqdm(normalize_prompts(cfg, args.max_prompts)):
        raw_path = raw_dir / f"{sample_name}.txt"
        if not raw_path.exists():
            rows.append({"sample_id": sample_name, "status": "missing_raw", "raw_path": str(raw_path)})
            continue

        _prompt_text, input_ids, phases, token_types = build_replay_ids(
            cfg, tokenizer, vq_model, question, raw_path, model.device
        )
        score_mask_values = []
        ablate_mask_values = []
        for phase, token_type in zip(phases, token_types):
            score_mask_values.append(
                phase == "output" and (args.target_token_type == "all" or token_type == args.target_token_type)
            )
            ablate_mask_values.append(args.ablate_phase == "all" or phase == args.ablate_phase)
        score_mask = torch.tensor(score_mask_values, dtype=torch.bool, device=input_ids.device)
        ablate_mask = torch.tensor(ablate_mask_values, dtype=torch.bool, device=input_ids.device)

        baseline = score_replay(model, input_ids, score_mask)
        ablator = MLPNeuronAblator(model, candidates, position_mask=ablate_mask)
        ablator.install()
        try:
            ablated = score_replay(model, input_ids, score_mask)
        finally:
            ablator.remove()

        rows.append(
            {
                "sample_id": sample_name,
                "status": "ok",
                "raw_path": str(raw_path),
                "target_token_type": args.target_token_type,
                "ablate_phase": args.ablate_phase,
                "candidates": ";".join(f"{layer}:{neuron}" for layer, neuron in candidates),
                "n_tokens": baseline["n_tokens"],
                "baseline_nll_mean": baseline["nll_mean"],
                "ablated_nll_mean": ablated["nll_mean"],
                "delta_nll_mean": ablated["nll_mean"] - baseline["nll_mean"],
                "baseline_nll_sum": baseline["nll_sum"],
                "ablated_nll_sum": ablated["nll_sum"],
                "delta_nll_sum": ablated["nll_sum"] - baseline["nll_sum"],
            }
        )

    write_csv_rows(rows, out_dir / "mlp_neuron_ablation_likelihood.csv")
    ok_rows = [
        row
        for row in rows
        if row.get("status") == "ok"
        and int(row["n_tokens"]) > 0
        and math.isfinite(float(row["delta_nll_mean"]))
        and math.isfinite(float(row["delta_nll_sum"]))
    ]
    if ok_rows:
        total_tokens = sum(int(row["n_tokens"]) for row in ok_rows)
        summary = [
            {
                "n_samples": len(ok_rows),
                "total_tokens": total_tokens,
                "mean_delta_nll_mean": sum(float(row["delta_nll_mean"]) for row in ok_rows) / len(ok_rows),
                "sum_delta_nll_sum": sum(float(row["delta_nll_sum"]) for row in ok_rows),
            }
        ]
        write_csv_rows(summary, out_dir / "summary.csv")
    print(f"[INFO] ablation likelihood saved to {out_dir}")


if __name__ == "__main__":
    main()
