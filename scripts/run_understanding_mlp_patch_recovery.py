#!/usr/bin/env python3
# Copyright 2026
# SPDX-License-Identifier: Apache-2.0

"""Clean/corrupt MLP-neuron activation patching for Emu3.5 hue understanding."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
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

from src.utils.activation_probe import write_csv_rows  # noqa: E402
from src.utils.input_utils import build_image  # noqa: E402
from src.utils.logits_processor import BOI, BOV, EOI, IMG  # noqa: E402
from src.utils.model_utils import build_emu3p5  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/ume_main_image_read_hue_control_gt_greedy_seed71.py")
    parser.add_argument("--manifest", default="/workspace/home/AAAI 2027/research_logs/hue_control_gt_images_seed70/manifest.json")
    parser.add_argument("--candidate", action="append", default=[], help="Layer/neuron as LAYER:NEURON.")
    parser.add_argument("--candidate-csv", default=None, help="CSV with layer and neuron columns.")
    parser.add_argument("--top-n", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=2)
    parser.add_argument("--question-template", default="What color is the {shape}? Answer with one word.")
    parser.add_argument(
        "--patch-scope",
        choices=[
            "prompt",
            "answer",
            "all",
            "visual-prompt",
            "eoi-prompt",
            "post-image-text-prompt",
            "question-text-no-score",
            "score-positions",
        ],
        default="prompt",
        help="Sequence positions where clean selected-neuron activations are patched into corrupt run.",
    )
    parser.add_argument("--out-dir", required=True)
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


def grouped_manifest(path: str, max_pairs: int):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    by_pair: Dict[str, List[dict]] = {}
    for row in rows:
        by_pair.setdefault(row["pair_id"], []).append(row)
    pairs = []
    for pair_id, pair_rows in sorted(by_pair.items()):
        if len(pair_rows) != 2:
            continue
        pair_rows = sorted(pair_rows, key=lambda item: item["variant"])
        pairs.append((pair_id, pair_rows[0], pair_rows[1]))
        if max_pairs and len(pairs) >= max_pairs:
            break
    return pairs


def build_prompt_ids(cfg, tokenizer, vq_model, image_path: str, question: str, device):
    _unc_prompt, template = cfg.build_unc_and_template(getattr(cfg, "task_type", "unknown"), with_image=True)
    image_str = build_image(Image.open(image_path).convert("RGB"), cfg, tokenizer, vq_model)
    prompt = template.format(question=question).replace("<|IMAGE|>", image_str)
    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    if input_ids[0, 0] != cfg.special_token_ids["BOS"]:
        bos = torch.tensor([[cfg.special_token_ids["BOS"]]], device=device, dtype=input_ids.dtype)
        input_ids = torch.cat([bos, input_ids], dim=1)
    return input_ids


def encode_answer(tokenizer, answer: str, device):
    variants = [answer, " " + answer, answer.capitalize(), " " + answer.capitalize()]
    encoded = []
    for text in variants:
        ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=False).to(device)
        if ids.numel() > 0:
            encoded.append((text, ids))
    # Prefer the shortest continuation; this avoids making recovery depend on prose.
    encoded.sort(key=lambda item: (item[1].shape[1], len(item[0])))
    return encoded[0]


class CleanMLPCollector:
    def __init__(self, model, candidates: List[Tuple[int, int]]):
        self.model = model
        self.by_layer: Dict[int, List[int]] = {}
        for layer, neuron in candidates:
            self.by_layer.setdefault(int(layer), []).append(int(neuron))
        self.handles = []
        self._gate = {}
        self.clean_selected = {}

    def install(self):
        for layer_idx in self.by_layer:
            mlp = self.model.model.layers[layer_idx].mlp
            self.handles.append(mlp.gate_proj.register_forward_hook(self._gate_hook(layer_idx, mlp.act_fn)))
            self.handles.append(mlp.up_proj.register_forward_hook(self._up_hook(layer_idx)))

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self._gate.clear()

    def _gate_hook(self, layer_idx, act_fn):
        def hook(_module, _inputs, output):
            self._gate[layer_idx] = act_fn(output.detach())

        return hook

    def _up_hook(self, layer_idx):
        def hook(_module, _inputs, output):
            gate = self._gate.pop(layer_idx, None)
            if gate is None:
                return
            neurons = torch.tensor(self.by_layer[layer_idx], device=output.device, dtype=torch.long)
            selected = (gate * output.detach()).index_select(dim=-1, index=neurons)
            self.clean_selected[layer_idx] = selected.detach()

        return hook


class MLPNeuronPatcher:
    def __init__(self, model, candidates: List[Tuple[int, int]], clean_selected: dict, position_mask: torch.Tensor):
        self.model = model
        self.clean_selected = clean_selected
        self.position_mask = position_mask
        self.by_layer: Dict[int, List[int]] = {}
        for layer, neuron in candidates:
            self.by_layer.setdefault(int(layer), []).append(int(neuron))
        self.handles = []
        self._gate = {}
        self._corrupt_selected = {}

    def install(self):
        for layer_idx in self.by_layer:
            mlp = self.model.model.layers[layer_idx].mlp
            self.handles.append(mlp.gate_proj.register_forward_hook(self._gate_hook(layer_idx, mlp.act_fn)))
            self.handles.append(mlp.up_proj.register_forward_hook(self._up_hook(layer_idx)))
            self.handles.append(mlp.register_forward_hook(self._mlp_hook(layer_idx)))

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self._gate.clear()
        self._corrupt_selected.clear()

    def _gate_hook(self, layer_idx, act_fn):
        def hook(_module, _inputs, output):
            self._gate[layer_idx] = act_fn(output.detach())

        return hook

    def _up_hook(self, layer_idx):
        def hook(_module, _inputs, output):
            gate = self._gate.pop(layer_idx, None)
            if gate is None:
                return
            neurons = torch.tensor(self.by_layer[layer_idx], device=output.device, dtype=torch.long)
            selected = (gate * output.detach()).index_select(dim=-1, index=neurons)
            self._corrupt_selected[layer_idx] = selected.detach()

        return hook

    def _mlp_hook(self, layer_idx):
        def hook(module, _inputs, output):
            clean = self.clean_selected.get(layer_idx)
            corrupt = self._corrupt_selected.pop(layer_idx, None)
            if clean is None or corrupt is None:
                return output
            seq_len = min(clean.shape[1], corrupt.shape[1], output.shape[1], self.position_mask.numel())
            delta = clean[:, :seq_len, :] - corrupt[:, :seq_len, :]
            weight = module.down_proj.weight.index_select(
                dim=1,
                index=torch.tensor(self.by_layer[layer_idx], device=output.device, dtype=torch.long),
            )
            contribution_delta = F.linear(delta.to(weight.dtype), weight).to(output.dtype)
            mask = self.position_mask[:seq_len].to(output.device).to(output.dtype).view(1, seq_len, 1)
            patched = output.clone()
            patched[:, :seq_len, :] = patched[:, :seq_len, :] + contribution_delta * mask
            return patched

        return hook


@torch.no_grad()
def answer_nll(model, prompt_ids, answer_ids):
    full_ids = torch.cat([prompt_ids, answer_ids], dim=1)
    answer_start = prompt_ids.shape[1]
    outputs = model.model(input_ids=full_ids, use_cache=False, return_dict=True)
    hidden_states = outputs.last_hidden_state[:, answer_start - 1 : -1, :]
    logits = model.lm_head(hidden_states).float()
    labels = answer_ids.to(logits.device)
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        reduction="none",
    )
    return {
        "full_ids": full_ids,
        "nll_mean": float(losses.mean().item()),
        "nll_sum": float(losses.sum().item()),
        "answer_tokens": int(labels.numel()),
    }


def build_patch_mask(corrupt_full, corrupt_prompt_len: int, answer_len: int, patch_scope: str):
    mask = torch.zeros(corrupt_full.shape[1], dtype=torch.bool, device=corrupt_full.device)
    if patch_scope == "prompt":
        mask[:corrupt_prompt_len] = True
    elif patch_scope == "answer":
        mask[corrupt_prompt_len:] = True
    elif patch_scope == "all":
        mask[:] = True
    elif patch_scope == "visual-prompt":
        prompt_ids = corrupt_full[0, :corrupt_prompt_len]
        mask[:corrupt_prompt_len] = prompt_ids >= BOV
    elif patch_scope == "eoi-prompt":
        prompt_ids = corrupt_full[0, :corrupt_prompt_len]
        mask[:corrupt_prompt_len] = prompt_ids == EOI
    elif patch_scope in ("post-image-text-prompt", "question-text-no-score"):
        prompt_ids = corrupt_full[0, :corrupt_prompt_len]
        eoi_positions = (prompt_ids == EOI).nonzero(as_tuple=False).flatten()
        start = int(eoi_positions[-1].item()) + 1 if int(eoi_positions.numel()) > 0 else 0
        post_image = torch.zeros_like(prompt_ids, dtype=torch.bool)
        post_image[start:corrupt_prompt_len] = True
        text_like = prompt_ids < BOV
        special_image = (prompt_ids == BOI) | (prompt_ids == IMG) | (prompt_ids == EOI)
        mask[:corrupt_prompt_len] = post_image & text_like & ~special_image
        if patch_scope == "question-text-no-score":
            score_start = max(0, corrupt_prompt_len - 1)
            mask[score_start:corrupt_prompt_len] = False
    elif patch_scope == "score-positions":
        start = max(0, corrupt_prompt_len - 1)
        mask[start : start + answer_len] = True
    else:
        raise ValueError(f"unknown patch_scope: {patch_scope}")
    return mask


def run_direction(model, cfg, tokenizer, vq_model, clean_row, corrupt_row, candidates, question_template, patch_scope):
    question = question_template.format(shape=clean_row["shape"])
    answer_text, answer_ids = encode_answer(tokenizer, clean_row["fill_color"], model.device)
    clean_prompt = build_prompt_ids(cfg, tokenizer, vq_model, clean_row["local_path"], question, model.device)
    corrupt_prompt = build_prompt_ids(cfg, tokenizer, vq_model, corrupt_row["local_path"], question, model.device)
    if clean_prompt.shape != corrupt_prompt.shape:
        raise RuntimeError(f"prompt shape mismatch: {clean_prompt.shape} vs {corrupt_prompt.shape}")

    clean = answer_nll(model, clean_prompt, answer_ids)
    torch.cuda.empty_cache()
    corrupt = answer_nll(model, corrupt_prompt, answer_ids)
    torch.cuda.empty_cache()
    clean_full = clean["full_ids"]
    corrupt_full = torch.cat([corrupt_prompt, answer_ids], dim=1)

    collector = CleanMLPCollector(model, candidates)
    collector.install()
    try:
        model.model(input_ids=clean_full, use_cache=False, return_dict=True)
    finally:
        collector.remove()
    torch.cuda.empty_cache()

    patch_mask = build_patch_mask(corrupt_full, corrupt_prompt.shape[1], answer_ids.shape[1], patch_scope)
    patcher = MLPNeuronPatcher(model, candidates, collector.clean_selected, patch_mask)
    patcher.install()
    try:
        patched = answer_nll(model, corrupt_prompt, answer_ids)
    finally:
        patcher.remove()
    torch.cuda.empty_cache()

    denom = corrupt["nll_sum"] - clean["nll_sum"]
    recovery = (corrupt["nll_sum"] - patched["nll_sum"]) / denom if abs(denom) > 1e-9 else float("nan")
    return {
        "pair_id": clean_row["pair_id"],
        "clean_image_id": clean_row["image_id"],
        "corrupt_image_id": corrupt_row["image_id"],
        "shape": clean_row["shape"],
        "clean_color": clean_row["fill_color"],
        "corrupt_color": corrupt_row["fill_color"],
        "answer_text": answer_text,
        "answer_tokens": clean["answer_tokens"],
        "clean_nll_sum": clean["nll_sum"],
        "corrupt_nll_sum": corrupt["nll_sum"],
        "patched_nll_sum": patched["nll_sum"],
        "clean_nll_mean": clean["nll_mean"],
        "corrupt_nll_mean": corrupt["nll_mean"],
        "patched_nll_mean": patched["nll_mean"],
        "corrupt_minus_clean_nll_sum": denom,
        "patched_delta_vs_corrupt_nll_sum": patched["nll_sum"] - corrupt["nll_sum"],
        "recovery_fraction": recovery,
        "patch_scope": patch_scope,
        "patched_positions": int(patch_mask.sum().item()),
    }


def main() -> None:
    args = parse_args()
    candidates = parse_candidates(args.candidate)
    if args.candidate_csv:
        candidates.extend(read_candidate_csv(args.candidate_csv, args.top_n))
    candidates = sorted(set(candidates))
    if not candidates:
        raise ValueError("provide candidates")

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
    rows = []
    for _pair_id, a_row, b_row in tqdm(grouped_manifest(args.manifest, args.max_pairs)):
        for clean_row, corrupt_row in ((a_row, b_row), (b_row, a_row)):
            row = run_direction(
                model,
                cfg,
                tokenizer,
                vq_model,
                clean_row,
                corrupt_row,
                candidates,
                args.question_template,
                args.patch_scope,
            )
            row["candidates"] = ";".join(f"{layer}:{neuron}" for layer, neuron in candidates)
            rows.append(row)

    write_csv_rows(rows, out_dir / "understanding_mlp_patch_recovery.csv")
    valid = [
        row
        for row in rows
        if math.isfinite(float(row["recovery_fraction"]))
    ]
    if valid:
        summary = [
            {
                "n_directions": len(valid),
                "mean_recovery_fraction": sum(float(row["recovery_fraction"]) for row in valid) / len(valid),
                "mean_patch_delta_vs_corrupt_nll_sum": sum(
                    float(row["patched_delta_vs_corrupt_nll_sum"]) for row in valid
                )
                / len(valid),
                "sum_corrupt_minus_clean_nll_sum": sum(float(row["corrupt_minus_clean_nll_sum"]) for row in valid),
                "sum_patch_delta_vs_corrupt_nll_sum": sum(
                    float(row["patched_delta_vs_corrupt_nll_sum"]) for row in valid
                ),
            }
        ]
        write_csv_rows(summary, out_dir / "summary.csv")
    print(f"[INFO] understanding MLP patch recovery saved to {out_dir}")


if __name__ == "__main__":
    main()
