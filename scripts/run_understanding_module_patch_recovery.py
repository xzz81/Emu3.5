#!/usr/bin/env python3
# Copyright 2026
# SPDX-License-Identifier: Apache-2.0

"""Clean/corrupt module-output patching for Emu3.5 hue understanding."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Dict, List, Sequence, Tuple

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


MODULE_CHOICES = ("self_attn", "mlp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/ume_main_image_read_hue_control_gt_greedy_seed71.py")
    parser.add_argument("--manifest", default="/workspace/home/AAAI 2027/research_logs/hue_control_gt_images_seed70/manifest.json")
    parser.add_argument("--layers", default="48,52,56,60,61,62", help="Comma-separated decoder layer ids, or 'all'.")
    parser.add_argument("--modules", default="self_attn,mlp", help="Comma-separated modules: self_attn,mlp.")
    parser.add_argument("--max-pairs", type=int, default=2)
    parser.add_argument("--question-template", default="What color is the {shape}? Answer with one word.")
    parser.add_argument(
        "--patch-scope",
        choices=["post-image-text-prompt", "question-text-no-score", "score-positions"],
        default="question-text-no-score",
        help="Sequence positions where clean module outputs are patched into corrupt run.",
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


def parse_layers(text: str, num_layers: int) -> List[int]:
    if text.strip().lower() == "all":
        return list(range(num_layers))
    layers = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        layer_idx = int(item)
        if layer_idx < 0 or layer_idx >= num_layers:
            raise ValueError(f"layer out of range: {layer_idx} for num_layers={num_layers}")
        layers.append(layer_idx)
    if not layers:
        raise ValueError("provide at least one layer")
    return sorted(set(layers))


def parse_modules(text: str) -> List[str]:
    modules = []
    for item in text.split(","):
        module = item.strip()
        if not module:
            continue
        if module not in MODULE_CHOICES:
            raise ValueError(f"unknown module {module}; choices={MODULE_CHOICES}")
        modules.append(module)
    if not modules:
        raise ValueError("provide at least one module")
    return sorted(set(modules))


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
    encoded.sort(key=lambda item: (item[1].shape[1], len(item[0])))
    return encoded[0]


def first_tensor(output):
    if isinstance(output, tuple):
        return output[0]
    return output


def replace_first_tensor(output, hidden_states):
    if isinstance(output, tuple):
        return (hidden_states,) + output[1:]
    return hidden_states


def get_module(model, layer_idx: int, module_name: str):
    layer = model.model.layers[layer_idx]
    if module_name == "self_attn":
        return layer.self_attn
    if module_name == "mlp":
        return layer.mlp
    raise ValueError(f"unknown module_name: {module_name}")


class CleanModuleCollector:
    def __init__(self, model, layers: Sequence[int], modules: Sequence[str]):
        self.model = model
        self.layers = list(layers)
        self.modules = list(modules)
        self.handles = []
        self.clean_outputs: Dict[Tuple[int, str], torch.Tensor] = {}

    def install(self):
        for layer_idx in self.layers:
            for module_name in self.modules:
                module = get_module(self.model, layer_idx, module_name)
                self.handles.append(module.register_forward_hook(self._hook(layer_idx, module_name)))

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _hook(self, layer_idx: int, module_name: str):
        def hook(_module, _inputs, output):
            self.clean_outputs[(layer_idx, module_name)] = first_tensor(output).detach()

        return hook


class ModuleOutputPatcher:
    def __init__(self, model, layer_idx: int, module_name: str, clean_output: torch.Tensor, position_mask: torch.Tensor):
        self.model = model
        self.layer_idx = layer_idx
        self.module_name = module_name
        self.clean_output = clean_output
        self.position_mask = position_mask
        self.handle = None

    def install(self):
        module = get_module(self.model, self.layer_idx, self.module_name)
        self.handle = module.register_forward_hook(self._hook)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def _hook(self, _module, _inputs, output):
        corrupt = first_tensor(output)
        seq_len = min(corrupt.shape[1], self.clean_output.shape[1], self.position_mask.numel())
        mask = self.position_mask[:seq_len].to(corrupt.device).to(corrupt.dtype).view(1, seq_len, 1)
        clean = self.clean_output[:, :seq_len, :].to(device=corrupt.device, dtype=corrupt.dtype)
        patched = corrupt.clone()
        patched[:, :seq_len, :] = torch.where(mask.bool(), clean, patched[:, :seq_len, :])
        return replace_first_tensor(output, patched)


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


def build_patch_mask(full_ids, prompt_len: int, answer_len: int, patch_scope: str):
    mask = torch.zeros(full_ids.shape[1], dtype=torch.bool, device=full_ids.device)
    if patch_scope == "score-positions":
        start = max(0, prompt_len - 1)
        mask[start : start + answer_len] = True
    elif patch_scope in ("post-image-text-prompt", "question-text-no-score"):
        prompt_ids = full_ids[0, :prompt_len]
        eoi_positions = (prompt_ids == EOI).nonzero(as_tuple=False).flatten()
        start = int(eoi_positions[-1].item()) + 1 if int(eoi_positions.numel()) > 0 else 0
        post_image = torch.zeros_like(prompt_ids, dtype=torch.bool)
        post_image[start:prompt_len] = True
        text_like = prompt_ids < BOV
        special_image = (prompt_ids == BOI) | (prompt_ids == IMG) | (prompt_ids == EOI)
        mask[:prompt_len] = post_image & text_like & ~special_image
        if patch_scope == "question-text-no-score":
            score_start = max(0, prompt_len - 1)
            mask[score_start:prompt_len] = False
    else:
        raise ValueError(f"unknown patch_scope: {patch_scope}")
    return mask


def run_direction(model, cfg, tokenizer, vq_model, clean_row, corrupt_row, layers, modules, question_template, patch_scope):
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

    collector = CleanModuleCollector(model, layers, modules)
    collector.install()
    try:
        model.model(input_ids=clean_full, use_cache=False, return_dict=True)
    finally:
        collector.remove()
    torch.cuda.empty_cache()

    patch_mask = build_patch_mask(corrupt_full, corrupt_prompt.shape[1], answer_ids.shape[1], patch_scope)
    rows = []
    for layer_idx in layers:
        for module_name in modules:
            key = (layer_idx, module_name)
            if key not in collector.clean_outputs:
                raise RuntimeError(f"missing clean output for layer={layer_idx} module={module_name}")
            patcher = ModuleOutputPatcher(model, layer_idx, module_name, collector.clean_outputs[key], patch_mask)
            patcher.install()
            try:
                patched = answer_nll(model, corrupt_prompt, answer_ids)
            finally:
                patcher.remove()
            torch.cuda.empty_cache()

            denom = corrupt["nll_sum"] - clean["nll_sum"]
            recovery = (corrupt["nll_sum"] - patched["nll_sum"]) / denom if abs(denom) > 1e-9 else float("nan")
            rows.append(
                {
                    "layer": layer_idx,
                    "module": module_name,
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
            )
    return rows


def summarize(rows: List[dict]) -> List[dict]:
    by_key: Dict[Tuple[int, str], List[dict]] = {}
    for row in rows:
        if math.isfinite(float(row["recovery_fraction"])):
            by_key.setdefault((int(row["layer"]), str(row["module"])), []).append(row)
    summary = []
    for (layer_idx, module_name), group_rows in sorted(by_key.items()):
        recoveries = [float(row["recovery_fraction"]) for row in group_rows]
        deltas = [float(row["patched_delta_vs_corrupt_nll_sum"]) for row in group_rows]
        gaps = [float(row["corrupt_minus_clean_nll_sum"]) for row in group_rows]
        summary.append(
            {
                "layer": layer_idx,
                "module": module_name,
                "n_directions": len(group_rows),
                "helped_directions": sum(1 for delta in deltas if delta < 0),
                "mean_recovery_fraction": sum(recoveries) / len(recoveries),
                "median_recovery_fraction": statistics.median(recoveries),
                "mean_patch_delta_vs_corrupt_nll_sum": sum(deltas) / len(deltas),
                "sum_corrupt_minus_clean_nll_sum": sum(gaps),
                "sum_patch_delta_vs_corrupt_nll_sum": sum(deltas),
                "mean_patched_positions": sum(int(row["patched_positions"]) for row in group_rows) / len(group_rows),
            }
        )
    return summary


def write_report(out_dir: Path, args: argparse.Namespace, summary_rows: List[dict]) -> None:
    lines = [
        "# Stage 5 Report: Understanding Module Output Patch Recovery",
        "",
        "## Purpose",
        "",
        "This experiment patches clean self-attention or MLP module outputs into corrupt-image hue QA runs.",
        "",
        "## Setup",
        "",
        f"- Layers: `{args.layers}`",
        f"- Modules: `{args.modules}`",
        f"- Patch scope: `{args.patch_scope}`",
        f"- Max pairs: `{args.max_pairs}`",
        "",
        "Metric:",
        "",
        "```text",
        "recovery = (corrupt_nll - patched_nll) / (corrupt_nll - clean_nll)",
        "```",
        "",
        "## Aggregate Results",
        "",
        "| layer | module | directions | helped | mean recovery | median recovery | sum delta NLL vs corrupt |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {layer} | {module} | {n_directions} | {helped_directions} | {mean_recovery_fraction:+.6f} | "
            "{median_recovery_fraction:+.6f} | {sum_patch_delta_vs_corrupt_nll_sum:+.6f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Interpretation Guide",
            "",
            "- Positive recovery means the clean module output carries information that helps recover the clean answer.",
            "- Comparing `self_attn` and `mlp` at the same layer helps localize which submodule builds the text-side representation.",
            "",
        ]
    )
    (out_dir / "understanding_module_patch_recovery_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
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
    layers = parse_layers(args.layers, len(model.model.layers))
    modules = parse_modules(args.modules)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for _pair_id, a_row, b_row in tqdm(grouped_manifest(args.manifest, args.max_pairs)):
        for clean_row, corrupt_row in ((a_row, b_row), (b_row, a_row)):
            rows.extend(
                run_direction(
                    model,
                    cfg,
                    tokenizer,
                    vq_model,
                    clean_row,
                    corrupt_row,
                    layers,
                    modules,
                    args.question_template,
                    args.patch_scope,
                )
            )

    write_csv_rows(rows, out_dir / "understanding_module_patch_recovery.csv")
    summary_rows = summarize(rows)
    if summary_rows:
        write_csv_rows(summary_rows, out_dir / "summary.csv")
    write_report(out_dir, args, summary_rows)
    print(f"[INFO] understanding module patch recovery saved to {out_dir}")


if __name__ == "__main__":
    main()
