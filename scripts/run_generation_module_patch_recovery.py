#!/usr/bin/env python3
# Copyright 2026
# SPDX-License-Identifier: Apache-2.0

"""Clean/corrupt module-output patching for Emu3.5 generation raw targets."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Dict, List, Sequence, Tuple

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

from scripts.run_generation_residual_patch_recovery import (  # noqa: E402
    build_patch_mask,
    build_prompt_ids,
    filter_pairs,
    first_tensor,
    get_target_ids,
    load_cfg,
    normalize_prompt_rows,
    parse_layers,
    replace_first_tensor,
    target_nll,
)
from src.utils.activation_probe import write_csv_rows  # noqa: E402
from src.utils.model_utils import build_emu3p5  # noqa: E402


MODULE_CHOICES = ("self_attn", "mlp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--layers", default="60,61", help="Comma-separated decoder layer ids, or 'all'.")
    parser.add_argument("--modules", default="self_attn,mlp", help="Comma-separated modules: self_attn,mlp.")
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument(
        "--pair-id",
        action="append",
        default=[],
        help="Only run matching pair id(s). Can be passed multiple times.",
    )
    parser.add_argument(
        "--patch-scope",
        choices=["prompt", "color-token", "object-token", "visual-score-positions", "all-score-positions"],
        default="prompt",
    )
    parser.add_argument(
        "--score-scope",
        choices=["visual", "visual-and-structure", "all-target"],
        default="visual",
    )
    parser.add_argument("--target-cache-dir", required=True)
    parser.add_argument("--max-target-tokens", type=int, default=0)
    parser.add_argument("--target-height", type=int, default=None)
    parser.add_argument("--target-width", type=int, default=None)
    parser.add_argument("--image-area", type=int, default=None)
    parser.add_argument("--generation-max-new-tokens", type=int, default=None)
    parser.add_argument("--classifier-free-guidance", type=float, default=None)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


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
    def __init__(
        self,
        model,
        layer_idx: int,
        module_name: str,
        clean_output: torch.Tensor,
        position_mask: torch.Tensor,
        source_position_for_target: Dict[int, int] | None = None,
    ):
        self.model = model
        self.layer_idx = int(layer_idx)
        self.module_name = module_name
        self.clean_output = clean_output
        self.position_mask = position_mask
        self.source_position_for_target = source_position_for_target or {}
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
        mask = self.position_mask[:seq_len].to(corrupt.device).to(torch.bool).view(1, seq_len, 1)
        clean = self.clean_output[:, :seq_len, :].to(device=corrupt.device, dtype=corrupt.dtype)
        patched = corrupt.clone()
        patched[:, :seq_len, :] = torch.where(mask, clean, patched[:, :seq_len, :])
        for target_pos, source_pos in self.source_position_for_target.items():
            if 0 <= target_pos < corrupt.shape[1] and 0 <= source_pos < self.clean_output.shape[1]:
                patched[:, target_pos, :] = self.clean_output[:, source_pos, :].to(
                    device=corrupt.device,
                    dtype=corrupt.dtype,
                )
        return replace_first_tensor(output, patched)


def run_direction(
    model,
    cfg,
    tokenizer,
    clean_row,
    corrupt_row,
    layers,
    modules,
    patch_scope,
    score_scope,
    max_target_tokens,
    target_cache_dir: Path,
):
    clean_prompt, unconditional_ids, full_unc_ids = build_prompt_ids(cfg, tokenizer, clean_row["prompt"], model.device)
    corrupt_prompt, _corrupt_unc, _corrupt_full_unc = build_prompt_ids(cfg, tokenizer, corrupt_row["prompt"], model.device)

    seed_offset = sum(ord(ch) for ch in clean_row["sample_id"])
    target_ids, target_source, target_cache_path = get_target_ids(
        cfg,
        model,
        tokenizer,
        clean_row,
        clean_prompt,
        unconditional_ids,
        full_unc_ids,
        int(cfg.seed) + seed_offset,
        target_cache_dir,
        False,
    )
    if max_target_tokens and target_ids.shape[1] > max_target_tokens:
        target_ids = target_ids[:, :max_target_tokens]

    clean = target_nll(model, clean_prompt, target_ids, score_scope)
    torch.cuda.empty_cache()
    corrupt = target_nll(model, corrupt_prompt, target_ids, score_scope)
    torch.cuda.empty_cache()
    clean_full = clean["full_ids"]
    corrupt_full = torch.cat([corrupt_prompt, target_ids], dim=1)

    collector = CleanModuleCollector(model, layers, modules)
    collector.install()
    try:
        model.model(input_ids=clean_full, use_cache=False, return_dict=True)
    finally:
        collector.remove()
    torch.cuda.empty_cache()

    patch_mask, patched_positions, patch_note, source_position_for_target = build_patch_mask(
        tokenizer,
        clean_full,
        corrupt_full,
        clean_prompt.shape[1],
        corrupt_prompt.shape[1],
        target_ids,
        patch_scope,
        clean_row["color"],
        corrupt_row["color"],
        clean_row["shape"],
        corrupt_row["shape"],
    )

    rows = []
    for layer_idx in layers:
        for module_name in modules:
            key = (layer_idx, module_name)
            if key not in collector.clean_outputs:
                raise RuntimeError(f"missing clean output for layer={layer_idx} module={module_name}")
            patcher = ModuleOutputPatcher(
                model,
                layer_idx,
                module_name,
                collector.clean_outputs[key],
                patch_mask,
                source_position_for_target=source_position_for_target,
            )
            patcher.install()
            try:
                patched = target_nll(model, corrupt_prompt, target_ids, score_scope)
            finally:
                patcher.remove()
            torch.cuda.empty_cache()

            denom = corrupt["nll_sum"] - clean["nll_sum"]
            valid_recovery_gap = denom > 1e-9
            recovery = (corrupt["nll_sum"] - patched["nll_sum"]) / denom if valid_recovery_gap else float("nan")
            rows.append(
                {
                    "layer": layer_idx,
                    "module": module_name,
                    "pair_id": clean_row["pair_id"],
                    "clean_sample_id": clean_row["sample_id"],
                    "corrupt_sample_id": corrupt_row["sample_id"],
                    "shape": clean_row["shape"],
                    "clean_color": clean_row["color"],
                    "corrupt_color": corrupt_row["color"],
                    "target_tokens": clean["target_tokens"],
                    "visual_tokens": clean["visual_tokens"],
                    "scored_tokens": clean["scored_tokens"],
                    "target_source": target_source,
                    "target_cache_path": target_cache_path,
                    "clean_nll_sum": clean["nll_sum"],
                    "corrupt_nll_sum": corrupt["nll_sum"],
                    "patched_nll_sum": patched["nll_sum"],
                    "clean_nll_mean": clean["nll_mean"],
                    "corrupt_nll_mean": corrupt["nll_mean"],
                    "patched_nll_mean": patched["nll_mean"],
                    "corrupt_minus_clean_nll_sum": denom,
                    "patched_delta_vs_corrupt_nll_sum": patched["nll_sum"] - corrupt["nll_sum"],
                    "recovery_fraction": recovery,
                    "valid_recovery_gap": int(valid_recovery_gap),
                    "patch_scope": patch_scope,
                    "score_scope": score_scope,
                    "patched_positions": patched_positions,
                    "patch_note": patch_note,
                }
            )
    return rows


def summarize(rows: List[dict]) -> List[dict]:
    grouped: Dict[Tuple[int, str], List[dict]] = {}
    all_grouped: Dict[Tuple[int, str], List[dict]] = {}
    for row in rows:
        key = (int(row["layer"]), row["module"])
        all_grouped.setdefault(key, []).append(row)
        if math.isfinite(float(row["recovery_fraction"])):
            grouped.setdefault(key, []).append(row)
    summary = []
    for key, all_rows in sorted(all_grouped.items()):
        layer_idx, module_name = key
        valid_rows = grouped.get(key, [])
        if not valid_rows:
            summary.append(
                {
                    "layer": layer_idx,
                    "module": module_name,
                    "n_directions": 0,
                    "invalid_gap_directions": len(all_rows),
                    "helped_directions": 0,
                    "mean_recovery_fraction": float("nan"),
                    "median_recovery_fraction": float("nan"),
                    "mean_patch_delta_vs_corrupt_nll_sum": float("nan"),
                    "sum_corrupt_minus_clean_nll_sum": sum(float(row["corrupt_minus_clean_nll_sum"]) for row in all_rows),
                    "sum_patch_delta_vs_corrupt_nll_sum": sum(float(row["patched_delta_vs_corrupt_nll_sum"]) for row in all_rows),
                }
            )
            continue
        recoveries = [float(row["recovery_fraction"]) for row in valid_rows]
        deltas = [float(row["patched_delta_vs_corrupt_nll_sum"]) for row in valid_rows]
        gaps = [float(row["corrupt_minus_clean_nll_sum"]) for row in valid_rows]
        summary.append(
            {
                "layer": layer_idx,
                "module": module_name,
                "n_directions": len(valid_rows),
                "invalid_gap_directions": len(all_rows) - len(valid_rows),
                "helped_directions": sum(1 for delta in deltas if delta < 0),
                "mean_recovery_fraction": sum(recoveries) / len(recoveries),
                "median_recovery_fraction": statistics.median(recoveries),
                "mean_patch_delta_vs_corrupt_nll_sum": sum(deltas) / len(deltas),
                "sum_corrupt_minus_clean_nll_sum": sum(gaps),
                "sum_patch_delta_vs_corrupt_nll_sum": sum(deltas),
            }
        )
    return summary


def write_report(out_dir: Path, args: argparse.Namespace, summary_rows: List[dict]) -> None:
    lines = [
        "# Generation Module Patch Recovery",
        "",
        f"- Config: `{args.cfg}`",
        f"- Layers: `{args.layers}`",
        f"- Modules: `{args.modules}`",
        f"- Patch scope: `{args.patch_scope}`",
        f"- Score scope: `{args.score_scope}`",
        f"- Target cache dir: `{args.target_cache_dir}`",
        "",
        "| layer | module | valid directions | invalid gaps | helped | mean recovery | median recovery | sum delta NLL vs corrupt |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {layer} | {module} | {n_directions} | {invalid_gap_directions} | {helped_directions} | "
            "{mean_recovery_fraction:+.6f} | {median_recovery_fraction:+.6f} | {sum_patch_delta_vs_corrupt_nll_sum:+.6f} |".format(
                **row
            )
        )
    (out_dir / "generation_module_patch_recovery_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.cfg)
    if args.target_height is not None:
        cfg.target_height = args.target_height
    if args.target_width is not None:
        cfg.target_width = args.target_width
    if args.image_area is not None:
        cfg.image_area = args.image_area
    if args.generation_max_new_tokens is not None:
        cfg.max_new_tokens = args.generation_max_new_tokens
        cfg.sampling_params["max_new_tokens"] = args.generation_max_new_tokens
    if args.classifier_free_guidance is not None:
        cfg.classifier_free_guidance = args.classifier_free_guidance

    model, tokenizer, vq_model = build_emu3p5(
        cfg.model_path,
        cfg.tokenizer_path,
        cfg.vq_path,
        vq_type=cfg.vq_type,
        model_device=cfg.hf_device,
        vq_device=cfg.vq_device,
        **getattr(cfg, "diffusion_decoder_kwargs", {}),
    )
    del vq_model
    model.eval()
    cfg.special_token_ids = {key: tokenizer.encode(value)[0] for key, value in cfg.special_tokens.items()}
    layers = parse_layers(args.layers, len(model.model.layers))
    modules = parse_modules(args.modules)
    pairs = filter_pairs(normalize_prompt_rows(cfg, args.max_pairs), args.pair_id)
    if not pairs:
        raise SystemExit("No generation pairs matched the requested filters.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for _pair_id, a_row, b_row in tqdm(pairs):
        for clean_row, corrupt_row in ((a_row, b_row), (b_row, a_row)):
            rows.extend(
                run_direction(
                    model,
                    cfg,
                    tokenizer,
                    clean_row,
                    corrupt_row,
                    layers,
                    modules,
                    args.patch_scope,
                    args.score_scope,
                    args.max_target_tokens,
                    Path(args.target_cache_dir),
                )
            )

    write_csv_rows(rows, out_dir / "generation_module_patch_recovery.csv")
    summary_rows = summarize(rows)
    if summary_rows:
        write_csv_rows(summary_rows, out_dir / "summary.csv")
    write_report(out_dir, args, summary_rows)
    metadata = {
        "cfg": args.cfg,
        "layers": args.layers,
        "modules": args.modules,
        "patch_scope": args.patch_scope,
        "score_scope": args.score_scope,
        "pair_id": args.pair_id,
        "max_pairs": args.max_pairs,
        "target_cache_dir": args.target_cache_dir,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] generation module patch recovery saved to {out_dir}")


if __name__ == "__main__":
    main()
