#!/usr/bin/env python3
# Copyright 2026
# SPDX-License-Identifier: Apache-2.0

"""Clean/corrupt attention-head output patching for Emu3.5 generation raw targets."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Dict, Iterable, List

import torch
from tqdm import tqdm

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

from scripts.run_generation_residual_patch_recovery import (  # noqa: E402
    build_patch_mask,
    build_prompt_ids,
    filter_pairs,
    get_target_ids,
    load_cfg,
    normalize_prompt_rows,
    target_nll,
)
from src.utils.activation_probe import write_csv_rows  # noqa: E402
from src.utils.model_utils import build_emu3p5  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--layer", type=int, default=60)
    parser.add_argument("--heads", default="all", help="Comma-separated head ids, or 'all'.")
    parser.add_argument(
        "--head-mode",
        choices=["individual", "joint", "both"],
        default="individual",
        help="Patch each requested head separately, all requested heads jointly, or both.",
    )
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
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def parse_heads(text: str, num_heads: int) -> List[int]:
    if text.strip().lower() == "all":
        return list(range(num_heads))
    heads = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        head_idx = int(item)
        if head_idx < 0 or head_idx >= num_heads:
            raise ValueError(f"head out of range: {head_idx} for num_heads={num_heads}")
        heads.append(head_idx)
    if not heads:
        raise ValueError("provide at least one head")
    return sorted(set(heads))


class CleanHeadOutputCollector:
    def __init__(self, model, layer_idx: int):
        self.model = model
        self.layer_idx = int(layer_idx)
        self.handle = None
        self.clean_o_proj_input = None

    def install(self):
        o_proj = self.model.model.layers[self.layer_idx].self_attn.o_proj
        self.handle = o_proj.register_forward_pre_hook(self._hook)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def _hook(self, _module, inputs):
        self.clean_o_proj_input = inputs[0].detach()
        return inputs


class AttentionHeadPatcher:
    def __init__(
        self,
        model,
        layer_idx: int,
        heads: Iterable[int],
        clean_o_proj_input: torch.Tensor,
        position_mask: torch.Tensor,
        source_position_for_target: Dict[int, int] | None = None,
    ):
        self.model = model
        self.layer_idx = int(layer_idx)
        self.heads = [int(head) for head in heads]
        self.clean_o_proj_input = clean_o_proj_input
        self.position_mask = position_mask
        self.source_position_for_target = source_position_for_target or {}
        self.handle = None
        attn = self.model.model.layers[self.layer_idx].self_attn
        self.num_heads = int(attn.num_heads)
        self.head_dim = int(attn.head_dim)

    def install(self):
        o_proj = self.model.model.layers[self.layer_idx].self_attn.o_proj
        self.handle = o_proj.register_forward_pre_hook(self._hook)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def _hook(self, _module, inputs):
        corrupt = inputs[0]
        seq_len = min(corrupt.shape[1], self.clean_o_proj_input.shape[1], self.position_mask.numel())
        mask = self.position_mask[:seq_len].to(corrupt.device).to(torch.bool).view(1, seq_len, 1)
        patched = corrupt.clone()
        clean = self.clean_o_proj_input.to(device=corrupt.device, dtype=corrupt.dtype)
        for head_idx in self.heads:
            start = head_idx * self.head_dim
            end = start + self.head_dim
            clean_slice = clean[:, :seq_len, start:end]
            patched[:, :seq_len, start:end] = torch.where(mask, clean_slice, patched[:, :seq_len, start:end])
            for target_pos, source_pos in self.source_position_for_target.items():
                if 0 <= target_pos < corrupt.shape[1] and 0 <= source_pos < clean.shape[1]:
                    patched[:, target_pos, start:end] = clean[:, source_pos, start:end]
        return (patched,) + tuple(inputs[1:])


def make_result_row(
    layer_idx,
    head_label,
    clean_row,
    corrupt_row,
    clean,
    corrupt,
    patched,
    patch_scope,
    patch_mask,
    patch_note,
):
    denom = corrupt["nll_sum"] - clean["nll_sum"]
    valid_recovery_gap = denom > 1e-9
    recovery = (corrupt["nll_sum"] - patched["nll_sum"]) / denom if valid_recovery_gap else float("nan")
    return {
        "layer": layer_idx,
        "head": head_label,
        "pair_id": clean_row["pair_id"],
        "clean_sample_id": clean_row["sample_id"],
        "corrupt_sample_id": corrupt_row["sample_id"],
        "shape": clean_row["shape"],
        "clean_color": clean_row["color"],
        "corrupt_color": corrupt_row["color"],
        "target_tokens": clean["target_tokens"],
        "visual_tokens": clean["visual_tokens"],
        "scored_tokens": clean["scored_tokens"],
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
        "patched_positions": int(patch_mask.sum().item()),
        "patch_note": patch_note,
    }


def run_direction(
    model,
    cfg,
    tokenizer,
    clean_row,
    corrupt_row,
    layer_idx,
    heads,
    patch_scope,
    score_scope,
    head_mode,
    max_target_tokens,
    target_cache_dir: Path,
):
    clean_prompt, unconditional_ids, full_unc_ids = build_prompt_ids(cfg, tokenizer, clean_row["prompt"], model.device)
    corrupt_prompt, _corrupt_unc, _corrupt_full_unc = build_prompt_ids(cfg, tokenizer, corrupt_row["prompt"], model.device)
    seed_offset = sum(ord(ch) for ch in clean_row["sample_id"])
    target_ids, _target_source, _target_cache_path = get_target_ids(
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

    collector = CleanHeadOutputCollector(model, layer_idx)
    collector.install()
    try:
        model.model(input_ids=clean_full, use_cache=False, return_dict=True)
    finally:
        collector.remove()
    if collector.clean_o_proj_input is None:
        raise RuntimeError(f"missing clean o_proj input for layer {layer_idx}")
    torch.cuda.empty_cache()

    patch_mask, _patched_positions, patch_note, source_position_for_target = build_patch_mask(
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
    if head_mode in ("individual", "both"):
        for head_idx in heads:
            patcher = AttentionHeadPatcher(
                model,
                layer_idx,
                [head_idx],
                collector.clean_o_proj_input,
                patch_mask,
                source_position_for_target=source_position_for_target,
            )
            patcher.install()
            try:
                patched = target_nll(model, corrupt_prompt, target_ids, score_scope)
            finally:
                patcher.remove()
            torch.cuda.empty_cache()
            rows.append(
                make_result_row(
                    layer_idx,
                    str(head_idx),
                    clean_row,
                    corrupt_row,
                    clean,
                    corrupt,
                    patched,
                    patch_scope,
                    patch_mask,
                    patch_note,
                )
            )
    if head_mode in ("joint", "both"):
        patcher = AttentionHeadPatcher(
            model,
            layer_idx,
            heads,
            collector.clean_o_proj_input,
            patch_mask,
            source_position_for_target=source_position_for_target,
        )
        patcher.install()
        try:
            patched = target_nll(model, corrupt_prompt, target_ids, score_scope)
        finally:
            patcher.remove()
        torch.cuda.empty_cache()
        rows.append(
            make_result_row(
                layer_idx,
                "+".join(str(head_idx) for head_idx in heads),
                clean_row,
                corrupt_row,
                clean,
                corrupt,
                patched,
                patch_scope,
                patch_mask,
                patch_note,
            )
        )
    return rows


def head_sort_key(head_label: str):
    return [int(item) for item in str(head_label).split("+")]


def summarize(rows: List[dict]) -> List[dict]:
    grouped: Dict[tuple[int, str], List[dict]] = {}
    all_grouped: Dict[tuple[int, str], List[dict]] = {}
    for row in rows:
        key = (int(row["layer"]), str(row["head"]))
        all_grouped.setdefault(key, []).append(row)
        if math.isfinite(float(row["recovery_fraction"])):
            grouped.setdefault(key, []).append(row)
    summary = []
    for key, all_rows in sorted(all_grouped.items(), key=lambda item: (item[0][0], head_sort_key(item[0][1]))):
        layer_idx, head_label = key
        valid_rows = grouped.get(key, [])
        if not valid_rows:
            continue
        recoveries = [float(row["recovery_fraction"]) for row in valid_rows]
        deltas = [float(row["patched_delta_vs_corrupt_nll_sum"]) for row in valid_rows]
        gaps = [float(row["corrupt_minus_clean_nll_sum"]) for row in valid_rows]
        summary.append(
            {
                "layer": layer_idx,
                "head": head_label,
                "n_directions": len(valid_rows),
                "invalid_gap_directions": len(all_rows) - len(valid_rows),
                "helped_directions": sum(1 for delta in deltas if delta < 0),
                "mean_recovery_fraction": sum(recoveries) / len(recoveries),
                "median_recovery_fraction": statistics.median(recoveries),
                "mean_patch_delta_vs_corrupt_nll_sum": sum(deltas) / len(deltas),
                "sum_corrupt_minus_clean_nll_sum": sum(gaps),
                "sum_patch_delta_vs_corrupt_nll_sum": sum(deltas),
                "mean_patched_positions": sum(int(row["patched_positions"]) for row in valid_rows) / len(valid_rows),
            }
        )
    return summary


def write_report(out_dir: Path, args: argparse.Namespace, summary_rows: List[dict]) -> None:
    top = sorted(summary_rows, key=lambda row: float(row["mean_recovery_fraction"]), reverse=True)[:10]
    lines = [
        "# Generation Attention Head Patch Recovery",
        "",
        f"- Config: `{args.cfg}`",
        f"- Layer: `{args.layer}`",
        f"- Heads: `{args.heads}`",
        f"- Head mode: `{args.head_mode}`",
        f"- Patch scope: `{args.patch_scope}`",
        f"- Score scope: `{args.score_scope}`",
        f"- Target cache dir: `{args.target_cache_dir}`",
        "",
        "## Top Heads By Mean Recovery",
        "",
        "| rank | head | directions | helped | mean recovery | median recovery | sum delta NLL vs corrupt |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(top, start=1):
        lines.append(
            "| {rank} | {head} | {n_directions} | {helped_directions} | {mean_recovery_fraction:+.6f} | "
            "{median_recovery_fraction:+.6f} | {sum_patch_delta_vs_corrupt_nll_sum:+.6f} |".format(
                rank=rank,
                **row,
            )
        )
    (out_dir / "generation_attention_head_patch_recovery_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


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
    del vq_model
    model.eval()
    cfg.special_token_ids = {key: tokenizer.encode(value)[0] for key, value in cfg.special_tokens.items()}
    layer_idx = int(args.layer)
    if layer_idx < 0 or layer_idx >= len(model.model.layers):
        raise ValueError(f"layer out of range: {layer_idx}")
    num_heads = int(model.model.layers[layer_idx].self_attn.num_heads)
    heads = parse_heads(args.heads, num_heads)
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
                    layer_idx,
                    heads,
                    args.patch_scope,
                    args.score_scope,
                    args.head_mode,
                    args.max_target_tokens,
                    Path(args.target_cache_dir),
                )
            )

    write_csv_rows(rows, out_dir / "generation_attention_head_patch_recovery.csv")
    summary_rows = summarize(rows)
    if summary_rows:
        write_csv_rows(summary_rows, out_dir / "summary.csv")
    write_report(out_dir, args, summary_rows)
    metadata = {
        "cfg": args.cfg,
        "layer": args.layer,
        "heads": args.heads,
        "head_mode": args.head_mode,
        "patch_scope": args.patch_scope,
        "score_scope": args.score_scope,
        "target_cache_dir": args.target_cache_dir,
        "pair_id": args.pair_id,
        "max_pairs": args.max_pairs,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] generation attention head patch recovery saved to {out_dir}")


if __name__ == "__main__":
    main()
