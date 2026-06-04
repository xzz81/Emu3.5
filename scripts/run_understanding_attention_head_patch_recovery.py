#!/usr/bin/env python3
# Copyright 2026
# SPDX-License-Identifier: Apache-2.0

"""Clean/corrupt attention-head output patching for Emu3.5 hue understanding."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import statistics
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
    parser.add_argument("--layer", type=int, default=62)
    parser.add_argument("--heads", default="all", help="Comma-separated head ids, or 'all'.")
    parser.add_argument(
        "--head-mode",
        choices=["individual", "joint", "both"],
        default="individual",
        help="Patch each requested head separately, all requested heads jointly, or both.",
    )
    parser.add_argument("--max-pairs", type=int, default=2)
    parser.add_argument("--question-template", default="What color is the {shape}? Answer with one word.")
    parser.add_argument(
        "--patch-scope",
        choices=["post-image-text-prompt", "question-text-no-score", "score-positions"],
        default="post-image-text-prompt",
        help="Sequence positions where clean head outputs are patched into corrupt run.",
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


class CleanHeadOutputCollector:
    def __init__(self, model, layer_idx: int):
        self.model = model
        self.layer_idx = layer_idx
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
    def __init__(self, model, layer_idx: int, head_idx: int, clean_o_proj_input: torch.Tensor, position_mask: torch.Tensor):
        self.model = model
        self.layer_idx = layer_idx
        self.head_idx = head_idx
        self.clean_o_proj_input = clean_o_proj_input
        self.position_mask = position_mask
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
        start = self.head_idx * self.head_dim
        end = start + self.head_dim
        mask = self.position_mask[:seq_len].to(corrupt.device).to(corrupt.dtype).view(1, seq_len, 1)
        clean_slice = self.clean_o_proj_input[:, :seq_len, start:end].to(device=corrupt.device, dtype=corrupt.dtype)
        patched = corrupt.clone()
        patched[:, :seq_len, start:end] = torch.where(mask.bool(), clean_slice, patched[:, :seq_len, start:end])
        return (patched,) + tuple(inputs[1:])


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


def make_result_row(
    layer_idx,
    head_label,
    clean_row,
    corrupt_row,
    answer_text,
    clean,
    corrupt,
    patched,
    patch_scope,
    patch_mask,
):
    denom = corrupt["nll_sum"] - clean["nll_sum"]
    recovery = (corrupt["nll_sum"] - patched["nll_sum"]) / denom if abs(denom) > 1e-9 else float("nan")
    return {
        "layer": layer_idx,
        "head": head_label,
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


def run_direction(
    model,
    cfg,
    tokenizer,
    vq_model,
    clean_row,
    corrupt_row,
    layer_idx,
    heads,
    question_template,
    patch_scope,
    head_mode,
):
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

    collector = CleanHeadOutputCollector(model, layer_idx)
    collector.install()
    try:
        model.model(input_ids=clean_full, use_cache=False, return_dict=True)
    finally:
        collector.remove()
    if collector.clean_o_proj_input is None:
        raise RuntimeError(f"missing clean o_proj input for layer {layer_idx}")
    torch.cuda.empty_cache()

    patch_mask = build_patch_mask(corrupt_full, corrupt_prompt.shape[1], answer_ids.shape[1], patch_scope)
    rows = []
    if head_mode in ("individual", "both"):
        for head_idx in heads:
            patcher = AttentionHeadPatcher(model, layer_idx, head_idx, collector.clean_o_proj_input, patch_mask)
            patcher.install()
            try:
                patched = answer_nll(model, corrupt_prompt, answer_ids)
            finally:
                patcher.remove()
            torch.cuda.empty_cache()

            rows.append(
                make_result_row(
                    layer_idx,
                    str(head_idx),
                    clean_row,
                    corrupt_row,
                    answer_text,
                    clean,
                    corrupt,
                    patched,
                    patch_scope,
                    patch_mask,
                )
            )
    if head_mode in ("joint", "both"):
        patchers = [
            AttentionHeadPatcher(model, layer_idx, head_idx, collector.clean_o_proj_input, patch_mask)
            for head_idx in heads
        ]
        for patcher in patchers:
            patcher.install()
        try:
            patched = answer_nll(model, corrupt_prompt, answer_ids)
        finally:
            for patcher in patchers:
                patcher.remove()
        torch.cuda.empty_cache()

        rows.append(
            make_result_row(
                layer_idx,
                "+".join(str(head_idx) for head_idx in heads),
                clean_row,
                corrupt_row,
                answer_text,
                clean,
                corrupt,
                patched,
                patch_scope,
                patch_mask,
            )
        )
    return rows


def head_sort_key(head_label: str):
    return [int(item) for item in str(head_label).split("+")]


def summarize(rows: List[dict]) -> List[dict]:
    by_head: Dict[Tuple[int, str], List[dict]] = {}
    for row in rows:
        if math.isfinite(float(row["recovery_fraction"])):
            by_head.setdefault((int(row["layer"]), str(row["head"])), []).append(row)
    summary = []
    for (layer_idx, head_label), group_rows in sorted(
        by_head.items(), key=lambda item: (item[0][0], head_sort_key(item[0][1]))
    ):
        recoveries = [float(row["recovery_fraction"]) for row in group_rows]
        deltas = [float(row["patched_delta_vs_corrupt_nll_sum"]) for row in group_rows]
        gaps = [float(row["corrupt_minus_clean_nll_sum"]) for row in group_rows]
        summary.append(
            {
                "layer": layer_idx,
                "head": head_label,
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
    top = sorted(summary_rows, key=lambda row: float(row["mean_recovery_fraction"]), reverse=True)[:10]
    lines = [
        "# Stage 6 Report: Attention Head Output Patch Recovery",
        "",
        "## Purpose",
        "",
        "This experiment patches clean per-head attention outputs into corrupt-image hue QA runs.",
        "",
        "## Setup",
        "",
        f"- Layer: `{args.layer}`",
        f"- Heads: `{args.heads}`",
        f"- Head mode: `{args.head_mode}`",
        f"- Patch scope: `{args.patch_scope}`",
        f"- Max pairs: `{args.max_pairs}`",
        "",
        "Metric:",
        "",
        "```text",
        "recovery = (corrupt_nll - patched_nll) / (corrupt_nll - clean_nll)",
        "```",
        "",
        "## Top Heads By Mean Recovery",
        "",
        "| rank | head | directions | helped | mean recovery | median recovery | sum delta NLL vs corrupt |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(top, start=1):
        lines.append(
            "| {rank} | {head} | {n_directions} | {helped_directions} | {mean_recovery_fraction:+.6f} | "
            "{median_recovery_fraction:+.6f} | {sum_patch_delta_vs_corrupt_nll_sum:+.6f} |".format(
                rank=rank, **row
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Guide",
            "",
            "- Positive recovery means the clean head output helps recover the clean answer.",
            "- Compare summed or top-head recovery against full self-attention module recovery to judge whether a small head set explains the module effect.",
            "",
        ]
    )
    (out_dir / "understanding_attention_head_patch_recovery_report.md").write_text("\n".join(lines), encoding="utf-8")


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
    layer_idx = int(args.layer)
    if layer_idx < 0 or layer_idx >= len(model.model.layers):
        raise ValueError(f"layer out of range: {layer_idx}")
    num_heads = int(model.model.layers[layer_idx].self_attn.num_heads)
    heads = parse_heads(args.heads, num_heads)

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
                    layer_idx,
                    heads,
                    args.question_template,
                    args.patch_scope,
                    args.head_mode,
                )
            )

    write_csv_rows(rows, out_dir / "understanding_attention_head_patch_recovery.csv")
    summary_rows = summarize(rows)
    if summary_rows:
        write_csv_rows(summary_rows, out_dir / "summary.csv")
    write_report(out_dir, args, summary_rows)
    print(f"[INFO] understanding attention head patch recovery saved to {out_dir}")


if __name__ == "__main__":
    main()
