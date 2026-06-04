#!/usr/bin/env python3
# Copyright 2026
# SPDX-License-Identifier: Apache-2.0

"""Attention-edge knockout for Emu3.5 hue understanding."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Dict, List

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
    parser.add_argument("--layers", default="61,62", help="Comma-separated decoder layer ids, or 'all'.")
    parser.add_argument("--max-pairs", type=int, default=2)
    parser.add_argument("--question-template", default="What color is the {shape}? Answer with one word.")
    parser.add_argument(
        "--source-scope",
        choices=["visual-prompt", "eoi-prompt", "image-boundary", "image-prompt", "nonimage-prompt"],
        default="visual-prompt",
        help="Source key/value positions to block for answer score-position queries.",
    )
    parser.add_argument(
        "--target-scope",
        choices=["score-positions", "post-image-text-prompt", "question-text-no-score"],
        default="score-positions",
        help="Target query positions. For one-token answers, this is the final prompt position.",
    )
    parser.add_argument(
        "--knockout-mode",
        choices=["individual", "joint", "both"],
        default="individual",
        help="Run each layer separately, all requested layers jointly, or both.",
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


def build_query_mask(full_ids, prompt_len: int, answer_len: int, target_scope: str):
    mask = torch.zeros(full_ids.shape[1], dtype=torch.bool, device=full_ids.device)
    if target_scope == "score-positions":
        start = max(0, prompt_len - 1)
        mask[start : start + answer_len] = True
    elif target_scope in ("post-image-text-prompt", "question-text-no-score"):
        prompt_ids = full_ids[0, :prompt_len]
        eoi_positions = (prompt_ids == EOI).nonzero(as_tuple=False).flatten()
        if int(eoi_positions.numel()) > 0:
            start = int(eoi_positions[-1].item()) + 1
        else:
            start = 0
        post_image = torch.zeros_like(prompt_ids, dtype=torch.bool)
        post_image[start:prompt_len] = True
        text_like = prompt_ids < BOV
        special_image = (prompt_ids == BOI) | (prompt_ids == IMG) | (prompt_ids == EOI)
        mask[:prompt_len] = post_image & text_like & ~special_image
        if target_scope == "question-text-no-score":
            score_start = max(0, prompt_len - 1)
            mask[score_start:prompt_len] = False
    else:
        raise ValueError(f"unknown target_scope: {target_scope}")
    return mask


def build_source_mask(full_ids, prompt_len: int, source_scope: str):
    mask = torch.zeros(full_ids.shape[1], dtype=torch.bool, device=full_ids.device)
    prompt_ids = full_ids[0, :prompt_len]
    if source_scope == "visual-prompt":
        mask[:prompt_len] = prompt_ids >= BOV
    elif source_scope == "eoi-prompt":
        mask[:prompt_len] = prompt_ids == EOI
    elif source_scope == "image-boundary":
        mask[:prompt_len] = (prompt_ids == BOI) | (prompt_ids == IMG) | (prompt_ids == EOI)
    elif source_scope == "image-prompt":
        mask[:prompt_len] = (prompt_ids >= BOV) | (prompt_ids == BOI) | (prompt_ids == IMG) | (prompt_ids == EOI)
    elif source_scope == "nonimage-prompt":
        image_like = (prompt_ids >= BOV) | (prompt_ids == BOI) | (prompt_ids == IMG) | (prompt_ids == EOI)
        mask[:prompt_len] = ~image_like
    else:
        raise ValueError(f"unknown source_scope: {source_scope}")
    return mask


class AttentionEdgeKnockout:
    def __init__(self, model, layer_idx: int, query_mask: torch.Tensor, source_mask: torch.Tensor):
        self.model = model
        self.layer_idx = layer_idx
        self.query_mask = query_mask
        self.source_mask = source_mask
        self.handle = None

    def install(self):
        attn = self.model.model.layers[self.layer_idx].self_attn
        self.handle = attn.register_forward_pre_hook(self._hook, with_kwargs=True)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def _hook(self, _module, args, kwargs):
        attention_mask = kwargs.get("attention_mask")
        if attention_mask is None:
            raise RuntimeError("attention knockout requires a 4D attention_mask; force eager attention")
        if attention_mask.dim() != 4:
            raise RuntimeError(f"expected 4D attention_mask, got shape {tuple(attention_mask.shape)}")
        q_len = attention_mask.shape[-2]
        kv_len = attention_mask.shape[-1]
        seq_len = min(q_len, kv_len, self.query_mask.numel(), self.source_mask.numel())
        query = self.query_mask[:seq_len].to(attention_mask.device)
        source = self.source_mask[:seq_len].to(attention_mask.device)
        if not bool(query.any()) or not bool(source.any()):
            return args, kwargs
        blocked = query.view(1, 1, seq_len, 1) & source.view(1, 1, 1, seq_len)
        patched_mask = attention_mask.clone()
        min_value = torch.finfo(patched_mask.dtype).min
        patched_mask[:, :, :seq_len, :seq_len] = torch.where(
            blocked,
            torch.full((), min_value, dtype=patched_mask.dtype, device=patched_mask.device),
            patched_mask[:, :, :seq_len, :seq_len],
        )
        kwargs = dict(kwargs)
        kwargs["attention_mask"] = patched_mask
        return args, kwargs


@torch.no_grad()
def answer_nll_with_knockout(model, prompt_ids, answer_ids, layers, query_mask, source_mask):
    knockouts = [AttentionEdgeKnockout(model, layer_idx, query_mask, source_mask) for layer_idx in layers]
    for knockout in knockouts:
        knockout.install()
    try:
        return answer_nll(model, prompt_ids, answer_ids)
    finally:
        for knockout in knockouts:
            knockout.remove()


def make_result_row(
    layer_label,
    clean_row,
    corrupt_row,
    answer_text,
    clean,
    corrupt,
    knocked,
    source_scope,
    target_scope,
    source_mask,
    query_mask,
):
    corrupt_gap = corrupt["nll_sum"] - clean["nll_sum"]
    delta = knocked["nll_sum"] - clean["nll_sum"]
    harm_fraction = delta / corrupt_gap if abs(corrupt_gap) > 1e-9 else float("nan")
    return {
        "layer": layer_label,
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
        "knockout_nll_sum": knocked["nll_sum"],
        "clean_nll_mean": clean["nll_mean"],
        "corrupt_nll_mean": corrupt["nll_mean"],
        "knockout_nll_mean": knocked["nll_mean"],
        "corrupt_minus_clean_nll_sum": corrupt_gap,
        "knockout_delta_vs_clean_nll_sum": delta,
        "harm_fraction_of_corrupt_gap": harm_fraction,
        "source_scope": source_scope,
        "target_scope": target_scope,
        "source_positions": int(source_mask.sum().item()),
        "target_positions": int(query_mask.sum().item()),
    }


def run_direction(
    model,
    cfg,
    tokenizer,
    vq_model,
    clean_row,
    corrupt_row,
    layers,
    question_template,
    source_scope,
    target_scope,
    knockout_mode,
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

    full_ids = clean["full_ids"]
    query_mask = build_query_mask(full_ids, clean_prompt.shape[1], answer_ids.shape[1], target_scope)
    source_mask = build_source_mask(full_ids, clean_prompt.shape[1], source_scope)

    rows = []
    if knockout_mode in ("individual", "both"):
        for layer_idx in layers:
            knocked = answer_nll_with_knockout(model, clean_prompt, answer_ids, [layer_idx], query_mask, source_mask)
            torch.cuda.empty_cache()
            rows.append(
                make_result_row(
                    str(layer_idx),
                    clean_row,
                    corrupt_row,
                    answer_text,
                    clean,
                    corrupt,
                    knocked,
                    source_scope,
                    target_scope,
                    source_mask,
                    query_mask,
                )
            )
    if knockout_mode in ("joint", "both"):
        knocked = answer_nll_with_knockout(model, clean_prompt, answer_ids, layers, query_mask, source_mask)
        torch.cuda.empty_cache()
        rows.append(
            make_result_row(
                "+".join(str(layer_idx) for layer_idx in layers),
                clean_row,
                corrupt_row,
                answer_text,
                clean,
                corrupt,
                knocked,
                source_scope,
                target_scope,
                source_mask,
                query_mask,
            )
        )
    return rows


def summarize(rows: List[dict]) -> List[dict]:
    by_layer: Dict[str, List[dict]] = {}
    for row in rows:
        if math.isfinite(float(row["harm_fraction_of_corrupt_gap"])):
            by_layer.setdefault(str(row["layer"]), []).append(row)
    summary = []
    for layer_label, layer_rows in sorted(by_layer.items(), key=lambda item: [int(x) for x in item[0].split("+")]):
        harms = [float(row["harm_fraction_of_corrupt_gap"]) for row in layer_rows]
        deltas = [float(row["knockout_delta_vs_clean_nll_sum"]) for row in layer_rows]
        gaps = [float(row["corrupt_minus_clean_nll_sum"]) for row in layer_rows]
        summary.append(
            {
                "layer": layer_label,
                "n_directions": len(layer_rows),
                "hurt_directions": sum(1 for delta in deltas if delta > 0),
                "mean_harm_fraction_of_corrupt_gap": sum(harms) / len(harms),
                "median_harm_fraction_of_corrupt_gap": statistics.median(harms),
                "mean_knockout_delta_vs_clean_nll_sum": sum(deltas) / len(deltas),
                "sum_corrupt_minus_clean_nll_sum": sum(gaps),
                "sum_knockout_delta_vs_clean_nll_sum": sum(deltas),
                "mean_source_positions": sum(int(row["source_positions"]) for row in layer_rows) / len(layer_rows),
                "mean_target_positions": sum(int(row["target_positions"]) for row in layer_rows) / len(layer_rows),
            }
        )
    return summary


def write_report(out_dir: Path, args: argparse.Namespace, summary_rows: List[dict]) -> None:
    lines = [
        "# Stage 3 Report: Understanding Attention Edge Knockout",
        "",
        "## Purpose",
        "",
        "This experiment blocks selected attention edges in a clean-image hue QA run and measures how much the clean answer likelihood is damaged.",
        "",
        "## Setup",
        "",
        f"- Layers: `{args.layers}`",
        f"- Source scope: `{args.source_scope}`",
        f"- Target scope: `{args.target_scope}`",
        f"- Knockout mode: `{args.knockout_mode}`",
        f"- Max pairs: `{args.max_pairs}`",
        "",
        "Metric:",
        "",
        "```text",
        "harm_fraction = (knockout_nll - clean_nll) / (corrupt_nll - clean_nll)",
        "```",
        "",
        "Positive values mean the knockout made the clean answer less likely. A value near 1 means the damage is comparable to replacing the clean image with the corrupt image.",
        "",
        "## Aggregate Results",
        "",
        "| layer | directions | hurt | mean harm | median harm | sum delta NLL vs clean |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {layer} | {n_directions} | {hurt_directions} | {mean_harm_fraction_of_corrupt_gap:+.6f} | "
            "{median_harm_fraction_of_corrupt_gap:+.6f} | {sum_knockout_delta_vs_clean_nll_sum:+.6f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Interpretation Guide",
            "",
            "- High positive harm supports a causal attention edge into the answer scoring state.",
            "- Near-zero harm suggests the edge set is not necessary at the tested layer, or information arrives through other paths.",
            "- Negative harm means the knockout made the clean answer more likely, so it is not a damaging gate under this metric.",
            "",
        ]
    )
    (out_dir / "understanding_attention_knockout_report.md").write_text("\n".join(lines), encoding="utf-8")


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
                    args.question_template,
                    args.source_scope,
                    args.target_scope,
                    args.knockout_mode,
                )
            )

    write_csv_rows(rows, out_dir / "understanding_attention_knockout.csv")
    summary_rows = summarize(rows)
    if summary_rows:
        write_csv_rows(summary_rows, out_dir / "summary.csv")
    write_report(out_dir, args, summary_rows)
    print(f"[INFO] understanding attention knockout saved to {out_dir}")


if __name__ == "__main__":
    main()
